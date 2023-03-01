import torch
import torch.nn as nn
import torchvision.transforms as transforms

import numpy as np
import json

from utils import NpEncoder
from verinet_line_segment_verification import verinet_verify_line_segment, augment_network_for_ls
# from verinet import Status
from verinet.verification.verifier_util import Status


def get_specification(dparams, test_attribute, target_attributes):
    """ Specifications available for different datasets:
        - MNIST, FashionMNIST, TrafficSignsDynSynth:
          -- for every image I in dataset, verification for [I, transform(I)] for all transform in test_transforms (till the range and in steps specified)
        - CelebA:
          -- invariance against a test_attribute can be verified
          -- invariance to head tilt change can be verified
        - Object10 (any transform/attribute in paths datasets):
          -- need left attribute (keep close to characteristic image) and right attributes, so verification for change from left to right transform
        In each case, verification for eps in conditional dims."""
    if dparams['dataset'] in ['MNIST', 'FashionMNIST', 'TrafficSignsDynSynth']:
        # keys of test_transforms should correspond to some conditional dim
        test_transforms = {"left_shear": {"range": 1./3, "steps": 3},
                           "right_shear": {"range": 1./3, "steps": 3},
                           "top_shear": {"range": 1./3, "steps": 3},
                           "bottom_shear": {"range": 1./3, "steps": 3},
                           "left_rotate": {"range": 30, "steps": 4},
                           "right_rotate": {"range": 30, "steps": 4},
                           "towards": {"range": 0.3, "steps": 4},
                           "far": {"range": 0.3, "steps": 4}}
        test_transform_names = [f'{k}_{t:.2f}\n' for k, v in test_transforms.items() for t in np.linspace(0, v['range'], v['steps']+1)[1:]]
        if type(dparams['conditional']) is not dict:
            dparams['conditional'] = {}
        dparams['conditional']['transforms'] = [test_transforms]
    elif dparams['dataset'] in ['CelebA', 'Fairfaces']:
        test_transform_names = [f'Inv. to {test_attribute}']
    elif dparams['dataset'] in ['Objects10_3Dpose', 'Runways']:
        test_transform_names = [f'{test_attribute} to {at}\n' for at in target_attributes]
    else:
        raise NotImplementedError(f'Verification setup for {dparams["dataset"]} not implemented')
    print("Verifying against the following:\n", test_transform_names)
    return test_transform_names


def get_specification_inputs_object3d(batch, model, device, dataloader, args):
    """Returns zs, zs_mu_sigma, y for a batch (two or more relevant images) for verification, x_pair for visualization"""
    assert dataloader.dataset_name == "Objects10_3Dpose", print("Wrong function for {dataloader.dataset_name}")
    x_in, y, y_attrs = batch
    dataset = dataloader.dataset
    x = dataset.get_image(x_in, args['test_attribute'])
    if x is None:
        return None, None, None
    assert x.dim() == 3, print(x.shape)
    y = y[0]

    model = model.to(device)  # why
    pred_logits = model(x.unsqueeze(0).to(device), only_cla=True)
    _, pred_label = torch.max(pred_logits, axis=1)
    if pred_label != y:
        return None, y.item(), y_attrs

    x_transformed_images, y_attrs_ = [], []
    for at in args['target_attributes']:
        x_transformed_image = dataset.get_image(x_in, at)  # returns transformed images
        if x_transformed_image is None:
            return None, None, y_attrs
        x_transformed_images.append(x_transformed_image)
        y_attrs_.append(dataset.get_attribute_from_path_(at))
    x_set = torch.stack([x] + x_transformed_images, dim=0)
    assert x_set.dim() == 4, print(x_set.shape)
    return x_set, y.item(), np.array(y_attrs_)


def get_specification_inputs(batch, model, device, dataloader, args):
    """Returns zs, zs_mu_sigma, y for a batch (two or more relevant images) for verification, x_pair for visualization"""
    x_in, y, y_attrs = batch
    dataset, dataset_name = dataloader.dataset, dataloader.dataset_name
    x, y = x_in[0].unsqueeze(0), y[0]
    assert x.dim() == 4, print(x.shape)

    model = model.to(device)  # why
    pred_logits = model(x.to(device), only_cla=True)
    _, pred_label = torch.max(pred_logits, axis=1)
    if pred_label != y:
        return None, y.item(), y_attrs

    x_set = x_in  # for most datasets, the relevant setpoints come from dataloading itself
    if dataset_name in ['CelebA', 'Fairfaces']:
        if args['flip_head']:
            # just flip the image, but verify for this image only if it belongs to the test attribute
            if args['test_attribute'] != "" and not dataset.image_has_attribute(y_attrs[0], args['test_attribute']):  # change of logic!!
                return None, None, y_attrs
            x_set = torch.stack([x.squeeze(0), dataset.transform_tensor_image(x, transforms.RandomHorizontalFlip(p=1))], dim=0)
        else:
            # for each input, find set of images that differ only in single attribute or should definitely belong to the same class
            x_set = dataset.get_same_class_attribute_change_pair(x, y, y_attrs[0], args['test_attribute'])
    assert x_set.dim() == 4, print(x_set.shape)
    return x_set, y, y_attrs


def pgd_on_ls(model, device, label, ls_endpoints, ver_results, index, feat_space_spec, pot_cex_alpha=None):
    prepend_layers = [layer.float() for layer in augment_network_for_ls(ls_endpoints)]
    prepend_net = nn.Sequential(*prepend_layers).to(device)
    loss_criterion = nn.CrossEntropyLoss()
    adv_alpha = pot_cex_alpha or torch.zeros((1, 1))
    adv_alpha = adv_alpha.float().to(device)
    adv_alpha = torch.autograd.Variable(adv_alpha, requires_grad=True)
    alpha, num_iters = 1e-3, 500
    for itr in range(num_iters):
        if pot_cex_alpha is not None:
            # pass through decoder and classifier for finetuning
            if feat_space_spec:
                adv_z, _ = model.encoding_head(prepend_net(adv_alpha))
            else:
                adv_z = model.encoding_head.construct_z(prepend_net(adv_alpha))
            out_logits = model.classification_head(model.fdn(model.decoder(adv_z)))
        else:
            # pass through classification head only
            if feat_space_spec:
                adv_feat = prepend_net(adv_alpha)
            else:
                adv_feat = nn.Sequential(*model.encoding_head.get_inverse_layers())(prepend_net(adv_alpha).double()).float()
            out_logits = model.classification_head(adv_feat)

        _, predLabel = torch.max(out_logits, axis=1)
        if predLabel != label:
            print("UNSAFE")
            if pot_cex_alpha is None:
                ver_results['counterexs_PGD_iters'].append(itr+1)
            else:
                ver_results['counterexs_verinet_PGD_iters'].append(itr+1)
                ver_results['counterexs_true_after_traversal'][index][label] += 1
            return prepend_net(adv_alpha)
        loss = loss_criterion(out_logits, label.unsqueeze(0).to(device))
        loss.backward()
        adv_alpha.data = (adv_alpha + alpha*adv_alpha.grad.detach().sign())
        adv_alpha.data = torch.clamp(adv_alpha.data, min=0, max=1)
        adv_alpha.grad.zero_()
        # project it within constraints, i.e., on the line segment
    return None


# internals
def verinet_verify_for_zs(x, y, model, ver_results, image_results, num_classes, device, ndims_to_check, dparams, feat_space_spec=False):

    def update_feat_stats(model, y, ver_status, ver_time, ceg, index):
        def feat_recons_pred(model, y, x_feat):
            model, x_feat = model.to(device), x_feat.to(device)
            z, _ = model.encoding_head(x_feat)
            y_logits = model(model.decoder(z), only_cla=True)
            _, predLabel = torch.max(y_logits, axis=1)
            return predLabel == y

        if ver_status == Status.Safe:
            ver_results['feat_ver_times']['safe'].append(ver_time)
            ver_results['feat_verified'][index][y] += 1
            return False
        if ver_status == Status.Unsafe:
            ver_results['feat_ver_times']['unsafe'].append(ver_time)
            ceg_type = "false" if feat_recons_pred(model, y, ceg) else "true"
            ver_results[f"feat_counterexs_{ceg_type}"][index][y] += 1
            return ceg_type == "false"
        ver_results['feat_undecided'][index][y] += 1
        ver_results['feat_ver_times']['undecided'].append(ver_time)
        return True

    def update_latent_stats(model, y, z_mu_lvar, z, ver_status, ver_time, ceg, bdeltas, index):
        def latent_vector_recons_diff_pred(model, y, z_mu_lvar, z=None):
            model, z_mu_lvar = model.to(device), z_mu_lvar.unsqueeze(0).to(device)
            with torch.no_grad():
                if z is None:
                    z = model.encoding_head.construct_z(z_mu_lvar, add_noise=False)
                y_logits, _, recons_z_mu_lvar = model(model.decoder(z.unsqueeze(0)), only_gen_z=True)
            _, predLabel = torch.max(y_logits, axis=1)
            return torch.abs(z_mu_lvar - recons_z_mu_lvar).squeeze(0).detach().to("cpu"), predLabel == y

        if ver_status == Status.Safe:
            ver_results['ver_times']['safe'].append(ver_time)
            bdeltas['verified'].append(latent_vector_recons_diff_pred(model, y, z_mu_lvar, z)[0])
            ver_results['verified'][index][y] += 1
            return False
        if ver_status == Status.Unsafe:
            ver_results['ver_times']['unsafe'].append(ver_time)
            delta, cla_correct = latent_vector_recons_diff_pred(model, y, ceg.squeeze(0))
            ceg_type = "false" if cla_correct else "true"
            bdeltas[f"counterexs_{ceg_type}"].append(delta)
            ver_results[f"counterexs_{ceg_type}"][index][y] += 1
            return cla_correct  # ceg is false
        ver_results['ver_times']['undecided'].append(ver_time)
        ver_results['undecided'][index][y] += 1
        return True  # try to find ceg with PGD

    with torch.no_grad():
        xs_feat, _, zs, zs_mu_lvar, _ = model(x.to(device))  # x shape: [N, 3, w, h]

    batch_deltas = {'verified': [], 'counterexs_true': [], 'counterexs_false': [], 'counterexs_true_after_traversal': []}
    ver_layers = model.classification_head.get_layers() if feat_space_spec else model.verification_model_layers()
    print(ver_layers)
    # to not continue checking transform after smaller transform fails
    completed_transform = None
    if dparams['dataset'] in ['MNIST', 'FashionMNIST', 'TrafficSignsDynSynth']:
        transforms = []
        for k, v in dparams['conditional']['transforms'][0].items():
            transforms += [k]*v['steps']

    points = xs_feat if feat_space_spec else zs_mu_lvar
    for pi, point in enumerate(points[1:]):
        if completed_transform is not None and transforms[pi] == completed_transform:
            continue
        ls_endpts = torch.flatten(torch.stack((points[0], point), dim=0), start_dim=1).double().to("cpu")  # [BS, ld]
        assert ls_endpts.dim() == 2, print("LS endpoints shape:", ls_endpts.shape)

        ls_ceg = pgd_on_ls(model, device, y, ls_endpts, ver_results, pi, feat_space_spec)  # prelim ceg check
        print("Prelim PGD safe?: ", ls_ceg is None)
        ver_status, ls_ceg, ceg_is_from_PGD, ver_time, ls_ceg_alpha = verinet_verify_line_segment(ver_layers, ls_endpts, y, num_classes, 30)
        print("Verinet ver status", ver_status)
        if ver_status == Status.Unsafe:
            assert ls_ceg is not None, print("Status unsafe, but ls_ceg is None.")
            if ceg_is_from_PGD is not None:
                ver_results['counterexs_wo_pgd'][pi][y] += (1-int(ceg_is_from_PGD))
            # check passes, can verify but not necessary to add in normal runs
            # ceg_hat = nn.Sequential(*model.encoding_head.get_inverse_layers())(ceg.double().to(device)).to("cpu")
            # assert_pt_on_line_segment(ceg_hat[0], x_feats_ls)

        if feat_space_spec:
            ceg_is_false = update_feat_stats(model, y, ver_status, ver_time, ls_ceg, pi)
        else:
            ceg_is_false = update_latent_stats(model, y, point, zs[pi], ver_status, ver_time, ls_ceg, batch_deltas, pi)

        if ver_status == Status.Safe:
            image_results['safe'].append(torch.stack((x[0], x[pi + 1])))
        elif ver_status == Status.Unsafe:
            image_results['unsafe'].append(torch.stack((x[0], x[pi + 1])))
            if ceg_is_false:
                ls_ceg_ = pgd_on_ls(model, device, y, ls_endpts, ver_results, pi, feat_space_spec, ls_ceg_alpha)
                if ls_ceg_ is not None:
                    ls_ceg = ls_ceg_
            if feat_space_spec:
                _, ls_ceg = model.encoding_head.to(device)(ls_ceg.to(device))  # now this ceg is enc_out_ceg
            image_results['ceg_z'].append(model.encoding_head.construct_z(ls_ceg, add_noise=False))
        if ver_status != Status.Safe and dparams['dataset'] in ['MNIST', 'FashionMNIST', 'TrafficSignsDynSynth']:
            completed_transform = transforms[pi]

    # all transforms done :D
    for k, v in batch_deltas.items():
        if len(v) == 0:
            continue
        v_tensor = torch.stack((v), dim=0)
        for di in range(ndims_to_check):
            dim_ver_delta, _ = np.histogram(v_tensor[:, di], bins=np.linspace(0, 3.5, 11))
            if ver_results['deltas'][k][di] is None:
                ver_results['deltas'][k][di] = dim_ver_delta
            else:
                ver_results['deltas'][k][di] += dim_ver_delta


def save_ver_results(dataset_name, model_path, results_file, ver_results, num_tested_images, test_transform_names=''):
    ver_res_str = f'Verification results for {num_tested_images} in {dataset_name}'
    if test_transform_names:
        ver_res_str += ' against {test_transform_names}'
    with open(results_file + ".txt", "w") as rf:
        rf.write(model_path + '\n' + ver_res_str + '\n')
        rf.write(json.dumps(ver_results, indent=4, cls=NpEncoder))


def prepare_query1_results_dict(nclasses, num_transforms, ndims_to_check):
    results_dict = {'total': [0]*nclasses,
                    'tested': [0]*nclasses,
                    'ver_times': {'safe': [], 'unsafe': [], 'undecided': []},
                    'counterexs_PGD_iters': [],
                    'counterexs_verinet_PGD_iters': [],
                    # class wise verification stats for every transform
                    'verified': [[0]*nclasses for j in range(num_transforms)],
                    'undecided': [[0]*nclasses for j in range(num_transforms)],
                    'counterexs_wo_pgd': [[0]*nclasses for j in range(num_transforms)],
                    'counterexs_true': [[0]*nclasses for j in range(num_transforms)],
                    'counterexs_false': [[0]*nclasses for j in range(num_transforms)],
                    'counterexs_true_after_traversal': [[0]*nclasses for j in range(num_transforms)],
                    # latent dimension wise stats
                    'deltas': {'verified': [None] * ndims_to_check,
                               'counterexs_wo_PGD': [None] * ndims_to_check,
                               'counterexs_true': [None] * ndims_to_check,
                               'counterexs_false': [None] * ndims_to_check},
                    }
    return results_dict

import torch
import torchvision.transforms as transforms
torch.cuda.empty_cache()

import random
import sys
sys.path.insert(0, '..')

from notebook_utils import get_classifier, get_encoder, get_decoder, Encoder, GenModel, build_srvp_pipeline, test_cla

from dataloader import CustomDataloader
from verinet_line_segment_verification import verinet_verify_line_segment, augment_network_for_ls
from verify_utils import pgd_on_ls
from verinet.verification.verifier_util import Status

# copy/change from the jupyter notebook when necessary
input_shape = [3, 128, 128]  # [3, 64, 64]
batch_size = 64
batch_input_shape = [batch_size, *input_shape]
classes = ["Male"]
num_classes = len(classes) + 1
latent_dims = 64  # 32
exp_disc = "baseline"


def pgd_ls_attack(model, cla, device, label, ls_endpoints):
    prepend_layers = [layer.float() for layer in augment_network_for_ls(ls_endpoints)]
    prepend_net = torch.nn.Sequential(*prepend_layers).to(device)
    loss_criterion = torch.nn.CrossEntropyLoss()
    adv_alpha = torch.zeros((1, 1)).float().to(device)
    adv_alpha = torch.autograd.Variable(adv_alpha, requires_grad=True)
    alpha, num_iters = 2e-3, 1000
    for itr in range(num_iters):
        out_logits = cla(model.dec(prepend_net(adv_alpha)))
        _, predLabel = torch.max(out_logits, axis=1)
        if predLabel != label:
            return True
        loss = loss_criterion(out_logits, label.unsqueeze(0).to(device))
        loss.backward()
        adv_alpha.data = (adv_alpha.data + alpha*adv_alpha.grad.detach().sign())
        # project it within constraints, i.e., on the line segment
        adv_alpha.data = torch.clamp(adv_alpha.data, min=0, max=1)
        adv_alpha.grad.zero_()
    return False


def verify_SRVP(srvp, srvp_ver_layers, x_set, y, device, sver_results):
    with torch.no_grad():
        pred_logits, _, zs_mu_lvar, x_hat = srvp.to(device)(x_set)
        _, pred_label = torch.max(pred_logits, axis=1)
        if (pred_label != y).any():
            return
        sver_results["std_acc"] += 1
    for i, z_mu_lvar in enumerate(zs_mu_lvar[1:]):
        enc_outs_ls = torch.stack((zs_mu_lvar[0], z_mu_lvar), dim=0).double().to("cpu")  # [BS, ld]
        assert enc_outs_ls.dim() == 2, print(enc_outs_ls.shape)
        # prelim PGD
        pgd_vanilla_found = pgd_on_ls(srvp, device, y, enc_outs_ls, sver_results, 0, feat_space_spec=False,
                                      pot_cex_alpha=None)
        ls_ver_status, _, _, ver_time, ceg_alpha, bounds_summary, mem_bytes = verinet_verify_line_segment(
            srvp_ver_layers, enc_outs_ls, y, num_classes, timeout_s=30)
        print("SRVP (PGD found):", pgd_vanilla_found is not None, ls_ver_status)
        if ls_ver_status in [Status.Safe, Status.Unsafe]:
            sver_results["voutcomes"][ls_ver_status] += 1
            sver_results["vtimes"][ls_ver_status].append(ver_time)
        else:
            sver_results["voutcomes"]["undec"] += 1
            sver_results["vtimes"]["undec"].append(ver_time)
        sver_results["mem"] = max(sver_results["mem"], mem_bytes)
        sver_results["recons"].append(torch.nn.MSELoss()(x_hat.to(device), x_set).item())
        if bounds_summary:
            sver_results["bounds"].append(bounds_summary[1])
        if ls_ver_status == Status.Unsafe:
            # guided PGD on failure, updates sver_results counterexs_* fields
            pgd_on_ls(srvp, device, y, enc_outs_ls, sver_results, 0, feat_space_spec=False,
                      pot_cex_alpha=ceg_alpha)


def verify_EDC(edc, edc_ver_layers, cla, x_set, y, ever_results):
    with torch.no_grad():
        zs, _, x_hat = edc.to(device)(x_set)
    pgd_vanilla_found = pgd_ls_attack(edc, cla, device, y, zs)
    ever_results['PGD_attacks'] += pgd_vanilla_found
    for i, z in enumerate(zs[1:]):
        enc_outs_ls = torch.stack((zs[0], z), dim=0).double().to("cpu")  # [BS, ld]
        assert enc_outs_ls.dim() == 2, print(enc_outs_ls.shape)
        ls_ver_status, _, _, ver_time, _, bounds_summary, mem_bytes = verinet_verify_line_segment(edc_ver_layers,
                                                                                                  enc_outs_ls, y,
                                                                                                  num_classes,
                                                                                                  timeout_s=60)
        print("EDC (PGD found):", pgd_vanilla_found, ls_ver_status)
        if ls_ver_status in [Status.Safe, Status.Unsafe]:
            ever_results["voutcomes"][ls_ver_status] += 1
            ever_results["vtimes"][ls_ver_status].append(ver_time)
        else:
            ever_results["voutcomes"]["undec"] += 1
            ever_results["vtimes"]["undec"].append(ver_time)
        ever_results["mem"] = max(ever_results["mem"], mem_bytes)
        ever_results["recons"].append(torch.nn.MSELoss()(x_hat.to(device), x_set).item())
        if bounds_summary:
            ever_results["bounds"].append(bounds_summary[1])


def SRVP_results(num_classes):
    return {"std_acc": 0,
            "voutcomes": {Status.Safe: 0, Status.Unsafe: 0, 'undec': 0},
            "vtimes": {Status.Safe: [], Status.Unsafe: [], 'undec': []},
            "recons": [],
            "bounds": [],
            "mem": 0,
            "counterexs_PGD_iters": [],
            "counterexs_verinet_PGD_iters": [],
            "counterexs_true_after_traversal": [[0] * num_classes for j in range(1)]}


def EDC_results():
    return {"voutcomes": {Status.Safe: 0, Status.Unsafe: 0, 'undec': 0},
            "vtimes": {Status.Safe: [], Status.Unsafe: [], 'undec': []},
            "recons": [],
            "bounds": [],
            "mem": 0,
            "PGD_attacks": 0}


# verify multiple EDC (dec tiny, small, deep, resnet) and SRVP (32, 64, 192, 392) against the same queries
def verify_pipelines(cla_type):
    cla_path = f"./models/classifiers_{input_shape[-1]}/{cla_type}_{classes[0]}.tar"
    cla = get_classifier(cla_type, latent_dims, num_classes, device, cla_path)

    # get the EDCs
    edc_dir = f"./models/gen_models_{input_shape[-1]}/EDC_{latent_dims}/{exp_disc}"
    dec_types = ["tiny", "small", "deeper"] # ["resnet"] #
    edc_gens = {}
    for edc_dec_type in dec_types:
        dec = get_decoder(edc_dec_type, input_shape, latent_dims)
        edc_gens[edc_dec_type] = GenModel(Encoder(get_encoder("big", input_shape, latent_dims), device),
                                          dec,
                                          device,
                                          f"{edc_dir}/enc_big_edc_{edc_dec_type}.tar")
        edc_ver_layers = list(edc_gens[edc_dec_type].dec.children()) + list(cla.children())
        print("EDC layers:", edc_ver_layers)

    # get the SRVPs
    srvp_dir = f"./models/SRVP_{input_shape[-1]}/cla_{cla_type}"
    srvps = {64: None, 192: None, 392: None}
    for ld in [64, 192, 392]:
        scla = get_classifier(cla_type, latent_dims, num_classes, device, cla_path)
        srvp_path = f"{srvp_dir}/ld{ld}.tar"
        srvps[ld] = build_srvp_pipeline(device, scla, input_shape, ld, "deeper", srvp_path)
        srvp_ver_layers = [srvps[ld][1], *srvps[ld][0].classification_head_layers]
        print("SRVP layers:", srvp_ver_layers)

    dparams = {"input_shape": input_shape, "classes": classes, "conditional": {"attrs": [["Smiling"]]},
               "data_balance_method": [], "conditional_loss_fn": ["CE"], "dataset": "CelebA"}
    dataloader = CustomDataloader(dparams, apply_random_transforms=False)
    dataset, dataset_name = dataloader.get_dataset_params(["dataset", "dataset_name"])
    train_dl, _, test_dl = dataloader.get_data(1)
    mean_error = test_cla(cla, train_dl, device, dbatcher=True)
    print("CLA's mean accuracy on training dataset:", 1 - mean_error)

    # args = {"test_attribute": "Smiling", "flip_head": True}
    ver_results = {'total': 0, 'EDCs_std_cor': 0}
    for dectype in dec_types:
        ver_results[f"EDC_{dectype}"] = EDC_results()
    for ld in srvps:
        ver_results[f"SRVP_{ld}"] = SRVP_results(num_classes)
    for bi, (x, y, y_attrs) in enumerate(train_dl.get_batch("cpu")):
        ver_results['total'] += 1
        x, y = x[0].float(), 1 - y[0]
        pred_logits = cla(x.unsqueeze(0).to(device))
        _, pred_label = torch.max(pred_logits, axis=1)
        if pred_label != y:
            continue
        ver_results['EDCs_std_cor'] += 1

        x_set = torch.stack([x.squeeze(0), dataset.transform_tensor_image(x, transforms.RandomHorizontalFlip(p=1))], dim=0)
        x_set = x_set.to(device)

        for dectype, edc_gen in edc_gens.items():
            edc_ver_layers = list(edc_gen.dec.children())[:-1] + list(cla.children())
            verify_EDC(edc_gen, edc_ver_layers, cla, x_set, y, ver_results[f"EDC_{dectype}"])
        for sld, srvp_gen in srvps.items():
            srvp_ver_layers = [srvp_gen[1], *srvp_gen[0].classification_head_layers]
            verify_SRVP(srvp_gen[0], srvp_ver_layers, x_set, y, device, ver_results[f"SRVP_{sld}"])  # srvp latent_dims

        if bi % 5 == 0:
            print(ver_results)
            print("\n\n")


if __name__ == "__main__":
    # set random seed for reproducibility.
    seed = 1123
    random.seed(seed)
    torch.manual_seed(seed)
    print("Random Seed: ", seed)

    # use GPU if available.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device, "will be used.")

    verify_pipelines("deeper")

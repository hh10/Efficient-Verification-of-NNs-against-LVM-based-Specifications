# Scripts trains a Discriminator plus Classifier to classify real vs fake images and classes belonging to a dataset:
# produces specification inputs as in verify.py, passes n interps from the decoder to see if they are classified correctly plus real/fake
# explores the conditional dims in steps to check the same thing
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from dataloader import CustomDataloader, DataBatcher
from utils import load_params, load_model
from verify_utils import get_specification_inputs, get_specification
from models_impl.generic_nets import conv_Nlayer_downscalar
from model import VaeClassifier

import numpy as np
import argparse
import random
import os
import enlighten
from datetime import datetime
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 300  # noqa: E702

# set random seed for reproducibility.
seed = 1123
random.seed(seed)
torch.manual_seed(seed)
print("Random Seed: ", seed)

# use GPU if available.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device, "will be used.")

DEFAULTS = {"FashionMNIST": {"input_shape": [1, 28, 28], "conditional": {"transforms": [["left_shear", "right_shear", "left_rotate", "right_rotate", "towards", "far"]]}},
            "CelebA": {"input_shape": [1, 28, 28], "conditional": {"transforms": [["left_shear", "right_shear", "left_rotate", "right_rotate", "towards", "far"]]}},
            "TrafficSignsDynSynth": {"input_shape": [1, 28, 28], "conditional": {"transforms": [["left_shear", "right_shear", "left_rotate", "right_rotate", "towards", "far"]]}},
            "Objects10_3Dpose": {"input_shape": [1, 28, 28], "conditional": {"transforms": [["left_shear", "right_shear", "left_rotate", "right_rotate", "towards", "far"]]}}}


class DiscClaAttrDet(nn.Module):
    def __init__(self, input_shape, nclasses, attributes):
        super(DiscClaAttrDet, self).__init__()
        self.nclasses = nclasses
        self.nconditional_ldims = int(np.sum([len(attrs_list) for attrs_list in attributes]))
        self.model = conv_Nlayer_downscalar(input_shape,
                                            1+self.nclasses+self.nconditional_ldims,
                                            [128], 4, 4, 5, last_layer_act=False)

    def forward(self, x):
        return self.model(x)

    def decode_logits(self, y):
        return y[:, 0], y[:, 1:1+self.nclasses], y[:, 1+self.nclasses:1+self.nclasses+self.nconditional_ldims]


def acc(logits, labels, ratio=True):
    if logits.dim() == 1:  # not one hot
        pred_label = torch.round(nn.Sigmoid()(logits))
    else:
        _, pred_label = torch.max(logits, axis=1)
    corr = torch.sum(pred_label == labels).item()
    if ratio:
        return corr/len(labels)
    return corr


def disc_cla_verdict(disc_cla, zs, model, dataloader, y, y_attr):
    classes, attributes, conditional_losses = dataloader.get_dataset_params(["classes", "attributes", "conditional_loss_fns"])
    n = zs.shape[0]
    recons = dataloader.denormalize(model.decoder(zs.to(device))).to(device)
    logits = disc_cla(recons).to("cpu")
    disc_logit, cla_logits, attrdet_logits = disc_cla.decode_logits(logits)
    disc_logit = torch.round(nn.Sigmoid()(disc_logit))

    real = torch.sum(disc_logit).item()
    correct_classes = acc(cla_logits, torch.Tensor([y]*n), ratio=False)
    
    correct_attributes, ai_ind = 0, 0
    for ai, attrs_list in enumerate(attributes):
        nattrs = len(attrs_list)
        correct_attributes = np.max((acc(attrdet_logits[:, ai_ind:ai_ind+nattrs], torch.Tensor([y_attr]*n), ratio=False), correct_attributes))
        ai_ind += nattrs
    return real, correct_classes, correct_attributes, disc_logit, recons


def get_varying_attr(attrs):
    atr = attrs.transpose()
    for ti, tt in enumerate(atr):
        if len(np.unique(tt)) > 1:
            return ti
    return -1


def validate(args, dparams, disc_cla, validity_dir):
    if disc_cla is not None:
        summary_dir = os.path.join(validity_dir, "summary")
        os.makedirs(summary_dir, exist_ok=True)
        sw = SummaryWriter(summary_dir)

    dparams['batch_size'] = 1
    dataloader = CustomDataloader(dparams, apply_random_transforms=True, apply_harmful_transforms=False)
    classes, attributes, conditional_ldims = dataloader.get_dataset_params(["classes", "attributes", "conditional_ldims"])

    mparams['cla_args'] = {'num_classes': len(classes)}
    model = VaeClassifier(mparams, device, attributes, add_variational_noise=False).to(device)
    mparams['model_path'] = args['model_path']
    load_model(model, mparams, device)
    model.eval()
    
    progress_manager = enlighten.get_manager()
    batch_progress = progress_manager.counter(total=args['num_test_images'], disc="\tBatches", unit="images", leave=False)
    
    dataloader = CustomDataloader(dparams, apply_random_transforms=True, apply_harmful_transforms=False)
    dataset, classes, attributes, conditional_ldims = dataloader.get_dataset_params(["dataset", "classes", "attributes", "conditional_ldims"])
    dl, _, _ = dataloader.get_data(dparams['batch_size'])
    if dparams['dataset'] == 'Objects10_3Dpose':
        image_paths, unique_attrs = dataset.get_image_paths()
        dl = DataBatcher(image_paths, False)
        assert args["test_attribute"] in unique_attrs, print(args["test_attribute"], unique_attrs)
        for at in args["target_attributes"]:
            assert at in unique_attrs, print(at, unique_attrs)

    dists, second_ratios = [], []  # stats for encoder sanity
    total, real, corr_class, corr_attrs = 0, 0, 0, 0  # stats for decoder sanity
    ndims_to_check = conditional_ldims+4
    deltas = [None] * ndims_to_check  # cyclic stats for encoder sanity 
    for bi, batch in enumerate(dl.get_batch("cpu")):
        batch_progress.update()
        if bi >= args['num_test_images']:
            break
        x_set, y, y_attrs = get_specification_inputs(batch, model, device, dataloader, args)
        if y is None or x_set is None:
            continue
        with torch.no_grad():
            y_outs, zs, zs_mu_lvar = model(x_set.to(device), only_gen_z=True)  # to be able to do the expectancy check

        z_ind, z1 = 1, zs_mu_lvar[0]
        if dparams['dataset'] == 'Objects10_3Dpose':
            dparams['conditional']['transforms'] = [{"steps": len(args['target_attributes'])}]
        for k, v in dparams['conditional']['transforms'][0].items():
            nattr_steps = v['steps']
            z2 = zs_mu_lvar[z_ind+nattr_steps-1]  # extremes
            
            y_attr = y_attrs[:, 0][z_ind]
            # here, take the extremes of zs of every type of transform, form a line and take the in between ones of the transform,
            # calculate their distance from the line and the second best conditional attribute distance
            for si in range(1, nattr_steps-1):
                z_mu_lvar = zs_mu_lvar[z_ind+si-1]
                zi1 = z_mu_lvar - z1
                z21 = z2 - z1
                t = torch.dot(zi1, z21)/torch.dot(z21, z21)
                dists_ = torch.norm(zi1 - t*z21).item()
                dists.append(dists_)  # np.mean(dists_))
                if dparams['dataset'] == 'Objects10_3Dpose':
                    vattr = get_varying_attr(y_attrs)
                    corr_attr_z_var = torch.abs(z1 - z_mu_lvar)[vattr]
                    z_mu_lvar[vattr] = z1[vattr]
                    max_incorr_attr_z_var = torch.max(torch.abs(z1 - z_mu_lvar))
                    second_ratios.append((corr_attr_z_var/max_incorr_attr_z_var).item())
                else:
                    corr_attr_z_value = torch.abs(z_mu_lvar[y_attr]).item()
                    z_mu_lvar[y_attr] = 0
                    max_incorr_attr_z_value = torch.max(torch.abs(z_mu_lvar[:conditional_ldims]))
                    second_ratios.append((max_incorr_attr_z_value/(corr_attr_z_value+1e-5)).item())

            z_ind += nattr_steps
            if disc_cla is None:
                continue
            # take interps of the line, run disc_cla on the reconstructions and plot the # of fake/real {correct class, attribute detection}
            n = 7
            total += n
            enc_out_interps = torch.stack([z1 + (z2 - z1)*t for t in np.linspace(0, 1, n)])  # 
            z_interps = model.encoding_head.construct_z(enc_out_interps)  # construct zs here
            
            x_real, x_class, x_attr, disc_logits, recons = disc_cla_verdict(disc_cla, z_interps, model, dataloader, y, y_attr)
            real += x_real
            corr_class += x_class
            corr_attrs += x_attr
            
            # for the real ones, plot the distribution of reencoded latent vector from original one
            rdeltas = []
            for i, dlogit in enumerate(disc_logits):  # expecting a vector here
                if dlogit == 0.:
                    sw.add_image("detected fake", make_grid(dataloader.denormalize(recons)), global_step=bi)
                    continue
                recons = dataloader.denormalize(model.decoder(z_interps[i].unsqueeze(0).to(device))).to(device)
                with torch.no_grad():
                    _, _, recons_z_mu_lvar = model(recons, only_gen_z=True)  # to be able to do the expectancy check
                delta = torch.abs(z_mu_lvar - recons_z_mu_lvar).squeeze(0).detach().to("cpu")
                rdeltas.append(delta)
            if len(rdeltas) > 0:
                # store axis wise deltas here
                v_tensor = torch.stack((rdeltas), dim=0)
                for di in range(ndims_to_check):
                    dim_ver_delta, _ = np.histogram(v_tensor[:, di], bins=np.linspace(0, 3.5, 11))
                    if deltas[di] is None:
                        deltas[di] = dim_ver_delta
                    else:
                        deltas[di] += dim_ver_delta
    progress_manager.stop()
    print(dists, second_ratios)
    fig, axes = plt.subplots(ncols=2)
    fig.tight_layout()
    axes[0].hist(dists)
    axes[1].hist(second_ratios)
    plt.savefig(os.path.join(validity_dir, "encoding_space_validity.png"))
    plt.close("all")
    if disc_cla is not None:
        print(total, real, corr_class, corr_attrs)
        print(deltas)


def train(dparams, validity_dir):
    """ Trains and saves a class, fake/real and attribute identifier network for a given dataset"""
    start_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    summary_dir = os.path.join(validity_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    sw = SummaryWriter(summary_dir)

    rdataloader = CustomDataloader(dparams, apply_random_transforms=True, apply_harmful_transforms=False)
    classes, attributes = rdataloader.get_dataset_params(["classes", "attributes"])
    real_train_dl, _, real_test_dl = rdataloader.get_data(dparams['batch_size'])
    fdataloader = CustomDataloader(dparams, apply_random_transforms=False, apply_harmful_transforms=True)
    fake_train_dl, _, fake_test_dl = fdataloader.get_data(dparams['batch_size'])

    disc_cla = DiscClaAttrDet(dparams['input_shape'], len(classes), attributes).to(device)

    nepochs = 10
    cla_loss_cri, attr_loss_cri = nn.CrossEntropyLoss(), nn.CrossEntropyLoss()
    disc_loss_crt = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(disc_cla.parameters(), lr=1e-3)

    progress_manager = enlighten.get_manager()
    epoch_progress = progress_manager.counter(total=nepochs, disc="Epochs", unit="batches")
    batch_progress = progress_manager.counter(total=len(real_train_dl) + len(fake_train_dl), disc="\tBatches", unit="images", leave=False)
    step, best_cla_acc, best_realism_acc, best_attrs_acc = 0, 0, 0, 0
    for epoch in range(nepochs):
        disc_cla.train()
        for bi, ((rx, y, y_attrs), (fx, _, _)) in enumerate(zip(real_train_dl.get_batch(device), fake_train_dl.get_batch(device))):
            # DISC
            loss_disc, cla_logits, attrdet_logits = 0, None, None
            for di, x in enumerate([fx, rx]):
                logits = disc_cla(x)
                disc_labels = torch.FloatTensor([di]*logits.shape[0]).to(device)  # fake: 0, real : 1
                disc_logits, cla_logits, attrdet_logits = disc_cla.decode_logits(logits)
                loss_disc += disc_loss_crt(disc_logits, disc_labels)
            # CLA
            loss_cla = cla_loss_cri(cla_logits, y)
            # ATTR_DET
            loss_attrs, ai_ind = 0, 0
            for ai, attrs_list in enumerate(attributes):
                nattrs = len(attrs_list)
                loss_attrs += attr_loss_cri(attrdet_logits[:, ai_ind:ai_ind+nattrs], y_attrs[:, ai])
                ai_ind += nattrs
            loss = loss_disc + loss_cla + loss_attrs
            opt.zero_grad()
            loss.backward()
            opt.step()

            batch_progress.update()
            sw.add_scalar("Discriminator loss", loss_disc, global_step=step)
            sw.add_scalar("Classifier loss", loss_cla, global_step=step)
            sw.add_scalar("Attribute detection loss", loss_attrs, global_step=step)
            if bi % 250 == 0:
                sw.add_image("real", make_grid(rdataloader.denormalize(rx)), global_step=step)
                sw.add_image("fake", make_grid(fdataloader.denormalize(fx)), global_step=step)
            step += 1
        epoch_progress.update()

        disc_cla.eval()
        batch_realism_accs, batch_cla_accs, batch_attrs_accs = [], [], []
        for bi, ((rx, y, y_attrs), (fx, _, _)) in enumerate(zip(real_test_dl.get_batch(device), fake_test_dl.get_batch(device))):
            cla_logits, attrdet_logits = None, None
            # DISC
            for di, x in enumerate([fx, rx]):
                with torch.no_grad():
                    logits = disc_cla(x)
                disc_labels = torch.FloatTensor([di]*logits.shape[0]).to(device)  # fake: 0, real : 1
                disc_logits, cla_logits, attrdet_logits = disc_cla.decode_logits(logits)
                batch_realism_accs.append(acc(disc_logits, disc_labels))
            # CLA
            batch_cla_accs.append(acc(cla_logits, y))
            # ATTR_DET
            attrs_acc, ai_ind = np.inf, 0
            for ai, attrs_list in enumerate(attributes):
                nattrs = len(attrs_list)
                attrs_acc = np.min((acc(attrdet_logits[:, ai_ind:ai_ind+nattrs], y_attrs[:, ai]), attrs_acc))  # stricter accuracy
                ai_ind += nattrs
            batch_attrs_accs.append(attrs_acc)
        batch_cla_acc, batch_realism_acc, batch_attrs_acc = np.mean(batch_cla_accs), np.mean(batch_realism_accs), np.mean(batch_attrs_accs)
        sw.add_scalar("Discriminator accuracy", batch_cla_acc, global_step=step)
        sw.add_scalar("Classifier accuracy", batch_realism_acc, global_step=step)
        sw.add_scalar("Attribute detection accuracy", batch_attrs_acc, global_step=step)
        print(f"Test CLA acc: {batch_cla_acc}, disc acc: {batch_realism_acc}, ATTRS acc: {batch_attrs_acc}")
        if batch_realism_acc > best_realism_acc and batch_cla_acc > best_cla_acc and batch_attrs_acc > best_attrs_acc:
            torch.save(disc_cla, os.path.join(validity_dir, f"{dparams['dataset']}_disc_cla_{start_time}.tar"))
        best_cla_acc = np.max((batch_cla_acc, best_cla_acc))
        best_realism_acc = np.max((batch_realism_acc, best_realism_acc))
        best_attrs_acc = np.max((batch_attrs_acc, best_attrs_acc))

    progress_manager.stop()
    return disc_cla


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Verify a trained model (classifier+encoder+decoder).')
    parser.add_argument('--dataset', dest='dataset', type=str, default="", help='Name of the dataset for which the disc_CLA_ATTRDET is to be trained')
    parser.add_argument('--disc_cla_path', dest='disc_cla_path', type=str, default="", help="Path to trained real/fake discriminator, classifier and attribute detector")
    parser.add_argument('--model_path', dest='model_path', type=str, default="", help='path to model tar')
    parser.add_argument('--test_attribute', dest='test_attribute', help='attribute against which invariance is to be verified')
    parser.add_argument('--target_attributes', dest='target_attributes', nargs="*", default=[])
    parser.add_argument('--flip_head', dest='flip_head', action='store_true')
    parser.add_argument('--num_test_images', dest='num_test_images', type=int, default=100, help='Number of images to locally verify model for')
    args = parser.parse_args()
    # providing dataset means that disc_cla_attr is to be trained
    # providing disc_cla_path means a model decoder should also be validated
    assert args.model_path != "" or args.dataset != "", print(args.dataset, args.model_path)
    # add assert for args.dataset

    dparams = None
    if args.model_path != "":
        model_root_dir = os.path.dirname(os.path.dirname(args.model_path))
        params, _ = load_params(os.path.join(model_root_dir, "config.txt"))
        dparams, mparams = params['dataset'], params['model']
        assert params['model']['train_vae']  # i.e., model is of a pipeline, not just cla
    else:
        dparams = {'dataset': args.dataset, 'input_shape': DEFAULTS[args.dataset]["input_shape"], 'batch_size': 128,
                   'conditional': DEFAULTS[args.dataset]["conditional"],
                   'conditional_loss_fn': ["CE"]}
        for dp in ['data_balance_method', 'classes']:
            dparams[dp] = []
    disc_cla_dir = os.path.join("data", "validators", args.dataset)

    disc_cla = None
    if args.dataset != "":
        if args.disc_cla_path == "":
            os.makedirs(disc_cla_dir, exist_ok=True)
            disc_cla = train(dparams, disc_cla_dir)
            exit(0)
        else:
            disc_cla = torch.load(args.disc_cla_path).to(device)

    get_specification(dparams, args.test_attribute, args.target_attributes)
    validity_dir = os.path.join(model_root_dir, "validity_results")
    os.makedirs(validity_dir, exist_ok=True)
    validate(args.__dict__, dparams, disc_cla, validity_dir)

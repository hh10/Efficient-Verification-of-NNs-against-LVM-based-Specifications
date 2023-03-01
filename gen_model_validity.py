# Scripts trains a Discriminator plus Classifier to classify real vs fake images and classes belonging to a dataset:
# produces specification inputs as in verify.py, passes n interps from the decoder to see if they are classified correctly plus real/fake
# explores the conditional dims in steps to check the same thing
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from dataloader import CustomDataloader, DataBatcher
from utils import load_params, load_model
from verify_utils import get_specification_inputs, get_specification
from model import VaeClassifier
from train_validator import discs_verdict, ClaAttrDet

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

SLP = True


def get_varying_attr(attrs):
    atr = attrs.transpose()
    for ti, tt in enumerate(atr):
        if len(np.unique(tt)) > 1:
            return ti
    return -1


def get_rf_segments(endpts, gan_disc, model, device):
    # load the real_vs_fake discriminator
    npts = 5  # interp pts to learn between endpoints
    # define interpolation points
    interp_pts = [endpts[0],
                  torch.autograd.Variable(endpts[0].clone(), requires_grad=True),
                  torch.autograd.Variable((endpts[0] + endpts[1])*0.5, requires_grad=True),
                  torch.autograd.Variable(endpts[1].clone(), requires_grad=True),
                  endpts[1]]
    alpha = 1e-3
    disc_real_target = torch.ones(npts-2).to(device)
    decodings_evolution = []
    for i in range(100):
        # define desirable criteria:
        paths = torch.stack([torch.abs(interp_pts[i+1]-interp_pts[i]) for i in range(npts-1)])
        path_lengths = torch.sum(paths**2, axis=1)
        # equal length constraint
        path_dev, _ = torch.std_mean(path_lengths)
        # shortest path constraint
        path_total = torch.sum(path_lengths)
        # dataset guidance
        z = model.encoding_head.construct_z(torch.stack(interp_pts[1:-1]), add_noise=False)
        decodings = model.decoder(z).to(device)
        if i % 20 == 0:
            decodings_evolution.append(decodings)
        gan_out = gan_disc(decodings)
        disc_out_max, _ = torch.max(gan_out, axis=1)
        disc_out_loss = torch.sum(disc_real_target - disc_out_max)
        interpolation_obj = disc_out_loss + 0.25*path_total + 0.25*path_dev
        interpolation_obj.backward()
        for i in range(1, 4):  # update the intermediate points
            interp_pts[i].data = (interp_pts[i] + alpha * interp_pts[i].grad.detach().sign())
            interp_pts[i].grad.zero_()
    return interp_pts, decodings_evolution


def validate(args, mparams, dparams, validity_dir, rf_disc=None, ac_disc=None):
    if rf_disc is not None or ac_disc is not None:
        summary_dir = os.path.join(validity_dir, "summary")
        os.makedirs(summary_dir, exist_ok=True)
        sw = SummaryWriter(summary_dir)

    dataloader = CustomDataloader(dparams, apply_random_transforms=True, apply_harmful_transforms=False)
    dataset, classes, attributes, conditional_ldims = dataloader.get_dataset_params(["dataset", "classes", "attributes", "conditional_ldims"])

    mparams['cla_args'] = {'num_classes': len(classes)}
    model = VaeClassifier(mparams, device, attributes, add_variational_noise=False).to(device)
    mparams['model_path'] = args['model_path']
    load_model(model, mparams, device)
    model.eval()
    
    progress_manager = enlighten.get_manager()
    batch_progress = progress_manager.counter(total=args['num_test_images'], disc="\tBatches", unit="images", leave=False)
    
    _, dl, _ = dataloader.get_data(batch_size=1)
    if dparams['dataset'] == 'Objects10_3Dpose':
        image_paths, unique_attrs = dataset.get_image_paths()
        dl = DataBatcher(image_paths, False)
        assert args["test_attribute"] in unique_attrs, print(args["test_attribute"], unique_attrs)
        for at in args["target_attributes"]:
            assert at in unique_attrs, print(at, unique_attrs)

    dists, second_ratios = [], []  # stats for encoder sanity
    ndims_to_check = conditional_ldims+4
    total, real, corr_class = 0, 0, 0  # stats for decoder sanity
    deltas = [None] * ndims_to_check  # cyclic stats for encoder sanity 
    num_tested_images = 0
    for bi, batch in enumerate(dl.get_batch("cpu")):
        batch_progress.update()
        x_set, y, y_attrs = get_specification_inputs(batch, model, device, dataloader, args)
        if y is None or x_set is None:  # happens when model's pred is inaccurate for x
            continue
        if num_tested_images >= args['num_test_images']:
            break
        num_tested_images += 1

        with torch.no_grad():
            y_outs, zs, zs_mu_lvar = model(x_set.to(device), only_gen_z=True)  # to be able to do the expectancy check

        z_ind, z1 = 1, zs_mu_lvar[0]
        if dparams['dataset'] == 'Objects10_3Dpose':
            dparams['conditional']['transforms'] = [{"steps": len(args['target_attributes'])}]
        for k, v in dparams['conditional']['transforms'][0].items():
            nattr_steps = v['steps']
            z2 = zs_mu_lvar[z_ind+nattr_steps-1]  # extremes
            z_segs = [z2 - z1]
            if not SLP:
                z_segs, decodings_evolution = get_rf_segments([z1, z2], rf_disc, model, device)
                if bi % 5 == 0:
                    sw.add_image("Decoding evolution", make_grid(dataloader.denormalize(torch.concat(decodings_evolution)), nrow=3), global_step=bi)

            y_attr = y_attrs[:, 0][z_ind]
            # here, take the extremes of zs of every type of transform, form a line and take the in between ones of the transform,
            # calculate their distance from the line and the second best conditional attribute distance
            for si in range(1, nattr_steps-1):
                z_mu_lvar = zs_mu_lvar[z_ind+si-1]
                zi1 = z_mu_lvar - z1
                min_dist_ = np.inf
                for z_seg in z_segs:
                    t = torch.dot(zi1, z_seg)/torch.dot(z_seg, z_seg)
                    dist_ = torch.norm(zi1 - t*z_seg).item()/np.sqrt(z1.shape[-1])  # torch.norm(z_seg).item()
                    min_dist_ = min(dist_, min_dist_)
                dists.append(min_dist_)

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
            if ac_disc is None:
                continue

            # take interps of the line, run disc_cla on the reconstructions and plot the # of fake/real {correct class}
            n = 5
            total += n
            enc_out_interps = torch.stack([z1 + (z2 - z1)*t for t in np.linspace(0, 1, n)])  # 
            z_interps = model.encoding_head.construct_z(enc_out_interps)  # construct zs here

            x_real, x_class, disc_logits, recons = discs_verdict(rf_disc, ac_disc, z_interps, model, dataloader, y)
            real += x_real
            corr_class += x_class

            # for the real ones, plot the distribution of reencoded latent vector from original one
            rdeltas = []
            for i, dlogit in enumerate(disc_logits):  # expecting a vector here
                if dlogit == 0.:
                    sw.add_image("detected fake", make_grid(dataloader.denormalize(recons)), global_step=bi)
                    continue
                recons = model.decoder(z_interps[i].unsqueeze(0).to(device)).to(device)
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
    fig, axes = plt.subplots(ncols=2)
    fig.tight_layout()
    density = False
    print(np.histogram(dists, density=density), np.histogram(second_ratios, density=density))
    print(np.histogram(dists, bins=[0, 0.3, 1, 2, 4, 6, 8], density=density), np.histogram(second_ratios, bins=[0, 0.3, 1, 2, 4, 6, 8], density=density))
    axes[0].hist(dists, bins=[0, 0.3, 1, 2, 4, 6, 8], density=density)
    axes[1].hist(second_ratios, bins=[0, 0.3, 1, 2, 4, 6, 8], density=density)
    plt.savefig(os.path.join(validity_dir, "encoding_space_validity.png"))
    plt.close("all")
    if rf_disc is not None:
        print(total, real, corr_class)
        print(deltas)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Verify a trained model (classifier+encoder+decoder).')
    parser.add_argument('--model_path', dest='model_path', type=str, default="", help='path to model tar')
    parser.add_argument('--num_test_images', dest='num_test_images', type=int, default=100, help='Number of images to locally verify model for')
    # optional
    parser.add_argument('--rf_disc_path', dest='rf_disc_path', type=str, default="", help="Path to trained real/fake discriminator")
    parser.add_argument('--ac_disc_path', dest='ac_disc_path', type=str, default="", help="Path to trained classifier and attribute detector")
    parser.add_argument('--test_attribute', dest='test_attribute', help='attribute against which invariance is to be verified')
    parser.add_argument('--target_attributes', dest='target_attributes', nargs="*", default=[])
    parser.add_argument('--flip_head', dest='flip_head', action='store_true')
    args = parser.parse_args()
    assert args.model_path != "", print(f"{args.model_path} not provided")

    model_root_dir = os.path.dirname(os.path.dirname(args.model_path))
    params, _ = load_params(os.path.join(model_root_dir, "config.txt"))
    dparams, mparams = params['dataset'], params['model']
    assert params['model']['train_vae']  # i.e., model is of a pipeline, not just cla

    rf_disc = None
    if args.rf_disc_path:
        rf_disc = torch.load(args.rf_disc_path).to(device)
    if not SLP:
        assert rf_disc is not None

    ac_disc = None
    if args.ac_disc_path:
        ac_disc = torch.load(args.ac_disc_path).to(device)

    get_specification(dparams, args.test_attribute, args.target_attributes)
    validity_dir = os.path.join(model_root_dir, "validity_results")
    os.makedirs(validity_dir, exist_ok=True)
    validate(args.__dict__, mparams, dparams, validity_dir, rf_disc, ac_disc)

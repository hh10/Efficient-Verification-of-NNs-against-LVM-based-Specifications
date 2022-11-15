import torch

import numpy as np
import random
import os
from datetime import datetime
import enlighten
import argparse

from dataloader import CustomDataloader, DataBatcher
from model import VaeClassifier
from utils import load_params, evaluate_accuracy, get_balanced_batch, load_model, get_conditional_limits
from visualizations import plot_verification_results, plot_local_conditional_effect, plot_verification_plots, plot_local_conditionals
from verify_utils import (get_specification, get_specification_inputs, verinet_verify_for_zs,
                          save_ver_results, prepare_query1_results_dict, get_specification_inputs_object3d)
from verinet_line_segment_verification import get_VerinetNN_solver
from verinet.verification.verifier_util import Status
from verinet.verification.objective import Objective


def verify_data(model, device, data, ndims_to_check, dparams, test_transform_names, dataloader, images_dir, args, save_args):
    classes, attributes, dataset_name = dataloader.get_dataset_params(["classes", "attributes", "dataset_name"])
    num_transforms, nimages = len(test_transform_names), args['num_test_images']

    ver_results = prepare_query1_results_dict(len(classes), num_transforms, ndims_to_check)
    feat_ver_results = prepare_query1_results_dict(len(classes), num_transforms, ndims_to_check)

    progress_manager = enlighten.get_manager()
    batch_progress = progress_manager.counter(total=nimages, desc="\tBatches", unit="images", leave=False)
    for bi, batch in enumerate(data.get_batch("cpu")):
        if bi >= nimages:
            break
        # get the x_set images for specification setpoints
        if dataset_name == "Objects10_3Dpose":
            x_set, y, _ = get_specification_inputs_object3d(batch, model, device, dataloader, args)
        else:
            x_set, y, _ = get_specification_inputs(batch, model, device, dataloader, args)
        if y is None:  # did not find relevant images
            continue
        ver_results['total'][y] += 1
        feat_ver_results['total'][y] += 1
        if x_set is None:  # implies classification for original image is incorrect
            continue
        ver_results['tested'][y] += 1
        feat_ver_results['tested'][y] += 1

        # Query 1: set of images that need to be interpolated among/between
        verinet_verify_for_zs(x_set, y, model, ver_results, len(classes), device, ndims_to_check, dparams)
        verinet_verify_for_zs(x_set, y, model, feat_ver_results, len(classes), device, ndims_to_check, dparams, feat_space_spec=True)

        if bi % 3 == 0 or bi == nimages-1:
            save_ver_results(*save_args, ver_results, bi+1, test_transform_names)
            save_ver_results(*save_args[:-1], save_args[-1] + "_feat", feat_ver_results, bi+1, test_transform_names)
        if bi % 5 == 0 or bi == nimages-1:
            plot_verification_results(model, device, ver_results["images"], os.path.join(images_dir, "ls"), dataloader, bi)
            plot_verification_results(model, device, ver_results["images"], os.path.join(images_dir, "feat"), dataloader, bi)
        if bi % 20 == 0 or bi == nimages-1:
            plot_verification_plots(ver_results, images_dir, classes, test_transform_names, bi)
        batch_progress.update()
    progress_manager.stop()
    return ver_results


def verinet_verify_conditional_zs(model, device, data, ndims_to_check, nimages, dataloader, images_dir, save_args):
    # get the verinet_model
    vmodel, solver = get_VerinetNN_solver(model.verification_model_layers())
    climits, _ = get_conditional_limits(dataloader)

    ver_results = {'cond_x': [], 'cond_ceg_z': []}
    progress_manager = enlighten.get_manager()
    batch_progress = progress_manager.counter(total=nimages, desc="\tBatches", unit="images", leave=False)
    for bi, (x, y, _) in enumerate(data.get_batch("cpu")):
        if bi >= nimages:
            break
        model, x, y = model.to(device), x.to(device), y.to(device)
        with torch.no_grad():
            pred_logits, _, z_mu_lvar = model(x, only_gen_z=True)
            _, pred_label = torch.max(pred_logits, axis=1)
            if pred_label != y:
                continue
            num_classes = pred_logits.shape[-1]
            z_mu_lvar = z_mu_lvar.squeeze(0)  # [2*ld]
            assert z_mu_lvar.dim() == 1, print("z_mu_lvar shape:", z_mu_lvar.shape)

            for ni in range(ndims_to_check):
                for eps in [0.02, 0.06, 0.1, 0.25, 0.5, 1., 1.5, 2.5]:
                    climit = climits[ni] if ni < len(climits) else [[-3.5], [3.5]]
                    input_bounds = np.zeros((z_mu_lvar.shape[0], 2), dtype=np.float32)
                    input_bounds[ni, 0] = np.max((z_mu_lvar[ni] - eps, climit[0][0]))
                    input_bounds[ni, 1] = np.min((z_mu_lvar[ni] + eps, climit[-1][-1]))
                    objective = Objective(input_bounds, output_size=num_classes, model=vmodel)

                    ver_status, counterex_using_PGD = solver.verify(objective=objective, timeout=30)

                    if eps not in ver_results['ldim_eps'][ni]:
                        ver_results['ldim_eps'][ni][eps] = 0
                    if ver_status == Status.Safe:
                        ver_results['ldim_eps'][ni][eps] += 1
                    else:
                        if ver_status == Status.Unsafe:
                            enc_out_ceg = solver.counter_example
                            assert enc_out_ceg is not None, print("Status unsafe, but enc_out_ceg is None.")
                            z_ceg = model.encoding_head.construct_z(enc_out_ceg)
                            ver_results['cond_x'].append(x)
                            ver_results['cond_ceg_z'].append((ni, eps, z_ceg))
                        break  # no need to check for larger eps if smaller fails
        if bi % 3 == 0 or bi == nimages-1:
            save_ver_results(*save_args[:-1], save_args[-1] + "_conditional", ver_results, bi+1)
        if bi % 5 == 0 or bi == nimages-1:
            plot_local_conditionals(ver_results['cond_x'], ver_results['cond_ceg_z'], model, device, os.path.join(images_dir, f'cond_unver_recons_{bi}.png'), dataloader)
        batch_progress.update()
    progress_manager.stop()
    solver.cleanup()
    del solver


def verinet_verify_region_zs(model, device, data, ndims_to_check, nimages, save_args):
    progress_manager = enlighten.get_manager()
    batch_progress = progress_manager.counter(total=nimages, desc="\tBatches", unit="images", leave=False)

    # get the verinet_model
    vmodel, solver = get_VerinetNN_solver(model.verification_model_layers())
    
    region_eps = {}
    for bi, (x, y, _) in enumerate(data.get_batch("cpu")):
        if bi >= nimages:
            break
        model, x, y = model.to(device), x.to(device), y.to(device)
        with torch.no_grad():
            pred_logits, _, z_mu_lvar = model(x.to(device), only_gen_z=True)
            _, pred_label = torch.max(pred_logits, axis=1)
            if pred_label != y:
                continue
            num_classes = pred_logits.shape[-1]
            z_mu_lvar = z_mu_lvar.squeeze(0)  # [2*ld]
            assert z_mu_lvar.dim() == 1, print("z_mu_lvar shape:", z_mu_lvar.shape)

            for eps in [0.001, 0.0025, 0.005, 0.01, 0.0125, 0.025, 0.05, 0.1]:
                input_bounds = np.zeros((z_mu_lvar.shape[0], 2), dtype=np.float32)
                input_bounds[:, 0] = z_mu_lvar - eps
                input_bounds[:, 1] = z_mu_lvar + eps
                objective = Objective(input_bounds, output_size=num_classes, model=vmodel)
                ver_status, counterex_using_PGD = solver.verify(objective=objective, timeout=30)

                if eps not in region_eps:
                    region_eps[eps] = 0
                if ver_status == Status.Safe:
                    region_eps[eps] += 1
                else:
                    break  # no need to check for larger eps if smaller fails
        if bi % 3 == 0 or bi == nimages-1:
            save_ver_results(*save_args[:-1], save_args[-1] + "_region", region_eps, bi)
        batch_progress.update()
    progress_manager.stop()
    solver.cleanup()
    del solver


def setup_and_verify(args):
    # set random seed for reproducibility.
    seed = 1123
    random.seed(seed)
    torch.manual_seed(seed)
    print("Random Seed: ", seed)

    # use GPU if available.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device, "will be used.")

    model_path, test_attribute, target_attributes = args['model_path'], args['test_attribute'], args['target_attributes']
    model_root_dir = os.path.dirname(os.path.dirname(model_path))
    params, _ = load_params(os.path.join(model_root_dir, "config.txt"))
    dparams, mparams = params['dataset'], params['model']
    assert mparams['train_vae']  # i.e., model is of a pipeline, not just cla

    # sanity check (part 1)
    if dparams['dataset'] in ['Objects10_3Dpose']:
        assert test_attribute != "" and target_attributes != [] and test_attribute not in target_attributes
    elif dparams['dataset'] == 'CelebA':
        assert test_attribute != ""

    test_transform_names = get_specification(dparams, test_attribute, target_attributes)

    # prepare dataset
    dparams['batch_size'] = 1
    dataloader = CustomDataloader(dparams, apply_random_transforms=False)
    dataset, classes, attributes, conditional_ldims = dataloader.get_dataset_params(["dataset", "classes", "attributes", "conditional_ldims"])
    train_dl, _, test_dl = dataloader.get_data(dparams['batch_size'])

    # output folder for runs (dir structure is verification_results/dataset/date/time_runShortDescription)
    results_dir = os.path.join(model_root_dir, "verification_results", os.path.basename(model_path), datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
    images_dir, results_file = os.path.join(results_dir, 'plots'), os.path.join(results_dir, "results")
    os.makedirs(os.path.join(images_dir, "ls"), exist_ok=True)
    os.makedirs(os.path.join(images_dir, "feat"), exist_ok=True)

    mparams['cla_args'] = {'num_classes': len(classes)}
    vae_cla = VaeClassifier(mparams, device, attributes, add_variational_noise=False).to(device)
    mparams['model_path'] = model_path
    load_model(vae_cla, mparams, device)
    vae_cla.eval()  # todo(hh): check why eval doesn't work well
    # save nominal test accuracy
    nominal_acc_str = "\nModels test accuracy is : {}".format(evaluate_accuracy(vae_cla, test_dl, device, show_progress=True))
    print(nominal_acc_str)
    with open(results_file + ".txt", "w") as rf:
        print(nominal_acc_str, file=rf)

    # some visualizations
    print("Collecting train class and attribute samples for visualizations...")
    for dli in range(3):
        sample_batch = get_balanced_batch(train_dl, dataloader)
        plot_local_conditional_effect(sample_batch, vae_cla, device, os.path.join(results_dir, f'Conditional_Reconstructions_set{dli}.png'), dataloader)

    # sanity check (part 2)
    if dparams['dataset'] == 'Objects10_3Dpose':
        image_paths, unique_attrs = dataset.get_image_paths()
        train_dl = DataBatcher(image_paths, False)
        assert test_attribute in unique_attrs, print(test_attribute, unique_attrs)
        for at in target_attributes:
            assert at in unique_attrs, print(at, unique_attrs)

    # VERIFY
    results_save_args, nimages = [dparams['dataset'], model_path, results_file], args['num_test_images']
    ndims_to_check = np.min((conditional_ldims+4, mparams['latent_dim']))
    ver_results = verify_data(vae_cla, device, train_dl, ndims_to_check, dparams, test_transform_names, dataloader, images_dir, args, results_save_args)
    if len(attributes) > 0:
        verinet_verify_conditional_zs(vae_cla, device, train_dl, ndims_to_check, nimages, dataloader, images_dir, results_save_args)
        verinet_verify_region_zs(vae_cla, device, train_dl, ndims_to_check, nimages, results_save_args)
    save_ver_results(*results_save_args, test_transform_names, ver_results, nimages)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Verify a trained model (classifier+encoder+decoder).')
    parser.add_argument('--model_path', dest='model_path', help='path to model tar', required=True)
    parser.add_argument('--test_attribute', dest='test_attribute', help='attribute against which invariance is to be verified')
    parser.add_argument('--target_attributes', dest='target_attributes', nargs="*", default=[])
    parser.add_argument('--flip_head', dest='flip_head', action='store_true')
    parser.add_argument('--save_images', dest='save_images', action='store_true')
    parser.add_argument('--num_test_images', dest='num_test_images', type=int, default=500, help='Number of images to locally verify model for')
    args = parser.parse_args()

    setup_and_verify(args.__dict__)    

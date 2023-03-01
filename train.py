import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

import numpy as np
import os
import time
import enlighten
import random
import argparse
import shutil
from typing import Union, Dict

from dataloader import CustomDataloader
from model import VaeClassifier
from utils import (load_params, accuracy_metric, evaluate_accuracy, get_balanced_batch, save_model,
                   load_model, update_experiments_summary, printl, prepare_training_results_dir)
from visualizations import plot_images, plot_visual_checks, plot_embeddings

## Use the following environment variable when profilling to force sync cuda/cpu after every op;
## rather than making large syncs at unexpected places within the program and hogging a lot of 
## time at seemingly simple operations
# CUDA_LAUNCH_BLOCKING=1
# torch.autograd.set_detect_anomaly(True)
# torch.set_num_interop_threads(1)

# set random seed for reproducibility.
seed = 1123
random.seed(seed)
torch.manual_seed(seed)
print("Random Seed: ", seed)

# use GPU if available.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device, "will be used.")
print("Number of interop threads used:", torch.get_num_interop_threads())
print("Number of threads used:", torch.get_num_threads())
print("-" * 25, '\n')


def get_epoch_loss_weights(tparams, epoch, train_sep, sw):
    """ Relative weightage of different components of the VAE-CLA weights:
     - loss_kl_weight, loss_recons_weight: set importance of different network sub-components
     - cla_weight: balances between the importance given to classification loss vs encoding loss
    """
    loss_kl_weight, loss_recons_weight, cla_weight = 1, 1, 1
    nepochs = tparams['num_epochs']
    if tparams['train_vae'] and tparams['train_cla'] and not train_sep:
        if epoch < tparams['only_vae_training_epochs']:
            cla_weight = 0
        else:
            loss_kl_weight = tparams['loss_kl_weight'] * np.min((1, float(epoch+nepochs/5)/nepochs))
            loss_recons_weight = tparams['loss_recons_weight']
            cla_weight = np.min((0.97, float(epoch+13)/(nepochs-tparams['only_cla_training_epochs']+4)))
        sw.add_scalar("Classification weightage", cla_weight, global_step=epoch)
    return loss_kl_weight, loss_recons_weight, cla_weight


def train(train_dl, test_dl, model, mparams, tparams: dict, dataloader, sample_batch, 
          results_dirs: Dict[str, str], train_sep: bool = False, dry_run: bool = False):
    """ Trains and reports per-epoch stats for the model """
    printl("Starting training Loop for {} epochs...".format(tparams['num_epochs']))
    sw = SummaryWriter(results_dirs['summary'])
    classes, conditional_ldims = dataloader.get_dataset_params(["classes", "conditional_ldims"])
    latent_dims = mparams["latent_dim"]

    # setup optimizers based on params
    train_both = tparams['train_vae'] and tparams['train_cla']
    if train_both and not train_sep:
        opt = torch.optim.Adam(model.parameters(), lr=tparams['lr'])
    else:
        if train_sep or tparams['train_vae']:
            vae_params = list(model.encoding_head.parameters()) + list(model.decoder.parameters())
            opt_vae = torch.optim.Adam(vae_params, lr=tparams['lr'])
        if train_sep or tparams['train_cla']:
            cla_params = list(model.fdn.parameters()) + list(model.classification_head.parameters())
            opt_cla = torch.optim.Adam(cla_params, lr=tparams['lr'])

    if tparams['train_vae'] and tparams['GAN_start_training_epochs'] > -1:
        Discriminator = nn.Sequential(nn.Flatten(), nn.Linear(2*latent_dims, 2)).to(device)
        opt_disc = torch.optim.Adam(Discriminator.parameters(), lr=tparams['lr']*0.05, betas=(0.5, 0.999))
        opt_gen = torch.optim.Adam(model.decoder.parameters(), lr=tparams['lr']*0.05, betas=(0.5, 0.999))
    
    step, best_test_acc, best_recons_loss = 0, 0, np.inf
    progress_manager = enlighten.get_manager()
    epoch_progress = progress_manager.counter(total=tparams['num_epochs'], desc="Epochs", unit="batches")
    batch_progress = progress_manager.counter(total=len(train_dl), desc="\tBatches", unit="images", leave=False)
    start_time = time.time()

    for epoch in range(tparams['num_epochs']):
        model.train()
        loss_kl_weight, loss_recons_weight, cla_weight = get_epoch_loss_weights(tparams, epoch, train_sep, sw)

        losses_recons, losses_kl, zs, z_noisys, ys = [], [], [], [], []  # for summary_writer
        epoch_start_time = time.time()
        for bi, (x, y, y_attrs) in enumerate(train_dl.get_batch(device)):
            if dry_run and bi > 9:
                break

            y_logits = model(x, y_attrs=y_attrs)
            if tparams['train_vae']:
                x_feat, y_logits, z, z_mu_lvar, x_hat = y_logits

            loss_cla, loss_vae, loss_recons = np.inf, np.inf, np.inf
            if tparams['train_cla']:
                loss_cla = nn.CrossEntropyLoss()(y_logits, y)
                loss = loss_cla
                acc = accuracy_metric(y_logits, y)
                sw.add_scalar("Training accuracy", acc, global_step=step)
            if tparams['train_vae'] or tparams['GAN_start_training_epochs'] > -1:
                zs.extend(z.detach().to('cpu').numpy()); ys.extend(y.to('cpu').numpy())  # noqa: E702
                if tparams['GAN_start_training_epochs'] > -1:
                    # train DISCRIMINATOR
                    real_label, fake_label = torch.ones(x.shape[0]).to(device).long(), torch.zeros(x.shape[0]).to(device).long()
                    loss_disc = nn.CrossEntropyLoss()(Discriminator(x_feat.detach()), real_label)
                    loss_disc += nn.CrossEntropyLoss()(Discriminator(model.fdn(model.decoder(z.detach()))), fake_label)
                    loss_disc *= 0.5
                    sw.add_scalar("Discriminator loss", loss_disc, global_step=step)
                    opt_disc.zero_grad()
                    loss_disc.backward(retain_graph=tparams['GAN_start_training_epochs'] > -1)
                    opt_disc.step()
            
                loss_kl, loss_conditional = model.get_encoding_losses()
                # adding loss for z cyclic consistency so counterexamples are true and encoding is interpretable
                z_mu_lvar_hat = model(x_hat)[3]
                loss_cc = ((z_mu_lvar - z_mu_lvar_hat) ** 2).sum()
                loss_vae = loss_kl * loss_kl_weight + loss_conditional * tparams['loss_conditional_weight'] * np.min((25., float(1+epoch))) + loss_cc
                if tparams['train_vae']:
                    # use reconstruction loss only if LVM is VAE, else only use encoding losses
                    loss_rec = ((x - x_hat) ** 2).sum()
                    loss_vae += loss_rec * loss_recons_weight
                loss, loss_recons = loss_vae, loss_rec.item()
                sw.add_scalar("Reconstruction loss", loss_recons, global_step=step); losses_recons.append(loss_recons)  # noqa: E702
                sw.add_scalar("KL loss", loss_kl, global_step=step); losses_kl.append(loss_kl.item())  # noqa: E702
                sw.add_scalar("Conditional loss", loss_conditional, global_step=step)
            if train_both:
                loss = (1 - cla_weight) * loss_vae + cla_weight * loss_cla
            sw.add_scalar("Training loss", loss, global_step=step)

            if train_both and not train_sep:
                opt.zero_grad()
                loss.backward()
                opt.step()
            else:
                if train_sep or tparams['train_cla']:
                    opt_cla.zero_grad()
                    loss_cla.backward()
                    opt_cla.step()
                if train_sep or tparams['train_vae']:
                    opt_vae.zero_grad()
                    loss_vae.backward()
                    opt_vae.step()
        
            if tparams['train_vae'] and tparams['GAN_start_training_epochs'] > -1 and epoch >= tparams['GAN_start_training_epochs']:
                # additional GAN training
                z_noisy = z.detach()  # detach because GAN loss is only for improving decoder and noise added only in non-conditional dimensions
                z_noisy[:, conditional_ldims:] += torch.normal(0, 0.5, size=(z_noisy.shape[0], latent_dims-conditional_ldims)).to(device)
                z_noisys.extend(z_noisy.to('cpu').numpy())
                xz_hat = model.decoder(z_noisy)
                loss_gen = (1-cla_weight) * nn.CrossEntropyLoss()(Discriminator(model.fdn(xz_hat)), real_label)
                sw.add_scalar("Generator loss", loss_gen, global_step=step)
                opt_gen.zero_grad()
                loss_gen.backward()
                opt_gen.step()
                if bi % 250 == 0:
                    sw.add_image("random reconstructed", make_grid(dataloader.denormalize(xz_hat)), global_step=step)

            if tparams['train_vae'] and bi % 250 == 0:
                sw.add_image("original", make_grid(dataloader.denormalize(x)), global_step=step)
                sw.add_image("original reconstructed", make_grid(dataloader.denormalize(x_hat)), global_step=step)
            step += 1
            batch_progress.update()
        sw.add_scalar("Epoch processing time", time.time() - epoch_start_time, global_step=epoch)
        batch_progress.count = 0
        epoch_progress.update()

        # check test accuracy; visualize encodings and reconstructions
        test_acc = evaluate_accuracy(model, test_dl, device)
        sw.add_scalar("Test accuracy", test_acc, global_step=epoch)
        if tparams['train_vae'] and epoch % 5 == 0:
            # plot_embeddings(zs, z_noisys, ys, conditional_ldims, len(classes), epoch, results_dirs['embeddings'], sw=None)
            plot_visual_checks(model, device, sample_batch, results_dirs, dataloader, mparams, epoch)

        # save models and update experiments list
        if epoch == tparams['num_epochs']-1 or test_acc > best_test_acc or loss_recons < best_recons_loss:
            if test_acc > best_test_acc or loss_recons < best_recons_loss:
                model_name = os.path.join(results_dirs['models'], "best_model_" + ("cla" if test_acc > best_test_acc else "vae"))
                model.train()
                save_model(model, loss, epoch, tparams, model_name)
            model_name = os.path.join(results_dirs['models'], f'Epoch_{epoch}_acc_{test_acc:.2f}')
            model_name += f'_vaeloss_{loss_vae:.2f}.tar' if tparams['train_vae'] else ".tar"
            model.train()
            save_model(model, loss, epoch, tparams, model_name)
            update_experiments_summary(results_dirs['main'], test_acc.item(),
                                       sum(losses_recons)/len(losses_recons) if tparams['train_vae'] else 0,
                                       epoch, results_dirs['summaries_file'])
        best_test_acc = np.max((test_acc.item(), best_test_acc))
        best_recons_loss = np.min((loss_recons, best_recons_loss))

    progress_manager.stop()
    training_time = time.time() - start_time
    printl('Training finished!\nTotal Time for Training: %.2fm' % (training_time / 60))
    print("Trained model path:", model_name)
    return model_name


def setup_and_train(config: Union[dict, str], dry_run: bool = False):
    params, config_file = load_params(config)
    dparams, mparams, tparams = params['dataset'], params['model'], params['training']
    assert tparams['train_vae'] or tparams['train_cla'], "Must train at least CLA or VAE"

    # prepare dataset
    dataloader = CustomDataloader(dparams, apply_random_transforms=False)
    train_dl, _, test_dl = dataloader.get_data(dparams['batch_size'])
    classes, attributes = dataloader.get_dataset_params(["classes", "attributes"])

    # prepare model
    mparams['cla_args'] = {'num_classes': len(classes)}
    vae_cla = VaeClassifier(mparams, device, attributes).to(device)
    load_model(vae_cla, mparams, device)

    # output folders for runs (dir structure is results/dataset/date/time_runShortDescription)
    results_dirs = prepare_training_results_dir(params, dry_run)
    if config_file:
        shutil.copy(config_file, os.path.join(results_dirs['main'], "config.txt"))

    # prepare for visualizations to evaluate reconstructions, if needed.
    sample_batch = None
    if tparams['train_vae']:
        print("Collecting train class and attribute samples for visualizations...")
        sample_batch = [get_balanced_batch(dl, dataloader) for dl in [train_dl, test_dl]]
        plot_images(sample_batch, ["Training images", "Test images"],
                    os.path.join(results_dirs['recons'], f'{dparams["dataset"]}_input_images.png'), row_labels=classes)

    # TRAINING
    trained_model_path = train(train_dl, test_dl, vae_cla, mparams, tparams, dataloader, sample_batch, results_dirs, dry_run=dry_run)
    mparams['model_path' if tparams['train_vae'] else 'classifier_path'] = trained_model_path

    # save post training conclusions
    load_model(vae_cla, mparams, device)
    vae_cla.eval()
    final_result_str = "\nModels test accuracy is : {}".format(evaluate_accuracy(vae_cla, test_dl, device, -1))
    print(final_result_str)
    with open(os.path.join(results_dirs['main'], "notes.txt"), "a") as notes_file:
        print(f'Trained model path: {trained_model_path}', file=notes_file)
        print(final_result_str, file=notes_file)
    if tparams['train_vae']:
        plot_visual_checks(vae_cla, device, sample_batch, results_dirs, dataloader, mparams, "Final")
    return vae_cla, trained_model_path, params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model of classifier+encoder+decoder trio.')
    parser.add_argument('--config_file', dest='config_file', help='path to config file containing training specs', required=True)
    args = parser.parse_args()
    setup_and_train(args.config_file)

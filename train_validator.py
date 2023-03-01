# Scripts trains a Discriminator plus Classifier to classify real vs fake images and classes belonging to a dataset:
# produces specification inputs as in verify.py, passes n interps from the decoder to see if they are classified correctly plus real/fake
# explores the conditional dims in steps to check the same thing
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from dataloader import CustomDataloader
from models_impl.generic_nets import conv_Nlayer_downscalar

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
            "CelebA": {"input_shape": [3, 64, 64], "conditional": {"transforms": [["left_shear", "right_shear", "left_rotate", "right_rotate", "towards", "far"]]}},
            "TrafficSignsDynSynth": {"input_shape": [3, 64, 64], "conditional": {"transforms": [["left_shear", "right_shear", "left_rotate", "right_rotate", "towards", "far"]]}},
            "Objects10_3Dpose": {"input_shape": [3, 64, 64], "conditional": {"transforms": [["left_shear", "right_shear", "left_rotate", "right_rotate", "towards", "far"]]}}}


class ClaAttrDet(nn.Module):
    def __init__(self, input_shape, nclasses, attributes):
        super().__init__()
        self.nclasses = nclasses
        self.nconditional_ldims = int(np.sum([len(attrs_list) for attrs_list in attributes]))
        self.model = conv_Nlayer_downscalar(input_shape,
                                            self.nclasses+self.nconditional_ldims,
                                            [128], 4, 4, 5, last_layer_act=False)

    def forward(self, x):
        return self.model(x)

    def decode_logits(self, y):
        return y[:, :self.nclasses], y[:, self.nclasses:self.nclasses+self.nconditional_ldims]


def disc_acc(logits, labels, ratio=True):
    pred_label = torch.round(nn.Sigmoid()(logits))
    corr = torch.sum(pred_label == labels).item()
    if ratio:
        return corr/len(labels)
    return corr


def acc(logits, labels, ratio=True):
    _, pred_label = torch.max(logits, axis=1)
    corr = torch.sum(pred_label == labels).item()
    if ratio:
        return corr/len(labels)
    return corr


def discs_verdict(rf_disc, attr_cla, zs, model, dataloader, y):
    classes, attributes, conditional_losses = dataloader.get_dataset_params(["classes", "attributes", "conditional_loss_fns"])
    n = zs.shape[0]
    recons = model.decoder(zs.to(device)).to(device)
    
    real_label = torch.ones(n).to(device).long()
    real_logit = rf_disc(recons)
    real = disc_acc(real_logit.squeeze(1), real_label, ratio=False)
    
    logits = attr_cla(recons).to("cpu")
    cla_logits, _ = attr_cla.decode_logits(logits)
    correct_classes = acc(cla_logits, torch.Tensor([y]*n), ratio=False)
    return real, correct_classes, torch.round(nn.Sigmoid()(real_logit)), recons


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

    rf_disc = conv_Nlayer_downscalar(dparams['input_shape'], 1, [128], 4, 4, 5, last_layer_act=False).to(device)
    attr_cla = ClaAttrDet(dparams['input_shape'], len(classes), attributes).to(device)
    print(rf_disc, attr_cla)

    nepochs = 10
    cla_loss_cri, attr_loss_cri = nn.CrossEntropyLoss(), nn.CrossEntropyLoss()
    disc_loss_crt = nn.BCEWithLogitsLoss()
    rf_opt = torch.optim.Adam(rf_disc.parameters(), lr=1e-3)
    ac_opt = torch.optim.Adam(attr_cla.parameters(), lr=1e-3)

    progress_manager = enlighten.get_manager()
    epoch_progress = progress_manager.counter(total=nepochs, disc="Epochs", unit="batches")
    batch_progress = progress_manager.counter(total=len(real_train_dl) + len(fake_train_dl), disc="\tBatches", unit="images", leave=False)
    step, best_cla_acc, best_realism_acc, best_attrs_acc = 0, 0, 0, 0
    for epoch in range(nepochs):
        rf_disc.train()
        attr_cla.train()
        for bi, ((rx, y, y_attrs), (fx, _, _)) in enumerate(zip(real_train_dl.get_batch(device), fake_train_dl.get_batch(device))):
            if bi > 20:
                break
            real_label, fake_label = torch.ones(rx.shape[0]).to(device).long(), torch.zeros(fx.shape[0]).to(device).long()
            # DISC
            real_fake_loss = disc_loss_crt(rf_disc(fx).squeeze(1), fake_label.float())
            real_fake_loss += disc_loss_crt(rf_disc(rx).squeeze(1), real_label.float())
            real_fake_loss *= 0.5
            rf_opt.zero_grad()
            real_fake_loss.backward()
            rf_opt.step()

            logits = attr_cla(rx)
            cla_logits, attrdet_logits = attr_cla.decode_logits(logits)
            loss_cla = cla_loss_cri(cla_logits, y)
            # ATTR_DET
            loss_attrs, ai_ind = 0, 0
            for ai, attrs_list in enumerate(attributes):
                nattrs = len(attrs_list)
                loss_attrs += attr_loss_cri(attrdet_logits[:, ai_ind:ai_ind+nattrs], y_attrs[:, ai])
                ai_ind += nattrs
            ac_loss = loss_cla + loss_attrs
            ac_opt.zero_grad()
            ac_loss.backward()
            ac_opt.step()

            batch_progress.update()
            sw.add_scalar("Discriminator loss", real_fake_loss, global_step=step)
            sw.add_scalar("Classifier loss", loss_cla, global_step=step)
            sw.add_scalar("Attribute detection loss", loss_attrs, global_step=step)
            if bi % 250 == 0:
                sw.add_image("real", make_grid(rdataloader.denormalize(rx)), global_step=step)
                sw.add_image("fake", make_grid(fdataloader.denormalize(fx)), global_step=step)
            step += 1
        epoch_progress.update()

        rf_disc.eval()
        attr_cla.eval()
        batch_realism_accs, batch_cla_accs, batch_attrs_accs = [], [], []
        with torch.no_grad():
            for bi, ((rx, y, y_attrs), (fx, _, _)) in enumerate(zip(real_test_dl.get_batch(device), fake_test_dl.get_batch(device))):
                real_label, fake_label = torch.ones(rx.shape[0]).to(device).long(), torch.zeros(fx.shape[0]).to(device).long()
                real_acc = disc_acc(rf_disc(rx).squeeze(1), real_label)
                fake_acc = disc_acc(rf_disc(fx).squeeze(1), fake_label)
                batch_realism_accs.extend([real_acc, fake_acc])

                logits = attr_cla(rx)
                cla_logits, attrdet_logits = attr_cla.decode_logits(logits)
                batch_cla_accs.append(acc(cla_logits, y))
                # ATTR_DET
                attrs_acc, ai_ind = np.inf, 0
                for ai, attrs_list in enumerate(attributes):
                    nattrs = len(attrs_list)
                    attrs_acc = np.min((acc(attrdet_logits[:, ai_ind:ai_ind+nattrs], y_attrs[:, ai]), attrs_acc))  # stricter accuracy
                    ai_ind += nattrs
                batch_attrs_accs.append(attrs_acc)
        batch_cla_acc, batch_realism_acc, batch_attrs_acc = np.mean(batch_cla_accs), np.mean(batch_realism_accs), np.mean(batch_attrs_accs)
        sw.add_scalar("Discriminator accuracy", batch_realism_acc, global_step=step)
        sw.add_scalar("Classifier accuracy", batch_cla_acc, global_step=step)
        sw.add_scalar("Attribute detection accuracy", batch_attrs_acc, global_step=step)
        print(f"Test CLA acc: {batch_cla_acc}, disc acc: {batch_realism_acc}, ATTRS acc: {batch_attrs_acc}")
        if batch_realism_acc > best_realism_acc:
            torch.save(rf_disc, os.path.join(validity_dir, f"{dparams['dataset']}_rf_disc_{start_time}.tar"))
        if batch_cla_acc > best_cla_acc and batch_attrs_acc > best_attrs_acc:
            torch.save(attr_cla, os.path.join(validity_dir, f"{dparams['dataset']}_attr_cla_{start_time}.tar"))
        best_cla_acc = np.max((batch_cla_acc, best_cla_acc))
        best_realism_acc = np.max((batch_realism_acc, best_realism_acc))
        best_attrs_acc = np.max((batch_attrs_acc, best_attrs_acc))

    progress_manager.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Real vs fake + Attributes + Class identifier for the decodings of a trained model (classifier+encoder+decoder).')
    parser.add_argument('--dataset', dest='dataset', type=str, default="", help='Name of the dataset for which the disc_CLA_ATTRDET is to be trained')
    args = parser.parse_args()
    assert args.dataset != "", print(args.dataset, args.model_path)

    dparams = {'dataset': args.dataset, 'input_shape': DEFAULTS[args.dataset]["input_shape"], 'batch_size': 64,
               'conditional': DEFAULTS[args.dataset]["conditional"],
               'conditional_loss_fn': ["CE"]}
    for dp in ['data_balance_method', 'classes']:
        dparams[dp] = []
    disc_cla_dir = os.path.join("data", "validators", args.dataset)

    os.makedirs(disc_cla_dir, exist_ok=True)
    train(dparams, disc_cla_dir)

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.utils import make_grid
from torchvision import models, transforms

import enlighten
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 300  # noqa: E702
import numpy as np
import pandas as pd
from PIL import Image
import os

from verinet.neural_networks.custom_layers import Reshape
from models_impl import create_mirror_decoder
from utils import weights_init, denormalize, test_model_inversion

TEST = False

class CelebADataset(Dataset):
    def __init__(self, root, attributes_file, transform, classes):
        self.root = root
        self.transform = transform
        df = pd.read_table(os.path.join(root, attributes_file), delim_whitespace=True)
        self.images = np.array(df['image_id'].values.tolist())
        self.class_attributes = classes  # df.columns.drop('image_id').values.tolist()
        self.labels = np.array(df[self.class_attributes].values.tolist())

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.get_image(self.images[index]), int(self.labels[index] > 0)

    def get_image(self, img_name):
        x = Image.open(os.path.join(self.root, img_name))
        if self.transform:
            x = self.transform(x)
        return x


class Encoder(nn.Module):
    def __init__(self, enc, device, add_noise: bool = True, inverse_layers: list = []):
        super(Encoder, self).__init__()
        self.enc = enc
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(device)  # sampling on the GPU
        self.N.scale = self.N.scale.to(device)
        self.add_noise = add_noise
        self.inverse_layers = inverse_layers
        
    def forward(self, x):
        mu_lvar = self.enc(x)
        return self.construct_z(mu_lvar), mu_lvar
    
    def construct_z(self, enc_out, add_noise=True):
        assert enc_out.dim() == 2, print(enc_out.dim())
        latent_dims = int(enc_out.shape[-1]/2)
        mu, sigma = enc_out[:, :latent_dims], torch.exp(enc_out[:, latent_dims:])
        z = mu + sigma * (self.N.sample(sigma.shape) if (self.add_noise and add_noise) else 1)
        return z

    def get_inverse_layers(self):
        return self.inverse_layers


class GenModel(nn.Module):
    def __init__(self, enc, dec, device, path: str=None):
        super(GenModel, self).__init__()
        self.enc = enc
        self.dec = dec
        if path:
            ckpt = torch.load(path, map_location=device)
            self.load_state_dict(ckpt["state_dict"])
        self = self.to(device)
        
    def forward(self, x):
        z, mu_lvar = self.enc(x)
        return z, mu_lvar, self.dec(z)


# ours #
class VerificationPipeline(nn.Module):
    def __init__(self, cla, cla_head_nlayers, enc, dec):
        super(VerificationPipeline, self).__init__()
        # self.cla = cla
        self.fdn = cla[:cla_head_nlayers]
        self.classification_head_layers = cla[cla_head_nlayers:]
        self.classification_head = nn.Sequential(self.classification_head_layers)
        self.encoding_head = enc
        self.decoder = dec
        # initialize sub networks
        # self.fdn.apply(weights_init)
        self.decoder.apply(weights_init)
        
    def forward(self, x):
        feat = self.fdn(x)
        z, mu_lvar = self.encoding_head(feat.double())
        y_logits = self.classification_head_layers(feat)
        return y_logits, z, mu_lvar, self.decoder(z.float())


def build_srvp_pipeline(device, cla, input_shape, latent_dims, dec_type, path=None):
    split_layer_indices = {32: -1, 64: -3, 192: -5, 392: -7, 256: -1, 512: -7}
    encoding_head = nn.Linear(2*latent_dims, 2*latent_dims, bias=False).double().to(device)
    rev_encoding_head = nn.Linear(2*latent_dims, 2*latent_dims, bias=False).double().to(device)
    rev_encoding_head.weight.data = nn.Parameter(torch.linalg.inv(encoding_head.weight)).double()
    encoder = Encoder(encoding_head, device, inverse_layers=[rev_encoding_head])
    decoder = get_decoder(dec_type, input_shape, latent_dims)
    srvp = VerificationPipeline(cla, split_layer_indices[latent_dims], encoder, decoder).to(device)
    if path:
        ckpt = torch.load(path, map_location=device)
        srvp.load_state_dict(ckpt["state_dict"], strict=False)
    rev_encoding_head.weight.data = nn.Parameter(torch.linalg.inv(encoding_head.weight)).double()
    test_model_inversion(srvp, latent_dims, device)
    return srvp, rev_encoding_head


# trainings ##
# train classifier manually
def train_cla(cla, cla_path, dl, num_epochs, device):
    opt_cla = torch.optim.Adam(cla.parameters(), lr=5e-3)
    losses, accs = [], []

    progress_manager = enlighten.get_manager()
    epoch_progress = progress_manager.counter(total=num_epochs, desc="Epochs", unit="batches")
    batch_progress = progress_manager.counter(total=len(dl), desc="\tBatches", unit="images", leave=False)
    for epoch in range(num_epochs):
        losses_epoch, accs_epoch = [], []
        for bi, (x, y) in enumerate(dl):
            if TEST and bi > 10:
                break
            x, y = x.to(device), y.to(device)
            y_logits = cla(x)
            loss = nn.CrossEntropyLoss()(y_logits, y)
            _, predLabel = torch.max(y_logits, axis=1)
            acc = torch.sum(predLabel == y)/len(y)
            losses_epoch.append(loss.item())
            accs_epoch.append(acc.item())
            opt_cla.zero_grad()
            loss.backward()
            opt_cla.step()
            batch_progress.update()
        losses.append(np.mean(losses_epoch))
        accs.append(np.mean(accs_epoch))
        batch_progress.count = 0
        epoch_progress.update()
    progress_manager.stop()
    os.makedirs(os.path.dirname(cla_path), exist_ok=True)
    torch.save({"state_dict": cla.state_dict()}, cla_path + ".tar")
    plt.figure()
    plt.plot(losses)
    plt.plot(accs)
    plt.savefig(cla_path + ".png")
    return cla


def test_cla(cla, dl, device, dbatcher: bool = False):
    accs = []
    progress_manager = enlighten.get_manager()
    batch_progress = progress_manager.counter(total=len(dl), desc="\tBatches", unit="images", leave=False)

    if dbatcher:
        for bi, (x, y, _) in enumerate(dl.get_batch(device)):
            if bi > 100:
                break
            x, y = x.to(device), y.to(device)
            y_logits = cla(x)
            _, predLabel = torch.max(y_logits, axis=1)
            acc = torch.sum(predLabel == y) / len(y)
            accs.append(acc.item())
            batch_progress.update()
    else:
        for bi, (x, y) in enumerate(dl):
            if bi > 100:
                break
            x, y = x.to(device), y.to(device)
            y_logits = cla(x)
            _, predLabel = torch.max(y_logits, axis=1)
            acc = torch.sum(predLabel == y)/len(y)
            accs.append(acc.item())
            batch_progress.update()
    progress_manager.stop()
    return np.mean(accs)


# encoder-decoder training
def train_gen_model(gen, opt_gen, gen_path, dl, num_epochs, device, with_cla=False):
    losses_vae, losses_recons, accs = [], [], []
    progress_manager = enlighten.get_manager()
    epoch_progress = progress_manager.counter(total=num_epochs, desc="Epochs", unit="batches")
    batch_progress = progress_manager.counter(total=len(dl), desc="\tBatches", unit="images", leave=False)
    for epoch in range(num_epochs):
        losses_epoch_vae, losses_epoch_recons, accs_epoch = [], [], []
        for bi, (x, y) in enumerate(dl):
            if TEST and bi > 500:
                break
            x, y = x.to(device), y.to(device)
            out = gen(x)
            z, mu_lvar, x_hat = out[-3:]
            if with_cla:
                y_logits = out[0]
                _, predLabel = torch.max(y_logits, axis=1)            
                acc = torch.sum(predLabel == y)/len(y)
                accs_epoch.append(acc.item())
            nld = z.shape[-1]
            mu, sigma = mu_lvar[:, :nld], torch.exp(mu_lvar[:, nld:])
            loss_recons = ((x-x_hat)**2).sum()
            loss_vae = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
            loss = loss_vae + loss_recons
            if torch.isnan(loss) or torch.isinf(torch.abs(loss)):
                print("loss:", loss.item(), loss_vae.item(), loss_recons.item(), torch.isnan(loss).item(), torch.isinf(torch.abs(loss)).item())
                print(x_hat)
                print(mu_lvar)
                return
            losses_epoch_vae.append(loss_vae.item())
            losses_epoch_recons.append(loss_recons.item())
            opt_gen.zero_grad()
            loss.backward()
            opt_gen.step()
            batch_progress.update()
        batch_progress.count = 0
        epoch_progress.update()
        os.makedirs(os.path.dirname(gen_path), exist_ok=True)
        torch.save({"state_dict": gen.state_dict()}, gen_path + ".tar")
        losses_vae.append(np.mean(losses_epoch_vae))
        losses_recons.append(np.mean(losses_epoch_recons))
        plt.figure(figsize=(8, 3))
        plt.plot(losses_vae)
        plt.plot(losses_recons)
        if with_cla:
            accs.append(np.mean(accs_epoch))
            # plt.plot(accs)
            print(accs)
        plt.savefig(gen_path + "_losses.png")
        plt.close()
        fig, axes = plt.subplots(figsize=(80, 10), ncols=2)
        axes[0].imshow(np.transpose(make_grid(denormalize("CelebA", x[:32]), ncols=num_epochs).cpu(), (1, 2, 0)))
        axes[1].imshow(np.transpose(make_grid(denormalize("CelebA", x_hat[:32]), ncols=num_epochs).cpu(), (1, 2, 0)))
        fig.tight_layout()
        plt.savefig(gen_path + "_recons.png")
        plt.close(fig)
    progress_manager.stop()
    return gen

def train_gen_model_all(gen, opt_gen, gen_path, dl, num_epochs, device, with_cla=True):
    losses_vae, losses_recons, accs = [], [], []
    progress_manager = enlighten.get_manager()
    epoch_progress = progress_manager.counter(total=num_epochs, desc="Epochs", unit="batches")
    batch_progress = progress_manager.counter(total=len(dl), desc="\tBatches", unit="images", leave=False)
    loss_cri = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        cla_weight, loss_cla = 0 if not with_cla else epoch/num_epochs, 0
        losses_epoch_vae, losses_epoch_recons, accs_epoch = [], [], []
        for bi, (x, y) in enumerate(dl):
            if TEST and bi > 500:
                break
            x, y = x.to(device), y.to(device)
            out = gen(x)
            z, mu_lvar, x_hat = out[-3:]
            if with_cla:
                y_logits = out[0]
                _, predLabel = torch.max(y_logits, axis=1)
                acc = torch.sum(predLabel == y)/len(y)
                accs_epoch.append(acc.item())
                loss_cla = loss_cri(y_logits, y)
            nld = z.shape[-1]
            mu, sigma = mu_lvar[:, :nld], torch.exp(mu_lvar[:, nld:])
            loss_recons = ((x-x_hat)**2).sum()
            loss_vae = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
            loss = (loss_vae + loss_recons)* (1-cla_weight) + loss_cla * cla_weight
            if torch.isnan(loss) or torch.isinf(torch.abs(loss)):
                print("loss:", loss.item(), loss_vae.item(), loss_recons.item(), torch.isnan(loss).item(), torch.isinf(torch.abs(loss)).item())
                print(x_hat)
                print(mu_lvar)
                return
            losses_epoch_vae.append(loss_vae.item())
            losses_epoch_recons.append(loss_recons.item())
            opt_gen.zero_grad()
            loss.backward()
            opt_gen.step()
            batch_progress.update()
        batch_progress.count = 0
        epoch_progress.update()
        os.makedirs(os.path.dirname(gen_path), exist_ok=True)
        torch.save({"state_dict": gen.state_dict()}, gen_path + ".tar")
        losses_vae.append(np.mean(losses_epoch_vae))
        losses_recons.append(np.mean(losses_epoch_recons))
        plt.figure(figsize=(8, 3))
        plt.plot(losses_vae)
        plt.plot(losses_recons)
        if with_cla:
            accs.append(np.mean(accs_epoch))
            # plt.plot(accs)
            print(accs)
        plt.savefig(gen_path + "_losses.png")
        plt.close()
        fig, axes = plt.subplots(figsize=(80, 10), ncols=2)
        axes[0].imshow(np.transpose(make_grid(denormalize("CelebA", x[:32]), ncols=num_epochs).cpu(), (1, 2, 0)))
        axes[1].imshow(np.transpose(make_grid(denormalize("CelebA", x_hat[:32]), ncols=num_epochs).cpu(), (1, 2, 0)))
        fig.tight_layout()
        plt.savefig(gen_path + "_recons.png")
        plt.close(fig)
    progress_manager.stop()
    return gen


def real_fake_discriminator(dl, disc, srvps, device, disc_path):
    opt_disc = torch.optim.Adam(disc.parameters(), lr=5e-4, betas=(0.5, 0.999))

    lds = [32, 64, 192, 392]
    ld = lds[0]
    losses_disc, accs_disc, corr_real, corr_fake = [], [], [], []
    progress_manager, num_epochs = enlighten.get_manager(), 10
    epoch_progress = progress_manager.counter(total=num_epochs, desc="Epochs", unit="batches")
    batch_progress = progress_manager.counter(total=len(dl), desc="\tBatches", unit="images", leave=False)
    for epoch in range(num_epochs):
        losses_disc_batch, accs_disc_batch, corr_real_batch, corr_fake_batch = [], [], [], []
        for bi, (x, y) in enumerate(dl):
            x = x.to(device)
            real_label, fake_label = 1*torch.ones(x.shape[0]).to(device).long(), torch.zeros(x.shape[0]).to(device).long()

            real_pred_logits = disc(x.detach())
            loss_disc = nn.CrossEntropyLoss()(real_pred_logits, real_label)
            x_out = srvps[ld](x)
            fake_pred_logits = disc(x_out[-1])
            loss_disc += nn.CrossEntropyLoss()(fake_pred_logits, fake_label)
            loss_disc *= 0.5
            losses_disc_batch.append(loss_disc.item())
            # print(real_label.shape, real_pred_logits.shape, torch.max(real_pred_logits, 1))
            corr_real_disc = 1-(torch.sum(real_label - torch.max(real_pred_logits, 1)[1])/(x.shape[0])).item()
            corr_fake_disc = 1-(torch.sum(torch.max(fake_pred_logits, 1)[1] - fake_label)/(x.shape[0])).item()
            corr_real_batch.append(corr_real_disc)
            corr_fake_batch.append(corr_fake_disc)
            accs_disc_batch.append(np.mean([corr_real_disc, corr_fake_disc]))
            opt_disc.zero_grad()
            loss_disc.backward()
            opt_disc.step()
            batch_progress.update()
        losses_disc.append(np.mean(losses_disc_batch))
        corr_real.append(np.mean(corr_real_batch))
        corr_fake.append(np.mean(corr_fake_batch))
        accs_disc.append(np.mean(accs_disc_batch))
        print(f"epoch {epoch}. Acc: {accs_disc[-1]} RealAcc: {corr_real[-1]} FakeAcc: {corr_fake[-1]} Loss_disc: {losses_disc[-1]}")
        torch.save({"state_dict": disc.state_dict()}, disc_path + ".tar")
        batch_progress.count = 0
        epoch_progress.update()
    progress_manager.stop()


# interpolation
def disc_guided_interpolation(endpts, gan_disc, dec, device, gan_path):
    # load the real_vs_fake discriminator
    ckpt = torch.load(gan_path, map_location=device)
    gan_disc.load_state_dict(ckpt["state_dict"], strict=False)
    
    npts = 5  # interp pts to learn between endpoints
    # define interpolation points
    interp_pts = [endpts[0], 
                  torch.autograd.Variable(endpts[0].clone(), requires_grad=True), 
                  torch.autograd.Variable((endpts[0] + endpts[1])*0.5, requires_grad=True),
                  torch.autograd.Variable(endpts[1].clone(), requires_grad=True), 
                  endpts[1]]
    alpha = 1e-4
    disc_real_target = torch.ones(npts-2).to(device)
    for i in range(100):
        # define desirable criteria:
        paths = torch.stack([torch.abs(interp_pts[i+1]-interp_pts[i]) for i in range(npts-1)])
        path_lengths = torch.sum(paths**2, axis=1)
        # equal length constraint
        path_dev, _ = torch.std_mean(path_lengths)
        # shortest path constraint
        path_total = torch.sum(path_lengths)
        # dataset guidance
        disc_out_max, _ = torch.max(gan_disc(dec(torch.stack(interp_pts[1:-1]).to(device))), axis=1)
        disc_out_loss = torch.sum(disc_real_target - disc_out_max)
        interpolation_obj = disc_out_loss + 0.5*path_total + 0.5*path_dev
        interpolation_obj.backward()
        for i in range(1, npts-1):  # update the intermediate points
            interp_pts[i].data = (interp_pts[i] + alpha * interp_pts[i].grad.detach().sign())
            interp_pts[i].grad.zero_()
    return interp_pts

# models
def get_classifier(cla_label, latent_dims, num_classes, device, path=None, in_channels=3):
    if cla_label == "small":
        clal = [nn.Conv2d(in_channels, 4, 4, 2),
                nn.ReLU(),
                nn.Conv2d(4, 4, 4, 2),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(784, 2*latent_dims),
                nn.ReLU(),
                nn.Linear(2*latent_dims, num_classes)]
    elif cla_label == "mid":
        clal = [nn.Conv2d(in_channels, 4, 4, 2),
                nn.ReLU(),
                nn.Conv2d(4, 4, 4, 2),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(784, 256),
                nn.ReLU(),
                nn.Linear(256, 2*latent_dims),
                nn.ReLU(),
                nn.Linear(2*latent_dims, num_classes)]
    elif cla_label == "deep":
        clal = [nn.Conv2d(in_channels, 4, 4, 2),
                nn.ReLU(),
                nn.Conv2d(4, 4, 4, 2),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(784, 384),
                nn.ReLU(),
                nn.Linear(384, 128),
                nn.ReLU(),
                nn.Linear(128, 2*latent_dims),
                nn.ReLU(),
                nn.Linear(2*latent_dims, num_classes)]
    elif cla_label == "deeper":
        clal = [nn.Conv2d(in_channels, 16, 4, 2),
                nn.ReLU(),
                nn.Conv2d(16, 8, 4, 2),
                nn.ReLU(),
                nn.Conv2d(8, 4, 4, 2),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(784, 384),
                nn.ReLU(),
                nn.Linear(384, 128),
                nn.ReLU(),
                nn.Linear(128, 2 * latent_dims),
                nn.ReLU(),
                nn.Linear(2 * latent_dims, num_classes)]
    elif cla_label == "sdeeper":
        clal = [nn.Conv2d(in_channels, 32, 4, 2),
                nn.ReLU(),
                nn.Conv2d(32, 16, 3, 1),
                nn.ReLU(),
                nn.Conv2d(16, 8, 3, 1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(5832, 1024),
                nn.ReLU(),
                nn.Linear(1024, 384),
                nn.ReLU(),
                nn.Linear(384, 128),
                nn.ReLU(),
                nn.Linear(128, 2 * latent_dims),
                nn.ReLU(),
                nn.Linear(2 * latent_dims, num_classes)]
    else:
        raise NotImplementedError(f"Classifier {cla_label} not available.")
    cla = nn.Sequential(*clal).to(device)
    if path:
        ckpt = torch.load(path, map_location=device)
        cla.load_state_dict(ckpt["state_dict"])
    return cla


def get_encoder(enc_label, input_shape, latent_dims, in_channels: int = 3):
    input_size = input_shape[-1]
    if enc_label == "big":
        if input_size == 128:
            return nn.Sequential(nn.Conv2d(in_channels, 4, 4, 2),
                                 nn.ReLU(),
                                 nn.Conv2d(4, 6, 4, 2),
                                 nn.ReLU(),
                                 nn.Conv2d(6, 8, 4, 2),
                                 nn.ReLU(),
                                 nn.Flatten(),
                                 nn.Linear(1568, 768),
                                 nn.ReLU(),
                                 nn.Linear(768, 128),
                                 nn.ReLU(),
                                 nn.Linear(128, 2*latent_dims))
        return nn.Sequential(nn.Conv2d(in_channels, 4, 4, 2),
                             nn.ReLU(),
                             nn.Conv2d(4, 4, 4, 2),
                             nn.ReLU(),
                             nn.Flatten(),
                             nn.Linear(784, 384),
                             nn.ReLU(),
                             nn.Linear(384, 128),
                             nn.ReLU(),
                             nn.Linear(128, 2*latent_dims))
    elif enc_label == "deep":
        return nn.Sequential(nn.Conv2d(in_channels, 4, 5, 2),
                             nn.ReLU(),
                             nn.Conv2d(4, 8, 7, 1),
                             nn.ReLU(),
                             nn.Conv2d(8, 16, 7, 1),
                             nn.ReLU(),
                             nn.Flatten(),
                             nn.Linear(5184, 1280),
                             nn.ReLU(),
                             nn.Linear(1280, 512),
                             nn.ReLU(),
                             nn.Linear(512, 2*latent_dims))
    else:
        raise NotImplementedError(f"Encoder {enc_label} not available.")


def get_decoder(dec_label, input_shape, latent_dims, out_channels: int=3):
    if dec_label == "tiny":
        layer = [nn.Linear(latent_dims, 200),
                 nn.LeakyReLU(1e-2),
                 nn.Linear(200, 1922),
                 nn.LeakyReLU(1e-2),
                 Reshape((-1, 2, 31, 31)),
                 nn.ConvTranspose2d(2, 3, 3, 2),
                 nn.LeakyReLU(1e-2),
                 nn.ConvTranspose2d(3, out_channels, 2, 1)]
    elif dec_label == "small":
        layer = [nn.Linear(latent_dims, 512),
                 nn.LeakyReLU(1e-2),
                 nn.Linear(512, 4805),
                 nn.LeakyReLU(1e-2),
                 Reshape((-1, 5, 31, 31)),
                 nn.ConvTranspose2d(5, 4, 3, 2),
                 nn.LeakyReLU(1e-2),
                 nn.ConvTranspose2d(4, out_channels, 2, 1)]
    elif dec_label == "bigger":
        layer = [nn.Linear(latent_dims, 2048),
                 nn.LeakyReLU(1e-2),
                 nn.Linear(2048, 5766),
                 nn.LeakyReLU(1e-2),
                 Reshape((-1, 6, 31, 31)),
                 nn.ConvTranspose2d(6, 4, 3, 2),
                 nn.LeakyReLU(1e-2),
                 nn.ConvTranspose2d(4, out_channels, 2, 1)]
    elif dec_label == "deeper":
        layer = [nn.Linear(latent_dims, 1024),
                 nn.ReLU(),
                 nn.Linear(1024, 2700),
                 nn.LeakyReLU(1e-2),
                 Reshape((-1, 12, 15, 15)),
                 nn.ConvTranspose2d(12, 9, 3, 2),
                 nn.LeakyReLU(1e-2),
                 nn.ConvTranspose2d(9, 6, 3, 2),
                 nn.LeakyReLU(1e-2),
                 nn.ConvTranspose2d(6, out_channels, 2, 1)]
    elif dec_label == "resnet":
        # note out_channels assumed to be 3, don't expect to need to use ResNet! for (F)MNIST! :)
        layer = create_mirror_decoder(models.resnet18, latent_dims, input_shape)
    else:
        raise NotImplementedError(f"Decoder {dec_label} not available.")
    # base conv models are to upscale to 64x64
    if input_shape[-1] == 128 and dec_label not in ["resnet"]:
        layer.extend([nn.LeakyReLU(), nn.ConvTranspose2d(3, 3, 2, stride=2, padding=0)])
    return nn.Sequential(*layer, nn.Tanh())


def test_gen_model(enc_type, dec_type, input_shape, latent_dims):
    test_img = torch.rand((*input_shape))
    test_lv = get_encoder(enc_type)(test_img)[:, :latent_dims]  # divide into 2
    test_img_hat = get_decoder(dec_type)(test_lv)
    assert test_img.shape == test_img_hat.shape, print(test_img.shape, test_img_hat.shape)
    print(f"Enc {enc_type} + Dec {dec_type} works")
    # print(summary(encoder[enc_type], test_img.shape, device="cpu"))
    # print(summary(decoder[dec_type], test_lv.shape, device="cpu"))


def get_srvp_decoder_ld(cla_label, split_layer):
    if cla_label == "deep":
        if split_layer == -1:
            return get_decoder("deeper"), 32
        elif split_layer in [-3, -5]:
            ld = 64 if split_layer == -3 else 192
            return nn.Sequential(nn.Linear(ld, 2700),
                                 nn.LeakyReLU(1e-2),
                                 Reshape((-1, 12, 15, 15)),
                                 nn.ConvTranspose2d(12, 9, 3, 2),
                                 nn.LeakyReLU(1e-2),
                                 nn.ConvTranspose2d(9, 6, 3, 2),
                                 nn.LeakyReLU(1e-2),
                                 nn.ConvTranspose2d(6, 3, 2, 1)), ld
        elif split_layer == -7:
            return nn.Sequential(nn.Linear(392, 8649),
                                 nn.LeakyReLU(1e-2),
                                 Reshape((-1, 9, 31, 31)),
                                 nn.ConvTranspose2d(9, 6, 3, 2),
                                 nn.LeakyReLU(1e-2),
                                 nn.ConvTranspose2d(6, 3, 2, 1)), 392
        else:
            raise NotImplementedError("SRVP decoder for latent dim {ld} & classifier {cla_label} not implemented.")
    else:
        raise NotImplementedError("SRVP decoder for classifier {cla_label} not implemented.")


def flog(log_path, contents, mode="a+"):
    with open(log_path, mode) as f:
        f.write("\n")
        for content in contents:
            f.write(f"{content}\n")
        f.write("\n")


def save_batch_images(xb, dirpath, bi):
    os.makedirs(dirpath, exist_ok=True)
    for xi, x in enumerate(xb):
        img = transforms.ToPILImage()(x.detach())
        img.save(f"{dirpath}/{bi}_{xi}.png")

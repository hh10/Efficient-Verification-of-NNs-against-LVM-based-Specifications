import torch
import torch.nn as nn
from torchinfo import summary
import torch.nn.functional as F

import numpy as np

from utils import weights_init, printl, get_conditional_mean
from models_impl.submodels import FDN, ClassificationHead, InvertibleEncodingHead, Decoder


class VaeInvertibleEncodingHead(nn.Module):
    def __init__(self, mparams, device, attributes, conditional_losses, add_noise):
        super(VaeInvertibleEncodingHead, self).__init__()
        self.attributes = attributes
        self.head = InvertibleEncodingHead(mparams, pnultimate_x=False)
        self.latent_dims = mparams['latent_dim']
        self.add_noise = add_noise
        self.conditional_losses = conditional_losses

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(device)  # sampling on the GPU
        self.N.scale = self.N.scale.to(device)
        self.device = device
        self.loss_kl, self.loss_conditional, self.batch_mean_grads = None, None, None

    def forward(self, x, add_noise: bool = True, y_attrs=None):
        self.loss_kl, self.loss_conditional, self.batch_mean_grads = None, None, None
        mu_lvar, _ = self.head(x)
        mu, full_sigma = self.__mu_sigma_from_enc_out(mu_lvar)
        z = self.__construct_z(mu, full_sigma, add_noise, y_attrs)
        if y_attrs is not None:  # implies training, so update losses
            self.__update_encoding_losses(z, mu, full_sigma, y_attrs)
        return z, mu_lvar

    def construct_z(self, enc_out, add_noise: bool = True, y_attrs=None):
        assert enc_out.dim() == 2, print(enc_out.dim())
        mu, full_sigma = self.__mu_sigma_from_enc_out(enc_out)
        return self.__construct_z(mu, full_sigma, add_noise, y_attrs=y_attrs)

    def get_encoding_losses(self):
        # make sure that losses are updated after forward pass
        assert any(elem is not None for elem in [self.loss_kl, self.loss_conditional])
        return self.loss_kl, self.loss_conditional

    def get_inverse_layers(self):
        return self.head.get_inverse_layers()

    def mu_sigma_from_enc_out(self, enc_out):
        return self.__mu_sigma_from_enc_out(enc_out)

    def __mu_sigma_from_enc_out(self, enc_out):
        assert enc_out.shape[-1] == 2*self.latent_dims
        # exp for sigma to be positive
        return enc_out[:, :self.latent_dims], torch.exp(0.5*enc_out[:, self.latent_dims:])

    def __construct_z(self, mu, full_sigma, add_noise, y_attrs):
        """ Constructs latent var z from encoding head output based on conditioning and reparameterization trick """
        z, z_ind, add_noise = mu.clone(), 0, add_noise and self.add_noise
        z_ind = self.__compute_conditional_dim(z, mu, full_sigma, add_noise, y_attrs)
        # if add_noise, reparametrization to sample a new latent variable from underlying distribution
        # else just add mu and sigma (ONLY for continuous dimensions)
        sigma = full_sigma[:, z_ind:]
        z[:, z_ind:] = mu[:, z_ind:] + sigma * (self.N.sample(sigma.shape) if add_noise else 1)
        return z

    def __update_encoding_losses(self, z, mu, full_sigma, y_attrs):
        z_ind, self.loss_conditional = self.__compute_conditional_loss(z, mu, full_sigma, y_attrs)
        # KL divergence loss between modelled distribution and normal distribution (regularization term in ELBO loss)
        sigma = full_sigma[:, z_ind:]
        self.loss_kl = (sigma**2 + mu[:, z_ind:]**2 - torch.log(sigma) - 1/2).sum()

    def __compute_conditional_dim(self, z, mu, full_sigma, add_noise, y_attrs):
        """ Computes the value of the conditional dimensions:
         - for discrete attributes, it uses Sigmoid or Softmax to indicate whether or which one of the multiple attributes is present
         - for continuous losses:
            - KL expects latent values normally distributed around well separated means in a single dimension
            - Generator (while training) changes the constant conditional dimension in a batch with their mean value for the
              batch while keeping the varying dimension values as is. (Only one dimension varies per batch)
        """
        z_ind = 0
        for ai, attrs_list in enumerate(self.attributes):
            closs = self.conditional_losses[ai]

            if closs == "CE":
                ncdims = len(attrs_list)
                if ncdims == 1:
                    z[:, z_ind] = nn.Sigmoid()(mu[:, z_ind])
                else:
                    z[:, z_ind:z_ind+ncdims] = nn.Softmax(dim=1)(mu[:, z_ind:z_ind+ncdims])

            elif closs == "KL":
                ncdims = 1
                sigma_i = full_sigma[:, z_ind]
                z[:, z_ind] = mu[:, z_ind] + sigma_i * (self.N.sample(sigma_i.shape) if add_noise else 1)

            elif closs == "Generator":
                ncdims = len(self.conditional_losses)
                if y_attrs is not None and len(y_attrs) > 1:
                    # None means no TRAINING with Generator closs (todo(hh): add sw to monitor diff to mean)
                    z_mean = torch.mean(z[:, z_ind:z_ind+ncdims], 0)  # [cond_dims]
                    vary_attr = torch.tensor(self.__get_varying_attr(y_attrs)).to(self.device)
                    z_1hot = F.one_hot(vary_attr, num_classes=ncdims)  # [BS, cond_dims]
                    z[:, z_ind:z_ind+ncdims] = z_1hot*z[:, z_ind:z_ind+ncdims] + (1-z_1hot)*z_mean
                z_ind += ncdims
                return z_ind  # currently Generator loss cannot be used in combination with another loss

            else:
                raise NotImplementedError(f'{closs} not implemented as encoding loss')
            z_ind += ncdims
        return z_ind

    def __compute_conditional_loss(self, z, mu, full_sigma, y_attrs):
        """ Computes the value of the conditional dimensions:
         - for discrete attributes, it uses BCE or CE loss to encourage whether or which one of the multiple attributes is present
         - for continuous losses:
            - KL encourages latent values to be normally distributed around conditional means in a single dimension
            - Generator applies KL divergence loss for the varying conditional dimension
        """
        z_ind, loss_conditional = 0, 0
        for ai, attrs_list in enumerate(self.attributes):
            closs = self.conditional_losses[ai]
            y_attr = y_attrs[:, ai].to(self.device)

            if closs == "CE":
                ncdims = len(attrs_list)
                if ncdims == 1:
                    loss_conditional += nn.BCEWithLogitsLoss()(mu[:, z_ind], y_attr.to(torch.float32))  # work with mu here because z already has sigmoid
                else:
                    loss_conditional += nn.CrossEntropyLoss()(mu[:, z_ind:z_ind+ncdims], y_attr)  # work with mu here because z already has softmax

            elif closs == "KL":
                ncdims = 1
                # get the conditional mean index from attribute value based on the # of setpoints for the attribute
                mean_setpt = get_conditional_mean(y_attr, len(attrs_list))
                sigma_i = full_sigma[:, z_ind]
                loss_conditional += (sigma_i**2 + (mu[:, z_ind] - mean_setpt.unsqueeze(1))**2 - torch.log(sigma_i) - 1/2).sum()

            elif closs == "Generator":
                ncdims = len(self.conditional_losses)
                mu_i, sigma_i = mu[:, z_ind:z_ind+ncdims], full_sigma[:, z_ind:z_ind+ncdims]
                conditional_kl = sigma_i**2 + mu_i**2 - torch.log(sigma_i) - 1/2
                # apply conditional KL only for the varying capturing dimension
                vary_attr = torch.tensor(self.__get_varying_attr(y_attrs)).to(self.device)
                z_1hot = F.one_hot(vary_attr, num_classes=ncdims)  # [BS, z_dim]
                loss_conditional += (z_1hot * conditional_kl).sum()
                z_ind += ncdims
                return z_ind, loss_conditional  # currently Generator loss cannot be used in combination with another loss

            else:
                raise NotImplementedError(f'{closs} not implemented as encoding loss')
            z_ind += ncdims
        return z_ind, loss_conditional

    def __get_varying_attr(self, attrs):
        atr = attrs.cpu().numpy().transpose()
        for ti, tt in enumerate(atr):
            if len(np.unique(tt)) > 1:
                return ti
        # all attributes are constant in batch
        attrs_options = [len(attrs_list) for attrs_list in self.attributes]
        return np.argmax(attrs_options)


class VaeClassifier(nn.Module):
    """ The multihead pipeline that:
     - instantiates a classifier -> splits it into FDN and classification_head (based on mparams["fdn_till_layer"])
     - adds an encoding head and decoder to the FDN
    """
    def __init__(self, mparams, device, attributes, add_variational_noise: bool = True):
        super(VaeClassifier, self).__init__()
        self.include_vae = mparams['train_vae']

        self.fdn = FDN(mparams, out_dims=2*mparams['latent_dim'])
        self.classification_head = ClassificationHead(mparams, in_dims=2*mparams['latent_dim'])
        if self.include_vae:
            self.encoding_head = VaeInvertibleEncodingHead(mparams, device, attributes, mparams['conditional_loss_fn'], add_variational_noise)
            self.decoder = Decoder(mparams, mparams['latent_dim'])
        self.summarize(mparams['fdn_args']['x_shape'], mparams['latent_dim'], device)

        # initialize sub networks
        self.fdn.apply(weights_init)
        if self.include_vae:
            self.decoder.apply(weights_init)
        '''
        # in case of severe difference in model.train() and model.eval(), esp. with smaller batch sizes
        # https://discuss.pytorch.org/t/performance-highly-degraded-when-eval-is-activated-in-the-test-phase/3323/15
        for child in self.fdn.children():
            for i in range(len(child)):
                if type(child[i])==nn.BatchNorm2d:
                    child[i].track_running_stats = False
        '''

    def verification_model_layers(self):
        if not self.include_vae:
            return []
        return self.encoding_head.get_inverse_layers() + self.classification_head.get_layers()

    def summarize(self, input_shape, latent_dims, device):
        """ Gives the pipeline summary per subnetwork """
        printl("Feature Detection Network")
        fdns = summary(self.fdn, input_size=(1, *input_shape), device='cpu')
        printl("Classification Head")
        summary(self.classification_head, input_size=fdns.summary_list[-1].output_size, device='cpu')
        if self.include_vae:
            printl("Encoding Head")
            encs = summary(self.encoding_head, input_size=fdns.summary_list[-1].output_size, device=device)
            assert encs.summary_list[-2].output_size[1] == 2*latent_dims, print(encs.summary_list[-2].output_size, latent_dims)
            del fdns, encs
            printl("Decoder")
            decs = summary(self.decoder, input_size=(1, latent_dims), device='cpu')
            assert decs.summary_list[-1].output_size[1:] == input_shape, print(decs.summary_list[-1].output_size, input_shape)

    def forward(self, x, only_cla=False, only_gen_z=False, y_attrs=None):
        features = self.fdn(x)
        y_hat = self.classification_head(features)
        if not self.include_vae or only_cla:
            return y_hat
        z, z_mu_lvar = self.encoding_head(features, y_attrs=y_attrs)
        if only_gen_z:
            return y_hat, z, z_mu_lvar
        x_hat = self.decoder(z)
        return features, y_hat, z, z_mu_lvar, x_hat

    def get_encoding_losses(self):
        return self.encoding_head.get_encoding_losses()

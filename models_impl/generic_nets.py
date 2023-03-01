import torch
import torch.nn as nn

import numpy as np
import copy


def conv_Nlayer_downscalar(x_shape, out_dims, linear_dims, nconv_layers=2, nf=8, ks=7, last_layer_act=True):
    assert x_shape[1] == x_shape[2]
    layers = []

    nfs = [x_shape[0]]
    nfs.extend(np.arange(1, nconv_layers+1)*nf)
    for i in range(nconv_layers):
        layers.extend([nn.Conv2d(nfs[i], nfs[i+1], ks, 1),
                       nn.BatchNorm2d(nfs[i+1]),
                       # nn.LeakyReLU(0.2, inplace=True)])
                       nn.ReLU(inplace=True)])
    
    layers.append(nn.Flatten())
    
    int_shape = (x_shape[1] - (ks-1)*nconv_layers)
    linear_dims = [nfs[nconv_layers]*int_shape*int_shape, *linear_dims, out_dims]
    for i, _ in enumerate(linear_dims[:-1]):
        layers.append(nn.Linear(linear_dims[i], linear_dims[i+1]))
        if last_layer_act or i != len(linear_dims[:-1]):
            # layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def conv_Nlayer_upscalar(in_dims, x_shape, linear_dims, nconv_layers=2, nf=8, ks=7):
    assert x_shape[1] == x_shape[2]
    int_shape = (x_shape[1] - (ks-1)*nconv_layers)
    nfs = list(np.arange(nconv_layers, 0, -1) * nf)
    nfs.append(x_shape[0])
    layers = []

    linear_dims = [in_dims, *linear_dims, nfs[0]*int_shape*int_shape]
    for i, _ in enumerate(linear_dims[:-1]):
        layers.extend([nn.Linear(linear_dims[i], linear_dims[i+1]),
                       nn.ReLU(inplace=True)])

    layers.append(nn.Unflatten(1, (int(nfs[0]), int_shape, int_shape)))

    for i in range(nconv_layers):
        layers.append(nn.ConvTranspose2d(nfs[i], nfs[i+1], ks, 1))
        if i != nconv_layers-1:
            layers.extend([nn.BatchNorm2d(nfs[i+1]),
                           nn.ReLU(inplace=True)])
    return nn.Sequential(*layers)


class LinearNLayerEncoder(nn.Module):  # don't remove nn.Module for backward compatibility with trained models
    def __init__(self, N, latent_dims):
        super(LinearNLayerEncoder, self).__init__()
        # keep sequential non-bipartite arch
        layers = []
        for i in range(N-1):
            layers.append(nn.Linear(2*latent_dims, 2*latent_dims, bias=False).double())
            layers.append(nn.LeakyReLU(5e-2).double())
        layers.append(nn.Linear(2*latent_dims, 2*latent_dims, bias=False).double())
        # layers[-1].weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.layers = layers
        self.model = nn.Sequential(*self.layers)  # don't remove nn.Module for backward compatibility with trained models

        # linear equivs. to torch.mm(inputs, self.model.weight.T).add(linear.bias)
        reverse_layers = copy.deepcopy(self.layers)
        reverse_layers.reverse()
        for rlayer in reverse_layers:
            if isinstance(rlayer, nn.LeakyReLU):
                rlayer.negative_slope = 1./rlayer.negative_slope
        self.reverse_layers = reverse_layers

    def get_inverse_layers(self):
        for ri, rlayer in enumerate(self.reverse_layers):
            if isinstance(rlayer, nn.Linear):
                rlayer.weight.data = nn.Parameter(torch.linalg.inv(self.layers[-ri-1].weight)).double()
        return self.reverse_layers


class NFLayerEncoder(nn.Module):  # don't remove nn.Module for backward compatibility with trained models
    def __init__(self, N, latent_dims):
        super(NFLayerEncoder, self).__init__()
        ndimHalf = int(latent_dims)
        self.W1 = nn.Linear(ndimHalf, ndimHalf, bias=True)
        self.S1 = nn.Parameter(torch.randn((ndimHalf)))
        self.W2 = nn.Linear(ndimHalf, ndimHalf, bias=True)
        self.S2 = nn.Parameter(torch.randn((ndimHalf)))
        self.W3 = nn.Linear(ndimHalf, ndimHalf, bias=True)
        self.S3 = nn.Parameter(torch.randn((ndimHalf)))
        self.W4 = nn.Linear(ndimHalf, ndimHalf, bias=True)
        self.S4 = nn.Parameter(torch.randn((ndimHalf)))
        self.ndimHalf = ndimHalf

    def forward(self, v):
        v = v.float()
        z1_l = v[:, :self.ndimHalf] * torch.exp(self.S1) + self.W1(v[:, self.ndimHalf:])
        z1_u = v[:, self.ndimHalf:] * torch.exp(self.S2) + self.W2(v[:, :self.ndimHalf])
        z = torch.concat((z1_l, z1_u), axis=1)
        z *= 0.01
        z = nn.LeakyReLU(1e-1)(z)
        z2_l = z[:, self.ndimHalf:] * torch.exp(self.S4) + self.W4(z[:, :self.ndimHalf])
        z2_u = z[:, :self.ndimHalf] * torch.exp(self.S3) + self.W3(z[:, self.ndimHalf:])
        return torch.concat((z2_l, z2_u), axis=1)

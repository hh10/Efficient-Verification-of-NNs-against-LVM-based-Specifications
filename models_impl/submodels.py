import torch.nn as nn
import torchvision
from torchvision import models

from models_impl.generic_nets import conv_Nlayer_downscalar, conv_Nlayer_upscalar, LinearNLayerEncoder, NFLayerEncoder
from verifiers_utils import get_verifier_reshape_op


def lrelu():
    return nn.LeakyReLU(2e-2)


class FDN(nn.Module):
    def __init__(self, mparams, out_dims):
        super().__init__()

        if mparams['source'] == 'hub':
            model = getattr(models, mparams['name'])(pretrained=mparams['pretrained'])
            fdn_till_layer = int(mparams['fdn_till_layer'])
            layers = list(model.children())[:fdn_till_layer]
            if mparams['name'] == 'mobilenet_v2':
                layers.append(nn.AdaptiveAvgPool2d((1, 1)))
            self.model = nn.Sequential(*layers)
        elif mparams['source'] == 'generic_conv':
            self.model = conv_Nlayer_downscalar(out_dims=out_dims, **mparams['fdn_args'])
        else:
            raise ValueError(f'FDN for {mparams["source"]} not implemented')

    def forward(self, x):
        return self.model(x)


class ClassificationHead(nn.Module):
    def __init__(self, mparams, in_dims):
        super().__init__()
        self.mparams = mparams
        if mparams['source'] == 'hub':
            model = getattr(models, mparams['name'])(pretrained=mparams['pretrained'])
            fdn_till_layer = int(mparams['fdn_till_layer'])
            model_actual_layers = list(model.children())[fdn_till_layer:]

            def get_layers(malayers, mlayers):
                for i, layer in enumerate(malayers):
                    if isinstance(layer, nn.Sequential):
                        get_layers(layer, mlayers)
                    else:
                        mlayers.append(layer)
            model_layers = []
            get_layers(model_actual_layers, model_layers)

            for i, layer in enumerate(model_layers):
                if type(layer) == nn.Linear:
                    break
            model_layers.insert(i, nn.Flatten())
            last_layer_in_feats = model_layers[-1].in_features
            self.model = nn.Sequential(*(model_layers[:-1]), nn.ReLU(), nn.Linear(last_layer_in_feats, mparams['cla_args']['num_classes']))
        else:
            self.model = nn.Sequential(nn.ReLU(), nn.Linear(in_dims, mparams['cla_args']['num_classes']))

    def forward(self, x):
        return self.model.float()(x)

    def get_layers(self):
        return list(self.model.children())


class Decoder(nn.Module):
    def __init__(self, mparams, in_dims):
        super().__init__()

        if mparams['source'] == 'hub':
            if mparams['name'] == 'mobilenet_v2':
                model = create_mobilenet_mirror_decoder(getattr(models, mparams['name']))
            else:
                model = create_mirror_decoder(getattr(models, mparams['name']), mparams['latent_dim'], mparams['fdn_args']['x_shape'])
        elif mparams['source'] == 'generic_conv':
            model = conv_Nlayer_upscalar(in_dims=in_dims, **mparams['fdn_args'])
        else:
            raise ValueError(f'Decoder for {mparams["source"]} not implemented')
        last_activation = nn.Tanh
        if mparams["dataset"] in ["MNIST", "FashionMNIST", "Zappos50k", "Objects10_3Dpose"]:
            last_activation = nn.Sigmoid
        self.model = nn.Sequential(*model, last_activation())

    def forward(self, x):
        return self.model(x.float())


class InvertibleEncodingHead(nn.Module):
    def __init__(self, mparams, pnultimate_x=False):
        super().__init__()
        # self.head = NFLayerEncoder(mparams['enc_args']['linear_layers'], mparams['latent_dim'])
        self.head = LinearNLayerEncoder(mparams['enc_args']['linear_layers'], mparams['latent_dim'])
        if pnultimate_x:
            self.last_layer = self.head.layers[-1]
            self.pnultimate_model = nn.Sequential(nn.Flatten(), *self.head.layers[:-1])
        else:
            self.pnultimate_model = None
            # self.model = nn.Sequential(nn.Flatten(), self.head)
            self.model = nn.Sequential(nn.Flatten(), *self.head.layers)

    def forward(self, x):
        if self.pnultimate_model:
            pnultimate_x = self.pnultimate_model(x.double())
            return self.last_layer(pnultimate_x), pnultimate_x
        return self.model(x.double()), None

    def get_inverse_layers(self):
        return self.head.get_inverse_layers()


# todo(hh): generalize for more model types: ResNetX, mobileNetV3
def create_mirror_decoder(downscaler_net, latent_dims, input_shape):
    dlayers, last_out_features = [], -1
    un_layers = list(downscaler_net().children())
    num_layers = len(un_layers) - 4 if input_shape[-1] == 64 else len(un_layers)
    un_layers.reverse()
    for i, layer in enumerate(un_layers):
        if type(layer) == nn.Linear:
            dlayer = [nn.Linear(latent_dims, layer.in_features), get_verifier_reshape_op()((-1, layer.in_features, 1, 1))]
            last_out_features = layer.in_features
        elif type(layer) in [nn.AdaptiveAvgPool2d]:
            dlayer = [nn.ConvTranspose2d(last_out_features, last_out_features, 4, stride=2, padding=0 if i == 1 else 1), lrelu()]
        elif type(layer) in [nn.MaxPool2d]:
            dlayer = [nn.ConvTranspose2d(last_out_features, last_out_features, 2, stride=2, padding=0), lrelu()]
        elif type(layer) == nn.Sequential:
            in_features = list(list(layer.children())[0].children())[0].out_channels
            out_features = list(list(layer.children())[0].children())[0].in_channels
            dlayer = [nn.ConvTranspose2d(in_features, out_features, 4, stride=2, padding=1),
                      nn.BatchNorm2d(out_features), lrelu()]
            last_out_features = out_features
        elif type(layer) == nn.Conv2d and i < len(un_layers) - 1:
            dlayer = [nn.ConvTranspose2d(layer.out_channels, layer.in_channels, 3, stride=1, padding=1)]
            last_out_features = layer.in_channels
        else:
            continue
        dlayers.extend(dlayer)
        if i > num_layers - 2:
            break
    dlayers.append(nn.ConvTranspose2d(last_out_features, 3, 3, stride=1, padding=1))
    return dlayers


def create_mobilenet_mirror_decoder(downscaler_net):
    indices = [4, 8, 12, 16]
    dlayers = []
    un_layers = list(list(downscaler_net().children())[0].children())
    un_layers.reverse()
    for i, layer in enumerate(un_layers):
        if type(layer) == torchvision.ops.misc.ConvNormActivation:
            in_channels = list(layer.children())[0].out_channels
            out_channels = list(layer.children())[0].in_channels
            # print("ConvNormActivation", in_channels, out_channels)
            dlayer = []
            if i == 0:
                dlayer.extend([nn.Unflatten(1, (int(in_channels/2), 1, 1)),
                               nn.ConvTranspose2d(int(in_channels/2), out_channels, 4, stride=2, padding=1)])
            else:
                dlayer.append(nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1))
            if i < len(un_layers)-1:
                dlayer.append(nn.ReLU())
        elif type(layer) == torchvision.models.mobilenetv2.InvertedResidual:
            # print("InvertedResidual")
            lchildren = list(list(layer.children())[0].children())
            in_channels = lchildren[-2].out_channels
            out_channels = list(lchildren[0].children())[0].in_channels
            if i in indices:
                dlayer = [nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1), nn.ReLU()]
            else:
                dlayer = [nn.ConvTranspose2d(in_channels, out_channels, 3, stride=1, padding=1), nn.ReLU()]
        else:
            continue
        dlayers.extend(dlayer)
    return nn.Sequential(*dlayers)

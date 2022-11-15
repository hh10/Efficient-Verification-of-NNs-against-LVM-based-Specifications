import torch
import torch.nn as nn
from torchvision import models, transforms
from torchinfo import summary

import numpy as np
from tempfile import NamedTemporaryFile
import shutil
import csv
import os
import enlighten
import json
import random
from datetime import datetime
from typing import Union
import tempfile


def test_model_inversion(model, latent_dims, device):
    tol = 1e-3
    rand_x = torch.rand((1, 2*latent_dims)).to(device)
    with torch.no_grad():
        _, z_mu_sigma = model.encoding_head(rand_x.double())
        x_hat = nn.Sequential(*model.encoding_head.get_inverse_layers())(z_mu_sigma.double())
        max_diff = torch.max(torch.abs(x_hat-rand_x))
    assert max_diff < tol, f'Original feats: {rand_x[0, :10]}\nInverted feats: {x_hat[0, :10]} (maximum inversion difference was {max_diff})'
    printl(f'Model inversion verified w.r.t {tol} (maximum inversion difference was {max_diff})')


def load_params(config: Union[dict, str]) -> dict:
    if type(config) is dict:
        cf = tempfile.NamedTemporaryFile("w", delete=False)
        cf.write(json.dumps(config, indent=4))
        cf.close()
        return config, cf.name
    print(f'Loading params from {config}')
    with open(config, "r") as config_file:
        params = config_file.read()
    params = json.loads(params)
    if "fdn_args" in params["model"]:
        params["model"]['fdn_args']['x_shape'] = params['input_shape']
    else:
        params["model"]['fdn_args'] = {'x_shape': params['input_shape']}
    
    mparams = params['model']
    latent_dims = params['latent_dim']
    if mparams['source'] == 'hub':
        tmodel = getattr(models, mparams['name'])()
        msummary = summary(tmodel, input_size=(1, *params['input_shape']), device='cpu', verbose=0)
        latent_dims = np.prod(msummary.summary_list[int(mparams['fdn_till_layer'])].input_size)
        assert latent_dims % 2 == 0  # need latent dims to be even
        latent_dims = int(latent_dims / 2)
    mparams['latent_dim'] = latent_dims
    print("Latent dims:", mparams['latent_dim'])
    assert 'classifier_path' not in params or 'model_path' not in params

    subdict = lambda d, cparams: {x: d[x] if x in d else None for x in cparams}  # noqa: E731
    return {**subdict(params, ['desc', 'notes']),
            "dataset": subdict(params, ['dataset', 'input_shape', 'data_balance_method', 'classes', 'conditional', 'conditional_loss_fn', 'batch_size']),
            "model": {**mparams, **subdict(params, ['dataset', 'model_path', 'classifier_path', 'conditional_loss_fn', 'train_vae'])},
            "training": subdict(params, ['num_epochs', 'lr', 'train_cla', 'train_vae', 'GAN_start_training_epochs', 'only_vae_training_epochs',
                                         'only_cla_training_epochs', 'loss_conditional_weight', 'loss_recons_weight', 'loss_kl_weight'])}, config


def load_model(model, mparams, device, optimizer=None):
    if mparams['train_vae']:
        test_model_inversion(model, mparams['latent_dim'], device)  # ensure model inversion!!

    path = mparams['model_path'] or mparams['classifier_path']
    if path is None:
        return
    ckpt = torch.load(path, map_location=device)
    if mparams['model_path'] is not None:
        print(f'Loading MODEL from {mparams["model_path"]}')
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        print(f'Loading CLA from {mparams["classifier_path"]}')
        model.fdn.load_state_dict(ckpt['fdn_state_dict'])
        model.classification_head.load_state_dict(ckpt['classifier_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    return  # ckpt['loss'], ckpt['epoch'], ckpt['params']


def save_model(model, loss, epoch, params, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'fdn_state_dict': model.fdn.state_dict(),
        'classifier_state_dict': model.classification_head.state_dict(),
        # 'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'params': params,
        }, path)
    torch.save({
        'fdn_state_dict': model.fdn.state_dict(),
        'classifier_state_dict': model.classification_head.state_dict(),
        }, path + "_classifier")


def accuracy_metric(logits, labels):
    _, predLabel = torch.max(logits, axis=1)
    return torch.sum(predLabel == labels)/len(labels)


def evaluate_accuracy(model, dl, device, max_batch=250, show_progress: bool = False):
    model.eval()
    batch_accs, progress_manager = [], None
    if show_progress:
        progress_manager = enlighten.get_manager()
        test_progress = progress_manager.counter(total=np.min((max_batch, len(dl))), desc="Batches", unit="batch")
    for i, (x, y, _) in enumerate(dl.get_batch(device)):
        with torch.no_grad():
            y_logits = model(x, only_cla=True)
        batch_accs.append(accuracy_metric(y_logits, y))
        if progress_manager is not None:
            test_progress.update()
        if i == max_batch - 1:
            break
    if progress_manager is not None:
        progress_manager.stop()
    return torch.stack(batch_accs).mean()


def printl(text):
    print('\n'); print("-" * 25); print(text); print("-" * 25)  # noqa: E702


def weights_init(m):
    """
    Initialise weights of the model.
    """
    if type(m) == nn.ConvTranspose2d or type(m) == nn.Conv2d:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif type(m) == nn.BatchNorm2d:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def update_experiments_summary(model_path, test_acc, loss_recons, epoch, filename):
    new_row = {'model_path': model_path, 
               'epochs': epoch,
               'test_acc': "{:.2f}".format(test_acc),
               'loss_recons': "{:.2f}".format(loss_recons)}
    fields = list(new_row.keys())
    if not os.path.exists(filename):
        with open(filename, 'w+') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fields)
            writer.writeheader()
            writer.writerow(new_row)
        return

    tempfile = NamedTemporaryFile(mode='w', delete=False)
    updated = False    
    with open(filename, 'r') as csvfile, tempfile:
        reader = csv.DictReader(csvfile, fieldnames=fields)
        writer = csv.DictWriter(tempfile, fieldnames=fields)
        for row in reader:
            if row['model_path'] == new_row['model_path']:
                row = new_row
                updated = True
            writer.writerow(row)
        if not updated:
            writer.writerow(new_row)
    shutil.move(tempfile.name, filename)


def get_balanced_batch(dl, dataloader, nimages_per_class=4, nimages_per_attr=1):
    """ Returns images with n-number of images per attribute and class """
    dataset_name, classes, attributes = dataloader.get_dataset_params(["dataset_name", "classes", "attributes"])
    progress_manager = enlighten.get_manager()
    class_progress, attr_progress = progress_manager.counter(total=len(classes), desc="Classes", unit="class"), None

    if type(attributes) == list and len(attributes) > 0:
        num_attrs = np.sum([len(attrs_list) for attrs_list in attributes])
        nimages_per_class = nimages_per_attr*num_attrs
        attr_progress = progress_manager.counter(total=num_attrs, desc="Attributes", unit="attr")

    def get_class_inputs_with_attributes(attr_index, class_index):
        x_attrs, attrs_list = [], attributes[attr_index]
        if np.all(attrs_list == classes) or ("Non" in classes[1] and attrs_list[0] == classes[0]):
            # checks against attributes being same as class
            return x_attrs
        for aj, atr in enumerate(attrs_list):
            x_atr, des_attr_label = [], aj
            for kk, (x, y, y_attrs) in enumerate(dl.get_batch("cpu")):
                x_c = x[torch.nonzero((y == class_index) * (y_attrs[:, attr_index] == des_attr_label))]
                if x_c.shape[0] > 0:
                    x_atr.extend(x_c[:nimages_per_attr-len(x_atr)])
                    if len(x_atr) >= nimages_per_attr:
                        if dataset_name != "CelebA" or (len(attrs_list) > 1 or des_attr_label == 1):
                            # look for next attribute, but if CelebA and single attribute, then also collect samples of reversed attribute
                            break
                        x_attrs.extend(x_atr)
                        x_atr, des_attr_label = [], 1
            assert len(x_atr) > 0, print(f"No input of class {classes[class_index]} with attr {atr} ({des_attr_label}) found!")
            x_attrs.extend(x_atr)
            attr_progress.update()
        return x_attrs

    x_classes = []
    for ci, c in enumerate(classes):
        x_class = []
        if attr_progress is not None:  # collection is per attribute
            attr_progress.count = 0
            for ai, _ in enumerate(attributes):
                ai_inputs = get_class_inputs_with_attributes(ai, ci)
                x_class.extend(ai_inputs)
        else:  # collection is only class wise
            for kk, (x, y, _) in enumerate(dl.get_batch("cpu")):
                x_c = x[torch.nonzero((y == ci))]
                if x_c.shape[0] != 0:
                    x_class.extend(x_c[:nimages_per_class-len(x_class)])
                    if len(x_class) >= nimages_per_class:
                        break

        x_classes.extend(x_class)
        class_progress.update()
    progress_manager.stop()
    out = torch.stack(x_classes, dim=0).squeeze(1)
    assert out.dim() == 4, print(out.shape)
    return out


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


# dataloader utils
def get_transforms(input_size: int, transforms_list: list = [], normalize: bool = False, ndims: int = 3, noise: bool = False):
    normalization = []
    if normalize:
        if ndims == 4:
            normalization = [transforms.Normalize((0.5, 0.5, 0.5, 0), (0.5, 0.5, 0.5, 1))]
        else:
            normalization = [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    final_transforms = [transforms.Resize(int(input_size)), transforms.CenterCrop(int(input_size))] + transforms_list + [transforms.ToTensor()]
    if noise:
        final_transforms.append(AddGaussianNoise(0.01, 0.01))
    final_transforms += normalization
    return transforms.Compose(final_transforms)


def denormalize(dataset_name, images):
    if dataset_name in ["MNIST", "FashionMNIST", "Zappos50k", "Objects10_3Dpose"]:  # "TrafficSignsSynth"]:
        return images.to('cpu')
    means,  = torch.tensor((0.5, 0.5, 0.5)).reshape(1, 3, 1, 1)
    std_devs = torch.tensor((0.5, 0.5, 0.5)).reshape(1, 3, 1, 1)
    return images.to('cpu') * std_devs + means


def transformed_images(image, transform_types):
    w = image.shape[-1]
    images, attrs = [], []
    if type(transform_types) == dict:
        images, attrs = [image], [-1]
    for it, transform in enumerate(transform_types):
        ts = None
        if type(transform_types) == dict:
            tinfo = transform_types[transform]
            ts = np.linspace(0, tinfo['range'], tinfo['steps']+1)[1:]

        if transform == "left_shear":
            ses = ts if ts is not None else [random.uniform(1./6, 1./3)]
            for s in ses:  # bigger s means greater shear
                start_pts = [[0, 0], [w, 0], [w, w], [0, w]]
                end_pts = [[s*w, s*w], [w, 0], [w, w], [s*w, (1-s)*w]]
                images.append(transforms.functional.perspective(image, start_pts, end_pts))
                attrs.append(it)
        elif transform == "right_shear":
            ses = ts if ts is not None else [random.uniform(1./6, 1./3)]
            for s in ses:
                start_pts = [[0, 0], [w, 0], [w, w], [0, w]]
                end_pts = [[0, 0], [(1-s)*w, s*w], [(1-s)*w, (1-s)*w], [0, w]]
                images.append(transforms.functional.perspective(image, start_pts, end_pts))
                attrs.append(it)
        elif transform == "top_shear":
            ses = ts if ts is not None else [random.uniform(1./6, 1./3.)]
            for s in ses:
                start_pts = [[0, 0], [w, 0], [w, w], [0, w]]
                end_pts = [[s*w, 0], [(1-s)*w, 0], [w, w], [0, w]]
                images.append(transforms.functional.perspective(image, start_pts, end_pts))
                attrs.append(it)
        elif transform == "bottom_shear":
            ses = ts if ts is not None else [random.uniform(1./6, 1./3.)]
            for s in ses:
                start_pts = [[0, 0], [w, 0], [w, w], [0, w]]
                end_pts = [[0, 0], [w, 0], [(1-s)*w, (1-s)*w], [s*w, (1-s)*w]]
                images.append(transforms.functional.perspective(image, start_pts, end_pts))
                attrs.append(it)
        elif transform == "left_rotate":
            angles = ts if ts is not None else [random.randrange(20, 50)]
            for angle in angles:
                images.append(transforms.functional.rotate(image, angle))
                attrs.append(it)
        elif transform == "right_rotate":
            angles = ts if ts is not None else [random.randrange(20, 50)]
            for angle in angles:
                images.append(transforms.functional.rotate(image, -angle))
                attrs.append(it)
        elif transform == "towards":
            ses = ts if ts is not None else [random.uniform(0.15, 0.35)]
            for s in ses:
                timage = transforms.Resize(size=int((1+s)*w))(image)
                images.append(transforms.CenterCrop(size=w)(timage))
                attrs.append(it)
        elif transform == "far":
            ses = ts if ts is not None else [random.uniform(0.25, 0.55)]
            for s in ses:
                timage = transforms.Resize(size=int((1-s)*w))(image)
                images.append(transforms.CenterCrop(size=w)(timage))
                attrs.append(it)
        else:
            raise NotImplementedError(f'Transform {transform} not supported')
    return images, [attrs]


def add_image(l_img, s_img, x_offset=8, y_offset=8):
    assert s_img.shape[-3] == 4 and l_img.shape[-3] == 4, print(s_img.shape, l_img.shape)  # check that images are RGBA
    y1, y2 = y_offset, y_offset + s_img.shape[-2]
    x1, x2 = x_offset, x_offset + s_img.shape[-1]

    alpha_s = s_img[3, :, :]
    alpha_l = 1.0 - alpha_s
    for c in range(0, 3):
        l_img[c, y1:y2, x1:x2] = (alpha_s * s_img[c, :, :] +
                                  alpha_l * l_img[c, y1:y2, x1:x2])
    return transforms.Resize(size=l_img.shape[-1])(l_img[:3, :, :])


def prepare_training_results_dir(params, dry_run: bool):
    results_root = os.path.join("results" if not dry_run else "/tmp/results", params['dataset']['dataset'])
    results_file = os.path.join(results_root, "all_experiment_results.txt")
    results_dir = os.path.join(results_root, datetime.now().strftime("%Y-%m-%d"), datetime.now().strftime("%H:%M:%S") + '_' + params['desc'])
    results_dirs = {'main': results_dir, 'summaries_file': results_file}
    for subdir in ['models', 'summary']:
        results_dirs[subdir] = os.path.join(results_dir, subdir)
        os.makedirs(results_dirs[subdir], exist_ok=True)
    if params['training']['train_vae']:
        for subdir in ['recons', 'grid_recons', 'embeddings']:
            results_dirs[subdir] = os.path.join(results_dir, subdir)
            os.makedirs(results_dirs[subdir], exist_ok=True)
    return results_dirs


def get_conditional_limits(dataloader):
    attributes, conditional_dims, conditional_loss_types = dataloader.get_dataset_params(["attributes", "conditional_ldims", "conditional_loss_fns"])

    setpt_limits = [[[0, 1.5]] for i in range(conditional_dims)]
    cldi, labels = 0, []
    for ai, attrs_list in enumerate(attributes):
        if conditional_loss_types[ai] in ["KL", "Generator"]:
            ncdims = 1
            label = [f'{attrs_list[0]}-{attrs_list[-1]}']
        else:
            ncdims = len(attrs_list)
            label = attrs_list
        labels.extend(label)

        if conditional_loss_types[ai] == "KL":
            setpts = [get_conditional_mean(j, len(attrs_list)) for j, _ in enumerate(attrs_list)]
            setpt_limits[cldi] = [[4*setpt - 1.5, 4*setpt + 1.5] for setpt in setpts]
        elif conditional_loss_types[ai] == "Generator":
            setpt_limits[cldi] = [[-2, 2]]
        cldi += ncdims
    assert len(labels) == conditional_dims and len(setpt_limits) == conditional_dims, print(labels, setpt_limits, conditional_dims)
    return setpt_limits, labels


def get_conditional_mean(attr_val, nattr_options):
    val = attr_val - int(nattr_options/2)
    mask = val > -1
    neg_mask = val < 0
    value = mask * (val+1) + neg_mask * val
    return 4*value

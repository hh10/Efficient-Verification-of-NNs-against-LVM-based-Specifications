import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split, Dataset, DataLoader

from typing import Dict, Union
import numpy as np
import os
import pandas as pd
from PIL import Image
import itertools
from pandarallel import pandarallel
import pprint
import copy
import glob
from random import shuffle

from utils import get_transforms, denormalize, transformed_images, add_image


class CustomDataset(Dataset):
    def __init__(self, root, attributes_file, class_attributes, attributes_list, input_size, balance, transform=None):
        pandarallel.initialize()
        self.root = root
        self.transform = transform
        self.input_size = input_size
        df = pd.read_table(os.path.join(root, attributes_file), delim_whitespace=True)
        # df = df.drop(df[df['Male'] == -1].index) # HACK for only male attributes invariance checks like beard
        
        # make subdf with only image_id, class and attribute columns (without duplication)
        if type(class_attributes) != list:  # assume all header columns in the file are classes
            class_attributes = df.columns.drop('image_id').values.tolist()
        df_cols = ['image_id'] + class_attributes
        df_cols += list(itertools.chain(*attributes_list))
        df_cols = list(np.unique(np.array(df_cols)))
        df = df[df_cols]
        self.attribute_names = attributes_list

        # remove rows which have multiple attribute options true for an attribute set
        for attrs_list in [*attributes_list, class_attributes]:
            if len(attrs_list) == 1:
                continue
            attrs_sum = df[attrs_list].sum(axis=1)
            df = df[attrs_sum == -(len(attrs_list)-2)]
        self.df = df

        def get_labels(df_, cols):
            if len(cols) == 1:
                return list(np.where(df_[cols[0]] == 1, 0, 1))  # label is 0 is present else 1
            subdf = df_[cols]

            def get_col_index(row):
                b = (subdf.loc[row.name, :] == 1)
                return subdf.columns.get_loc(b.index[b.argmax()])
            return subdf.parallel_apply(get_col_index, axis=1).values.tolist()

        self.images = np.array(df['image_id'].values.tolist())
        self.labels = np.array(get_labels(df, class_attributes))
        attrs_labels, self.attributes_labels = [], [[]]*len(self.images)
        if len(attributes_list) > 0:
            for attrs_list in attributes_list:
                attr_labels = get_labels(df, attrs_list)
                attrs_labels.append(attr_labels)
            self.attributes_labels = np.transpose(np.array(attrs_labels), (1, 0))

        self.class_attributes = class_attributes
        self.classes = copy.deepcopy(class_attributes)
        if len(self.classes) == 1:
            self.classes.append(f'Non{self.classes[0]}')

        # data overview for printing
        self.data_overview, min_cat = self.get_data_overview()
        min_cat = 2 * min_cat
        print("-" * 20)
        print("Dataset summary")
        pprint.pprint(self.data_overview)

        # balance the dataset so the manifold learnt is not biased and variations can be generated from it
        if balance:
            for ci, cla in enumerate(self.classes):
                if len(attributes_list) > 0 and attributes_list[0] != self.class_attributes:
                    for j, attrs_list in enumerate(attributes_list):
                        if attrs_list == self.class_attributes:
                            continue
                        for k in range(np.max((len(attrs_list), 2))):
                            cla_attr_ind = np.where((self.labels == ci) & (self.attributes_labels.T[j] == k))[0]
                            del_ind = cla_attr_ind[min_cat:]
                            self.images = np.delete(self.images, del_ind)
                            self.labels = np.delete(self.labels, del_ind)
                            self.attributes_labels = np.delete(self.attributes_labels, del_ind, axis=0)
                else:
                    del_ind = np.where(self.labels == ci)[0][min_cat:]
                    self.images = np.delete(self.images, del_ind)
                    self.labels = np.delete(self.labels, del_ind)
                    self.attributes_labels = np.delete(np.array(self.attributes_labels), del_ind, axis=0)

            self.data_overview, _ = self.get_data_overview()
            print("-" * 10)
            print("Dataset summary after balancing for classes and attributes")
            pprint.pprint(self.data_overview)
        assert len(self.labels) == len(self.images)
        assert len(self.attributes_labels) == len(self.images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.get_image(self.images[index]), self.labels[index], self.attributes_labels[index]

    def get_data_overview(self):
        data_overview, min_cat = {}, np.inf
        attrs_labels = np.transpose(np.array(self.attributes_labels), (1, 0))
        for ci, cla in enumerate(self.classes):
            cla_mask = np.where(np.array(self.labels) == ci)[0]
            data_overview[cla] = {'Total': len(cla_mask)}
            min_cat = np.min((min_cat, len(cla_mask)))
            if len(self.attribute_names) > 0 and self.attribute_names[0] != self.class_attributes:
                for j, attrs_list in enumerate(self.attribute_names):
                    for k, attr in enumerate(attrs_list):
                        data_overview[cla][attr] = np.sum(np.where(np.array(attrs_labels[j])[cla_mask] == k, 1, 0))
                        if data_overview[cla][attr] > 0:
                            min_cat = np.min((min_cat, data_overview[cla][attr]))
        return data_overview, int(min_cat)  # can be 30*

    def get_image(self, img_name):
        x = Image.open(os.path.join(self.root, img_name))
        if self.transform:
            x = self.transform(x)
        return x

    def transform_tensor_image(self, x, transform_to_apply):
        xi = denormalize("CelebA", x).squeeze(0)
        return get_transforms(self.input_size, [transforms.ToPILImage(), transform_to_apply], normalize=True)(xi)

    def image_has_attribute(self, image_attrs, test_attr) -> bool:
        assert test_attr in [attr for attrs_list in self.attribute_names for attr in attrs_list]
        for j, attrs_list in enumerate(self.attribute_names):
            for k, attr in enumerate(attrs_list):
                if attr == test_attr:
                    return image_attrs[j] == attrs_list.index(test_attr)
        raise AssertionError("Impossible to reach here, what did you do wrong hh")

    def get_same_class_attribute_change_pair(self, x, y, y_attr, test_attr):
        """ Find candidate images similar to x that don't have the attribute but belong to same class for Segment queries """
        if self.image_has_attribute(y_attr, test_attr):
            # test_attr is present in x, so ignore all images that have the test_attr
            subdf = self.df.drop(self.df[self.df[test_attr] == 1].index)
        else:
            subdf = self.df.drop(self.df[self.df[test_attr] == -1].index)

        def class_index_to_label(class_attributes, index):
            if index < len(class_attributes):
                return class_attributes[index], 1
            else:
                return class_attributes[index-1], -1

        class_name, class_val = class_index_to_label(self.class_attributes, y)
        # get subset of all inputs of same class and attribute change
        subdf = subdf.loc[self.df[class_name] == class_val]

        # refine pair search for an image that is closest
        cand_images = []
        for index, row in subdf.iterrows():
            cand_images.append(self.get_image(row['image_id']))
            if index > 10000:
                break
        cand_images = torch.stack(cand_images)
        cand_images_flatten = torch.flatten(cand_images, start_dim=1)
        sorted_diff, _ = torch.sort(torch.abs(cand_images_flatten - torch.flatten(x)), descending=True)
        diff = torch.sum(sorted_diff[:, :100], dim=1)
        diff_ind = torch.argmin(diff, dim=0)
        img_name = subdf['image_id'].iloc[[diff_ind]].item()
        return torch.stack([x, self.get_image(img_name)])


class ImageFolderWithPaths(datasets.ImageFolder):
    """ datasets.ImageFolder derived class that derives attribute from image name and returns it
        along with image, class label in dataloading (file structure assumed is
        root/class/*__(attr_attr_val).png, In codebase, used of Object3D dataset. """
    def __init__(self, root: str, std_transform, transforms_list):
        super().__init__(root, std_transform)
        self.root, self.std_transform = root, std_transform
        self.attr_classes = transforms_list

    def get_attribute_from_path(self, path):
        act_attr = os.path.basename(path).split('.')[0]
        act_attr = ('__').join(act_attr.split('__')[1:])
        return self.get_attribute_from_path_(act_attr)

    def get_attribute_from_path_(self, act_attr, path=None):
        infos = act_attr.split('__')
        attributes = []
        for info in infos:
            atype, aval = info.split('_')
            aval = int(aval)
            for attr, setpts in self.attr_classes.items():
                if atype != attr:
                    continue
                attribute = None
                for si, setpt in enumerate(setpts):
                    if aval == setpt:
                        attribute = [si]
                        break
                assert attribute is not None, print(f"Image path: {path or infos} (des_attr: {atype} {aval})")
                attributes.append(attribute)
                break
        return np.array(attributes).reshape(len(self.attr_classes.keys()),)

    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        attr = self.get_attribute_from_path(path)
        return (original_tuple + (attr,))

    def get_image(self, index, attr):
        img_path = os.path.join(index + '__' + attr + '.png')
        if not os.path.exists(img_path):
            return None
        return self.transform(Image.open(img_path).convert('RGB'))

    def get_image_paths(self):
        dataset_classes = self.find_classes(self.root)
        paths = glob.glob(os.path.join(self.root, "*", "*.png"))
        indices = np.unique(np.array([path.split('__')[0] for path in paths]))
        attrs = np.unique(np.array([('__').join(os.path.basename(path).split('__')[1:]).split('.')[0] for path in paths]))

        data = []
        for index in indices:
            cla = torch.Tensor([dataset_classes[0].index(os.path.basename(os.path.dirname(index)))]).int()
            data.append([index, cla, torch.Tensor([])])
        return data, attrs


class DatasetsWithAnnotatedTransforms():
    """ Class to provide train/test torch datasets with attributes (if conditional on transforms and/or classes").
        In codebase, used for (F)MNIST, Traffic Signs dataset.
    """
    def __init__(self, conditional: Dict[str, Union[list, dict]]):
        self.transform_types = None if "transforms" not in conditional else conditional["transforms"]
        self.conditioning_classes = None if "classes" not in conditional else conditional["classes"]

    def __getitem__(self, dataset, index):
        (images, labels) = dataset.__getitem__(index)  # this returns standard image, label pair
        if not (self.transform_types or self.conditioning_classes):
            return [images.squeeze(0)], labels, torch.Tensor([])
        attrs = []
        if self.transform_types:
            imgs = []
            for ti, ttypes in enumerate(self.transform_types):
                ims, atrs = transformed_images(images, ttypes)  # this assumes a list of lists and list of dicts
                imgs.extend(ims)
                attrs.extend(atrs)
            images = imgs
            labels = [labels]*len(images)
        if self.conditioning_classes:
            attrs.append(labels)
        attributes = np.transpose(np.array(attrs), (1, 0))  # transpose attributes from being per image to BS dominant
        return images, torch.Tensor(labels).long(), torch.Tensor(attributes).long()

    def get_attributes(self):
        attrs = []
        if self.transform_types:
            attrs = [list(ttypes.keys()) if type(ttypes) is dict else ttypes for ttypes in self.transform_types]
        if self.conditioning_classes:
            attrs.append(self.conditioning_classes)
        return attrs


class TorchDatasetWithAnnotatedTransforms(DatasetsWithAnnotatedTransforms):
    def __init__(self, dataset_name, data_folder, input_size, train, conditional, input_transforms=[], noise_transforms=False):
        super().__init__(conditional)
        self.dataset = getattr(datasets, dataset_name)(root=data_folder, train=train, download=True,
                                                       transform=get_transforms(input_size, input_transforms, noise=noise_transforms))

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, index):
        images_list, labels, attrs = super(TorchDatasetWithAnnotatedTransforms, self).__getitem__(self.dataset, index)
        return torch.stack(images_list), labels, attrs


class MergedDatasetsWithAnnotatedTransforms(DatasetsWithAnnotatedTransforms):
    """ Class to load object and background images in different sizes, apply transforms to the former and
        alpha blend them to latter. In codebase, used for Traffic Signs dataset. """
    def __init__(self, ods_path, bds_path, input_shape, transforms, conditional):
        def pil_RGBA_loader(path):
            # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
            with open(path, 'rb') as f:
                return Image.open(f).convert('RGBA')

        self.transforms = get_transforms(input_shape, transforms, normalize=True)
        self.bdataset = datasets.ImageFolder(bds_path, transform=get_transforms(input_shape, normalize=False, ndims=4), loader=pil_RGBA_loader)
        self.odataset = datasets.ImageFolder(ods_path, transform=get_transforms(input_shape * 7/9, normalize=False, ndims=4), loader=pil_RGBA_loader)
        self.classes, _ = self.odataset.find_classes(ods_path)  # exposed
        if "attrs" in conditional and "CLASSES" in conditional["attrs"]:
            conditional["classes"] = self.classes
        super().__init__(conditional)

    def __len__(self):
        return self.odataset.__len__()*self.bdataset.__len__()

    def __getitem__(self, index):
        oindex, bindex = int(index/self.bdataset.__len__()), index % self.bdataset.__len__()
        bimage = self.bdataset.__getitem__(bindex)[0]
        oimages, labels, attrs = super(MergedDatasetsWithAnnotatedTransforms, self).__getitem__(self.odataset, oindex)
        images = []  # transformed_oimages_against_all_bkgnds
        for oimage in oimages:
            image = add_image(bimage.clone(), oimage)  # [3, w, w]
            image = self.transforms(transforms.ToPILImage()(image))
            images.append(image)
        return torch.stack(images), labels, attrs


class DataBatcher():
    """ Class to rearrange dataset batch to multiple sub-batches for Generator/graphics code based training such
        that all sub-batches have the same and only one attribute varying in it """
    def __init__(self, dl, batch_per_attribute, attributes=None, conditional_ldims=None):
        self.dl = dl
        self.batch_per_attribute = batch_per_attribute
        assert not self.batch_per_attribute or (attributes and conditional_ldims)
        self.nattrs_options = [len(attrs_list) for attrs_list in attributes] if attributes else []
        self.conditional_ldims = conditional_ldims

    @staticmethod
    def decode_batch(batch, device):
        assert len(batch) == 3, print(len(batch))
        x = batch[0].to(device) if type(batch[0]) == torch.Tensor else batch[0]
        return x, batch[1].to(device), batch[2].to(device)

    def get_batch(self, device):
        for batch in self.dl:
            if not self.batch_per_attribute:
                yield self.decode_batch(batch, device)
            else:
                aligned_batches = self.annotated_dataset_per_attribute(batch)
                for abatch in aligned_batches:
                    if len(abatch) == 0:
                        continue
                    yield self.decode_batch(abatch, device)

    def annotated_dataset_per_attribute(self, batch):
        """ Divides the batch into sub-batches such that each sub-batch has only one attribute changing """
        batches = []
        imgs, targets, attrs = batch

        def remove_index(li, i):
            return [*li[0:i], *li[i+1:]]

        for ci in range(self.conditional_ldims):
            nattr_options = remove_index(self.nattrs_options, ci)
            breaks = int(np.ceil(np.prod(np.array(self.nattrs_options))/(8*np.prod(np.array(nattr_options)))))
            attr_combs = itertools.product(*[torch.arange(max([no, 2])) for no in nattr_options])
            for comb in list(attr_combs):  # static values for the non-active cond dims
                subbatch_indices = [i for i, attr in enumerate(attrs) if remove_index(attr, ci) == list(comb)]
                if len(subbatch_indices) == 0:
                    continue
                fimgs, ftargets, fattrs = [], [], []
                for ind in subbatch_indices:
                    fimgs.append(imgs[ind].unsqueeze(0))
                    ftargets.append(targets[ind])
                    fattrs.append(attrs[ind])
                sbi, sblength = 0, int(len(fimgs)/breaks)
                for bi in range(breaks):
                    batches.append([torch.cat(fimgs[sbi:sbi+sblength], dim=0),
                                    torch.Tensor(ftargets[sbi:sbi+sblength]).long(),
                                    torch.stack(fattrs[sbi:sbi+sblength])])
                    sbi += sblength
            shuffle(batches)
        return batches

    def __len__(self):
        # todo(hh): non-critical but fix n_sub_batches
        n_sub_batches = 2**self.conditional_ldims * int(np.max(np.array(self.nattrs_options))/8.) if self.batch_per_attribute else 1
        return len(self.dl) * n_sub_batches


# additional dataset for review (not clean)
class FairfacesDataset(Dataset):
    def __init__(self, dataset_path: str, img_size: int, dataset_rows_path=None):
        self.means = (0.5, 0.5, 0.5)  # (0.485, 0.456, 0.406)  # (0.5, 0.5, 0.5)
        self.std_devs = (0.5, 0.5, 0.5)  # (0.229, 0.224, 0.225)  # (0.5, 0.5, 0.5)
        transforms_list = [transforms.Resize(int(img_size)),
                           transforms.CenterCrop(int(img_size)),
                           transforms.ToTensor(),
                           transforms.Normalize(self.means, self.std_devs)]
        self.transform = transforms.Compose(transforms_list)
        self.dataset_path = dataset_path
        self.input_size = img_size
        if dataset_rows_path is None:
            self.dataset_rows = pd.read_csv(f"{dataset_path}/fairface_label_train.csv")
        else:
            self.dataset_rows = pd.read_csv(dataset_rows_path)

        self.gender2label = {"Male": 0,
                             "Female": 1}
        self.race2label = {"Black": 0,
                           "Indian": 1,
                           # "Southeast Asian": 2,
                           # "Latino_Hispanic": 3,
                           # "Middle Eastern": 4,
                           "East Asian": 2}
                           #"White": 6}
        print(f"Loaded dataset with classes: {len(self.gender2label)} and attributes: {len(self.race2label)}")

    def __len__(self):
        return len(self.dataset_rows)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.dataset_path, self.dataset_rows.iloc[idx, 0])
        image = Image.open(img_name)
        gender = self.dataset_rows.iloc[idx, 2]
        assert(gender in self.gender2label), print(f"Unknown gender {gender}")
        gender = int(self.gender2label[gender])
        race = self.dataset_rows.iloc[idx, 3]
        if race not in self.race2label:
            return self.__getitem__((idx+1) % self.__len__())
        race = int(self.race2label[race])
        if self.transform:
            image = self.transform(image)
        return image, gender, torch.Tensor([race]).long()

    def denormalize(self, images: torch.Tensor):
        means, = torch.tensor(self.means).reshape(1, 3, 1, 1)
        std_devs = torch.tensor(self.std_devs).reshape(1, 3, 1, 1)
        return images.to('cpu') * std_devs + means

    def get_classes(self):
        return list(self.gender2label.keys())

    def get_attributes(self):
        return list(self.race2label.keys())

    def transform_tensor_image(self, x, transform_to_apply):
        xi = denormalize("Fairfaces", x).squeeze(0)
        return get_transforms(self.input_size, [transforms.ToPILImage(), transform_to_apply], normalize=True)(xi)


class CustomDataloader():
    
    def __init__(self, params: dict, apply_random_transforms: bool = True, apply_harmful_transforms: bool = False):
        self.dataset_name = params['dataset']
        input_shape = params['input_shape']
        assert input_shape[1] == input_shape[2]
        w = input_shape[2]
        # Directory containing the data.
        data_folder = os.path.join("data", self.dataset_name)
        self.collate_fn = None  # needed when multiple inputs are generated from one input in dataloading
        self.balance_methods = params['data_balance_method']
        conditional, self.conditional_loss_fns = params["conditional"] or {}, params["conditional_loss_fn"] or []
        assert "Generator" not in self.conditional_loss_fns or all([closs == "Generator" for closs in self.conditional_loss_fns])

        transforms_list = [transforms.GaussianBlur(11, sigma=(8, 13)),
                           # transforms.RandomPosterize(bits=4.),
                           transforms.ColorJitter(),
                           transforms.TrivialAugmentWide(),
                           transforms.RandomAdjustSharpness(sharpness_factor=0.1, p=0.99)] if apply_harmful_transforms else []
        dataset, attributes, conditional_ldims, classes = None, None, None, None  # must be set based on dataset in init

        if self.dataset_name in ["MNIST", "FashionMNIST"]:
            classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"] if self.dataset_name == "FashionMNIST" else np.arange(10)
            if "attrs" in conditional and "CLASSES" in conditional["attrs"]:
                conditional["classes"] = classes
            if "transforms" not in conditional and apply_random_transforms:
                transforms_list.extend([transforms.RandomResizedCrop(size=(w, w), scale=(0.8, 1.4)),
                                        transforms.RandomRotation(degrees=15)])

            dataset, self.test_dataset = [TorchDatasetWithAnnotatedTransforms(
                                        self.dataset_name, data_folder, w, train, conditional,
                                        input_transforms=transforms_list, noise_transforms=apply_harmful_transforms) for train in [True, False]]
            if conditional:
                self.collate_fn = self.annotated_dataset_collate_fn
            attribute_names = dataset.get_attributes()

        # CelebA DATASET AND DATA_DESC FILE MUST ALREADY BE DOWNLOADED IN THE self.data_folder DIRECTORY!
        elif self.dataset_name == "CelebA":
            transforms_list.extend([transforms.RandomHorizontalFlip(p=0.5),  # head flips
                                    transforms.RandomAutocontrast(p=0.85),  # useful for finer features such as glasses/mustache
                                    transforms.RandomAdjustSharpness(sharpness_factor=3, p=0.9)])
            dataset = CustomDataset(data_folder, "list_attr_celeba.txt", params["classes"], conditional["attrs"] if "attrs" in conditional else [], w,
                                    balance="reduce" in params['data_balance_method'], transform=get_transforms(w, transforms_list, True))

            classes, attribute_names = dataset.classes, dataset.attribute_names

        elif self.dataset_name == "TrafficSignsDynSynth":
            dataset = MergedDatasetsWithAnnotatedTransforms(os.path.join(data_folder, "signs"), os.path.join(data_folder, "backgrounds"), w, transforms_list, conditional)
            classes = dataset.classes
            attribute_names = dataset.get_attributes()
            self.collate_fn = self.annotated_dataset_collate_fn

        elif self.dataset_name in ["Objects10_3Dpose", "Runways"]:
            transforms_list = [transforms.RandomAutocontrast(p=0.99),
                               transforms.RandomAdjustSharpness(sharpness_factor=5, p=0.99)]
            dataset = ImageFolderWithPaths(data_folder, get_transforms(w, transforms_list, normalize=self.dataset_name in ["Runways"]), conditional)
            classes, _ = dataset.find_classes(data_folder)
            attribute_names = [[f'{k}_{vv}' for vv in v] for (k, v) in conditional.items()]

        elif self.dataset_name == "Fairfaces":
            dataset = FairfacesDataset(data_folder, w)
            classes = dataset.get_classes()
            attribute_names = [dataset.get_attributes()]

        else:
            raise NotImplementedError(f'Dataloader for {self.dataset_name} not implemented')

        assert len(self.conditional_loss_fns) == len(attribute_names), print(self.conditional_loss_fns, attribute_names)
        # compute conditional_ldims here; attribute names are as per the ordering in the configs
        conditional_ldims, attributes = 0, attribute_names
        for (cfunc, attrs_list) in zip(self.conditional_loss_fns, attribute_names):
            conditional_ldims += (1 if cfunc in ["KL", "Generator"] else len(attrs_list))

        reqd_elems = [dataset, classes, attributes, conditional_ldims]
        assert any(elem is not None for elem in reqd_elems), print(reqd_elems)
        self.dataset, self.classes, self.attributes, self.conditional_ldims = reqd_elems
        print("-" * 20)
        print("Classes:", self.classes)
        if len(self.attributes) > 0:
            print(f'Conditioning attributes: {self.attributes}')
            print(f'Conditional losses: {self.conditional_loss_fns} (total conditional_ldims: {self.conditional_ldims})')

    def get_dataset_params(self, params):
        return [getattr(self, param) for param in params]  # risky but convenient

    def decode_batch(self, batch, device):
        x, y, y_attrs = batch[0], batch[1].to(device), None
        if type(x) == torch.Tensor:
            x = x.to(device)
        if len(self.attributes) > 0:
            y_attrs = batch[2] if len(batch) == 3 else batch[1].unsqueeze(1)
            y_attrs = y_attrs.to(device)
        return x, y, y_attrs

    def get_data(self, batch_size):
        class_weights = None
        if self.dataset_name == "CelebA":
            class_sample_count = [self.dataset.data_overview[cla]['Total'] for cla in self.dataset.data_overview]
            class_weights = 1 / torch.Tensor(class_sample_count)

        data_len, test_ratio, val_ratio = len(self.dataset), 0.01, 0.01
        dataset, test_dataset = random_split(
            self.dataset, [int(np.ceil((1-test_ratio)*data_len)), int(np.floor(test_ratio*data_len))]
        )
        data_len = len(dataset)
        train_dataset, val_dataset = random_split(
            dataset, [int(np.ceil((1-val_ratio)*data_len)), int(np.floor(val_ratio*data_len))]
        )

        shuffle, sampler = True, None
        if "increase" in self.balance_methods and class_weights is not None:
            samples_weights = np.array([class_weights[t[1]] for t in train_dataset])
            samples_weights = torch.from_numpy(samples_weights)
            sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weights, batch_size)
            shuffle = False
        # Create dataloader.
        train_dl = DataLoader(
            train_dataset, batch_size=batch_size, pin_memory=False, shuffle=shuffle, sampler=sampler, collate_fn=self.collate_fn,
        )
        val_dl, test_dl = [DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=self.collate_fn) for ds in [val_dataset, test_dataset]]

        print("-" * 20)
        print("Dataset: {}\nBatch Size: %d\nTraining Batches: %d\nValidation Batches: %d\nTest Batches: %d".format(
            self.dataset_name) % (batch_size, len(train_dl), len(val_dl), len(test_dl))
        )
        print("-" * 20)
        train_dl = DataBatcher(train_dl, "Generator" in self.conditional_loss_fns, self.attributes, self.conditional_ldims)
        test_dl = DataBatcher(test_dl, False)
        val_dl = DataBatcher(val_dl, False)
        return train_dl, val_dl, test_dl

    def denormalize(self, images):
        return denormalize(self.dataset_name, images)

    @staticmethod
    def annotated_dataset_collate_fn(batch):
        imgs, targets, attrs = zip(*batch)
        return torch.cat(imgs), torch.cat(targets), torch.cat(attrs)

# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import json

# from torchvision import datasets, transforms
from torchvision import datasets as torch_datasets
from torchvision import transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

###
import torch
from datasets import load_dataset

from smoe_files.smoe_data_utils import get_imagenet_val_subset

def make_preprocess_function(image_transforms):
    def preprocess(example_batch):
        example_batch["pixel_value"] = torch.stack(
            [
                image_transforms(image.convert("RGB"))
                for image in example_batch["image"]
            ]
        )
        return example_batch

    return preprocess

def batch_sampler(examples):
    pixel_value = torch.stack([example["pixel_value"] for example in examples])
    label = torch.tensor([example["label"] for example in examples])
    return pixel_value, label
###

class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.data_set == 'CIFAR':
        dataset = torch_datasets.CIFAR100(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = torch_datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'HF_IMNET':
        # datasets_lib = __import__('datasets', fromlist=['load_dataset'])
        # load_dataset = datasets_lib.load_dataset
        if os.environ["HOSTNAME"] == "gpu30.int.autonlab.org":
            os.environ["HF_DATASETS_CACHE"] = "/home/scratch/youngsec/hf_datasets_cache"
        ###
        if not is_train and hasattr(args, "val_subset") and args.val_subset in [1000, 3000, 5000, 10000]:
            print(f"Loading a subset of validation set with size {args.val_subset}")
            dataset = get_imagenet_val_subset(
                subset_size=args.val_subset,
                batch_size=None,
                num_data_workers=None,
                return_dataset=True,
                dataset_cache=os.environ["HF_DATASETS_CACHE"])
            assert isinstance(dataset, torch.utils.data.Dataset)
        ###
        else:
            print(f"Loading data from cache: {os.environ['HF_DATASETS_CACHE']}")
            dataset = load_dataset(
                "imagenet-1k",
                # split=f"{'train' if is_train else 'validation'}" + "[:1000]",
                split=f"{'train' if is_train else 'validation'}",
                cache_dir=os.environ["HF_DATASETS_CACHE"],
                trust_remote_code=True,
            )
            preprocess_train = make_preprocess_function(transform)
            dataset.set_transform(preprocess_train)
        nb_classes = 1000
    elif args.data_set == 'INAT':
        dataset = INatDataset(args.data_path, train=is_train, year=2018,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'INAT19':
        dataset = INatDataset(args.data_path, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int(args.input_size / args.eval_crop_ratio)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)

import os

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import ViTImageProcessor


HF_CACHED_HOSTS = [
    "gpu30.int.autonlab.org",
    "gpu14.int.autonlab.org",
]


def get_loaders(
    tiny_dataset=True,
    full_dataset = False,
    return_train_loader=False,
    return_clean_train_loader=False,
    return_val_loader=False,
    train_batch_size=None,
    val_batch_size=None,
    train_num_workers=None,
    val_num_workers=None,
    dataset_cache=os.environ["HF_DATASETS_CACHE"],
):

    if dataset_cache is not None:
        assert os.path.exists(dataset_cache)

    # Configure dataset names
    if tiny_dataset:  # tiny-imagenet
        dataset_name = "Maysee/tiny-imagenet"
        train_split_name = "train"
        val_split_name = "valid"
        num_classes = 200
    else:  # imagenet1k
        dataset_name = "imagenet-1k"
        train_split_name = "train"
        val_split_name = "validation"
        num_classes = 1000

    if full_dataset:
        dataset_subset_suffix = ""
    else:
        dataset_subset_suffix = "[:20]"
        train_batch_size = 5
        val_batch_size = 5

    # Image processor
    IMAGE_PROCESSOR_NAME = "google/vit-base-patch16-224-in21k"
    image_processor = ViTImageProcessor.from_pretrained(IMAGE_PROCESSOR_NAME)

    # Helper functions
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
    

    # Dataset and loaders
    trainset, train_loader = None, None
    clean_trainset, clean_train_loader = None, None
    valset, val_loader = None, None

    if return_train_loader:
        trainset = load_dataset(
            dataset_name,
            split=train_split_name + dataset_subset_suffix,
            cache_dir=dataset_cache,
            trust_remote_code=True,
        )
        print(
            f"Loaded {train_split_name} dataset for tiny={tiny_dataset} with {num_classes} classes"
        )
        print(trainset)
        # Transformations for train images
        train_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
                ),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.Normalize(
                    mean=image_processor.image_mean, std=image_processor.image_std
                ),
            ]
        )
        preprocess_train = make_preprocess_function(train_transforms)
        trainset.set_transform(preprocess_train)
        train_loader = DataLoader(
            trainset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=train_num_workers,
            collate_fn=batch_sampler,
        )

    if return_clean_train_loader:
        clean_trainset = load_dataset(
            dataset_name,
            split=train_split_name + dataset_subset_suffix,
            cache_dir=dataset_cache,
            trust_remote_code=True,
        )
        print(
            f"Loaded clean version of {train_split_name} dataset for tiny={tiny_dataset} with {num_classes} classes"
        )
        print(clean_trainset)
        # Transformations for train images
        clean_train_transforms = transforms.Compose(
            [
                transforms.Resize(
                    (image_processor.size["height"], image_processor.size["width"])
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=image_processor.image_mean, std=image_processor.image_std
                ),
            ]
        )
        preprocess_clean_train = make_preprocess_function(clean_train_transforms)
        clean_trainset.set_transform(preprocess_clean_train)
        clean_train_loader = DataLoader(
            clean_trainset,
            batch_size=train_batch_size,
            shuffle=False,
            num_workers=train_num_workers,
            collate_fn=batch_sampler,
        )

    if return_val_loader:
        valset = load_dataset(
            dataset_name,
            split=val_split_name + dataset_subset_suffix,
            cache_dir=dataset_cache,
            trust_remote_code=True,
        )
        print(
            f"Loaded {val_split_name} dataset for tiny={tiny_dataset} with {num_classes} classes"
        )
        print(valset)
        # Transformations for validation images
        val_transforms = transforms.Compose(
            [
                transforms.Resize(
                    (image_processor.size["height"], image_processor.size["width"])
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=image_processor.image_mean, std=image_processor.image_std
                ),
            ]
        )
        preprocess_val = make_preprocess_function(val_transforms)
        valset.set_transform(preprocess_val)
        val_loader = DataLoader(
            valset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=val_num_workers,
            collate_fn=batch_sampler,
        )

    return train_loader, clean_train_loader, val_loader


def get_val_subset_loader(
    subset_size: int, 
    batch_size: int,
    num_data_workers: int,
    dataset_cache=os.environ["HF_DATASETS_CACHE"]
):
    import pickle as pkl
    
    assert os.path.exists(dataset_cache)
    assert os.path.exists(f"{dataset_cache}/imagenet-1k/val_subsets")

    filename = f"{dataset_cache}/imagenet-1k/val_subsets/valset_{subset_size}.pkl"
    with open(filename, 'rb') as f:
        mini_data = pkl.load(f)
    
    # Convert to tensors and create dataloader (modify based on your dataloader creation)
    mini_x, mini_y = zip(*mini_data)
    mini_x = torch.cat(mini_x, axis=0)
    mini_y = torch.tensor(mini_y)

    print(mini_x.shape, mini_y.shape)
    mini_dataset = torch.utils.data.TensorDataset(mini_x, mini_y)

    return torch.utils.data.DataLoader(
        mini_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_data_workers,
    )

    

def get_datasets(
    tiny_dataset = True,
    full_dataset = False,
    return_train = False,
    return_val = True,
    dataset_cache=os.environ["HF_DATASETS_CACHE"],
):
    """ Return just the raw HuggingFace datasets
    """

    if dataset_cache is not None:
        assert os.path.exists(dataset_cache)

    # Configure dataset names
    if tiny_dataset:  # tiny-imagenet
        dataset_name = "Maysee/tiny-imagenet"
        train_split_name = "train"
        val_split_name = "valid"
        num_classes = 200
    else:  # imagenet1k
        dataset_name = "imagenet-1k"
        train_split_name = "train"
        val_split_name = "validation"
        num_classes = 1000

    if full_dataset:
        dataset_subset_suffix = ""
    else:
        dataset_subset_suffix = "[:20]"

    trainset, valset = None, None
    if return_train:
        trainset = load_dataset(
            dataset_name,
            split=train_split_name + dataset_subset_suffix,
            cache_dir=dataset_cache,
            trust_remote_code=True,
        )
        print(
            f"Loaded {train_split_name} dataset for tiny={tiny_dataset} with {num_classes} classes"
        )
        print(trainset)
    if return_val:
        valset = load_dataset(
            dataset_name,
            split=val_split_name + dataset_subset_suffix,
            cache_dir=dataset_cache,
            trust_remote_code=True,
        )
        print(
            f"Loaded {val_split_name} dataset for tiny={tiny_dataset} with {num_classes} classes"
        )
        print(valset)

    return trainset, valset


def match_full_tiny_classes():
    import json
    import pickle as pkl
    os.chdir("/home/scratch/youngsec/rs/moe/git_packages/soft-moe-vit/utils")
    full_classes = json.load(open("imagenet1k_classes.json", 'r'))
    tiny_classes = pkl.load(open("tinyimagenet_classes.pkl", 'rb'))

    full_class_idx_list = []
    tiny_class_idx_list = []
    full_class_name_list = []
    found_tiny_str_list = []
    notfound_tiny_str_list = []
    for tiny_class_idx, tiny_class_str in enumerate(tiny_classes):
        for full_class_idx, full_class_identifiers in full_classes.items():
            full_class_str = full_class_identifiers[0]
            full_class_name =full_class_identifiers[1]
            if tiny_class_str == full_class_str:
                tiny_class_idx_list.append(tiny_class_idx)
                full_class_idx_list.append(full_class_idx)
                full_class_name_list.append(full_class_name)
                found_tiny_str_list.append(tiny_class_str)
                break
        if tiny_class_str not in found_tiny_str_list:
            notfound_tiny_str_list.append(tiny_class_str)

    conversion_dict = {
        "tiny_class_idx_list": tiny_class_idx_list,
        "full_class_idx_list": full_class_idx_list,
        "full_class_name_list": full_class_name_list,
    }
    # pkl.dump(conversion_dict, open("tiny_to_full_class_conversion.pkl", 'wb'))

    

if __name__ == "__main__":
    # # BEGIN: Test get_loaders
    # train_loader, val_loader = get_loaders(
    #     tiny_dataset=False,
    #     return_train_loader=True,
    #     return_val_loader=True,
    #     train_batch_size=256,
    #     val_batch_size=128,
    #     train_num_workers=16,
    #     val_num_workers=16,
    #     dataset_cache=os.environ["HF_DATASETS_CACHE"],
    # )
    # print("Data loaders created")
    # print(f"Train loader: {train_loader}")
    # print(f"Val loader: {val_loader}")
    # print("Done")

    # for loader in [train_loader, val_loader]:
    #     for batch in loader:
    #         for k, v in batch.items():
    #             print(f"{k}: {v.shape}")
    #         break
    # # END: Test get_loaders
    match_full_tiny_classes()


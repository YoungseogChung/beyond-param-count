



import os
import torch
import pickle as pkl

def get_imagenet_val_subset(
    subset_size: int,
    batch_size: int,
    num_data_workers: int,
    return_dataset: bool,
    dataset_cache=os.environ["HF_DATASETS_CACHE"],
):
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
    if return_dataset:
        return mini_dataset

    return torch.utils.data.DataLoader(
        mini_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_data_workers,
    )
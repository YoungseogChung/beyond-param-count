import os
import json
import random
import argparse

import numpy as np
import torch


""" Miscellanous utilities """


def set_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def softmax(x, axis):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


def remove_elements(row, indices):
    """ Remove elements from a row given indices.

    Args:
        row (np.ndarray): flat array
        indices (np.ndarray): flat array of indices to remove

    Returns:
        np.ndarray: row with elements removed, shape (len(row) - len(indices),)
    """
    assert len(row.shape) == 1 and len(indices.shape) == 1
    return np.delete(row, indices)


def cosine_similarity(x: np.ndarray, y: np.ndarray, axis: int):
    assert x.shape == y.shape
    if len(x.shape) == 1:
        # output is a scalar, of shape (,)
        return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    else:
        # output is one dim smaller than original
        return np.sum(x * y, axis=axis) / (np.linalg.norm(x, axis=axis) * np.linalg.norm(y, axis=axis))    
    

def evaluate_classification(model, testloader, device):
    model.eval()
    with torch.no_grad():
        total_correct, total_num = 0.0, 0.0
        # TODO: expects testloader to be a tuple of (ims, labs)
        for ims, labs in testloader:
            ims = ims.to(device)
            ims = ims.reshape(ims.shape[0], -1)
            labs = labs.to(device)
            out = model(ims)
            total_correct += out.argmax(1).eq(labs).sum().cpu().item()
            total_num += ims.shape[0]
        print(f"Accuracy: {total_correct / total_num * 100:.1f}%")

def load_args(path: str):
    with open(path, 'r') as f:
        args = json.load(f)
    if isinstance(args, dict):
        args = argparse.Namespace(**args)
    return args


def print_args(args: argparse.Namespace):
    print(json.dumps(vars(args), indent=4, sort_keys=True))


def change_child_directory(original_path, new_child_directory):
    path_parts = original_path.split(os.sep)
    # Change the last part to "new_child_directory"
    path_parts[-1] = new_child_directory
    new_path = os.sep.join(path_parts)
    return new_path
    
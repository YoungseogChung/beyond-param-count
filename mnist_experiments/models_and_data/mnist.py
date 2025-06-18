"""
Handle MNIST dataset and model construction.
Author: Youngseog Chung
Date: November 18, 2023
"""
import warnings
from functools import partial

import numpy as np
import torch
import torchvision

from models_and_data import SMOE_PATH
from models_and_data.data_utils import (
    patchify,
    patchify_and_flip,
)
from models_and_data.model_utils import (
    construct_mlp_small,
    construct_linear_layer,
)


MNIST_NUM_CLASSES = 10


def get_mnist_dataloader(
    tr_batch_size=128,
    te_batch_size=128,
    tr_shuffle=False,
    te_shuffle=False,
    subsample=False,
    indices=None,
    drop_last=False,
    data_root=f"{SMOE_PATH}/data/mnist",
):
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            # torchvision.transforms.Normalize((0.0,), (1.0,)),
        ]
    )

    trainset = torchvision.datasets.MNIST(
        root=data_root, download=True, train=True, transform=transforms
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=tr_batch_size, shuffle=tr_shuffle, num_workers=0
    )

    testset = torchvision.datasets.MNIST(
        root=data_root, download=True, train=False, transform=transforms
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=te_batch_size, shuffle=te_shuffle, num_workers=0
    )

    # if subsample and split == "train" and indices is None:
    #     dataset = torch.utils.data.Subset(dataset, np.arange(6_000))

    # if indices is not None:
    #     if subsample and split == "train":
    #         print("Overriding `subsample` argument as `indices` was provided.")
    #     dataset = torch.utils.data.Subset(dataset, indices)

    # return torch.utils.data.DataLoader(
    #     dataset=dataset,
    #     shuffle=shuffle,
    #     batch_size=batch_size,
    #     num_workers=0,
    #     drop_last=drop_last,
    # )

    return trainloader, testloader


""" NN model for MNIST """


def construct_mlp(num_inputs=784, num_classes=10):
    # Configurations used in the "influence memorization" paper:
    # https://github.com/google-research/heldout-influence-estimation/blob/master/mnist-example/mnist_infl_mem.py.
    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(num_inputs, 512, bias=False),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 256, bias=False),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 128, bias=False),
        torch.nn.ReLU(),
        torch.nn.Linear(128, num_classes, bias=False),
    )
    return model


# def construct_mlp_small(num_inputs, num_outputs, multiplier=1):
#     model = torch.nn.Sequential(
#         torch.nn.Flatten(),
#         torch.nn.Linear(num_inputs, int(num_outputs * multiplier), bias=False),
#         torch.nn.ReLU(),
#         torch.nn.Linear(int(num_outputs * multiplier), num_outputs, bias=False),
#     )
#     return model


# def construct_linear_layer(num_inputs, num_outputs):
#     model = torch.nn.Sequential(
#         torch.nn.Flatten(),
#         torch.nn.Linear(num_inputs, num_outputs, bias=False),
#     )
#     return model


def construct_mnist_components(args):
    tr_loader, te_loader = get_mnist_dataloader(args.batch_size, args.batch_size)
    loss_fn = torch.nn.CrossEntropyLoss()
    # 0) Patch function for MNIST
    if args.patch_size is None:
        # default patch size
        args.patch_size = 14
    if hasattr(args, "flip_pixels") and args.flip_pixels:
        patch_fn = partial(patchify_and_flip, patch_size=args.patch_size)
    else:
        patch_fn = partial(patchify, patch_size=args.patch_size)
    # 1) Encoder for MNIST
    if args.encoder is None:
        # encoder = lambda x: x.reshape(x.shape[0], -1)
        encoder = lambda x: x
        # args.encoder_out_dim = 784
        # expert_fn = construct_mlp
        args.encoder_out_dim = args.patch_size * args.patch_size
    elif args.encoder == "mlp":
        encoder = construct_mlp_small(
            num_inputs=784, num_outputs=args.encoder_out_dim, multiplier=1
        )
    else:
        raise ValueError(f"Unknown encoder {args.encoder}")
    # 2) Expert function for MNIST
    if args.encoder is None and args.decoder is None:
        # expert_fn = construct_mlp
        expert_fn = partial(construct_mlp, num_inputs=args.encoder_out_dim)
        args.expert_out_dim = MNIST_NUM_CLASSES
    else:
        if args.decoder is None:
            # yes encoder, no decoder
            args.expert_out_dim = MNIST_NUM_CLASSES
        else:
            # - no encoder, yes decoder
            # - yes encoder, yes decoder
            if args.decoder in ["mean"]:
                args.expert_out_dim = MNIST_NUM_CLASSES
            else:
                if args.expert_out_dim is None:
                    warnings.warn(
                        f"expert_out_dim not specified, using encoder_out_dim: {args.encoder_out_dim}"
                    )
                    args.expert_out_dim = args.encoder_out_dim
        expert_fn = partial(
            construct_mlp_small,
            num_inputs=args.encoder_out_dim,
            num_outputs=args.expert_out_dim,
            multiplier=args.expert_fn_multiplier,
        )
    # 3) Decoder for MNIST
    if args.decoder is None:
        decoder = lambda x: x
    elif args.decoder == "mean":
        decoder = lambda x: torch.mean(x, dim=1)
    elif args.decoder == "linear":
        assert args.expert_out_dim is not None, "Must specify expert_out_dim"
        num_tokens = (28 // args.patch_size) ** 2 + args.num_dummy
        decoder = construct_linear_layer(
            num_inputs=num_tokens * args.expert_out_dim, num_outputs=MNIST_NUM_CLASSES
        )
    else:
        raise ValueError(f"Unknown decoder {args.decoder}")
    # 4) Other info for MNIST

    components = {
        "tr_loader": tr_loader,
        "te_loader": te_loader,
        "loss_fn": loss_fn,
        "encoder": encoder,
        "decoder": decoder,
        "expert_fn": expert_fn,
        "patch_fn": patch_fn,
    }

    return components


""" Dummy Token Generation """


def make_dummy_tokens(type_dummy, num_dummy, X):
    """Make dummy tokens for MNIST.

    Args:
        type_dummy (str): type of dummy tokens to make
        num_dummy (int): number of dummy tokens to make
        X (np.ndarray): input data, (num_data, num_tokens, patch_dim)

    Returns:
        callable: dummy token generation function
    """
    num_data, num_tokens, patch_dim = X.shape

    if num_dummy == 0:
        return None

    # Constant dummy tokens
    if type_dummy in ["min", "max"]:
        if type_dummy == "min":
            apply_pixel_value = X.min()
        elif type_dummy == "max":
            apply_pixel_value = X.max()

        def dummy_token_fn(batch_size):
            return apply_pixel_value * torch.ones(batch_size, num_dummy, patch_dim)

    # Marginal random dummy tokens
    elif type_dummy in ["normal", "uniform"]:
        max_v = X.max()
        min_v = X.min()
        if type_dummy == "uniform":
            range_v = max_v - min_v

            def dummy_token_fn(batch_size):
                dummy_token = (
                    range_v * torch.rand(size=(batch_size, num_dummy, patch_dim))
                    + min_v
                )
                return dummy_token

        elif type_dummy == "normal":
            mean_v = X.mean()
            std_v = X.std()

            def dummy_token_fn(batch_size):
                dummy_token = (
                    std_v * torch.randn(size=(batch_size, num_dummy, patch_dim))
                    + mean_v
                )
                dummy_token = torch.clamp(dummy_token, min_v, max_v)
                return dummy_token

    # Pixel-wise random dummy tokens
    elif type_dummy in ["pixel_normal", "pixel_uniform"]:
        assert num_dummy == num_tokens
        max_v = torch.from_numpy(
            np.max(X, axis=0, keepdims=True)
        )  # (1, num_tokens, patch_dim)
        min_v = torch.from_numpy(
            np.min(X, axis=0, keepdims=True)
        )  # (1, num_tokens, patch_dim)
        if type_dummy == "pixel_uniform":
            range_v = max_v - min_v

            def dummy_token_fn(batch_size):
                dummy_token = (
                    range_v * torch.rand(size=(batch_size, num_dummy, patch_dim))
                    + min_v
                )
                return dummy_token

        elif type_dummy == "pixel_normal":
            mean_v = torch.from_numpy(np.mean(X, axis=0, keepdims=True))
            std_v = torch.from_numpy(np.std(X, axis=0, keepdims=True))

            def dummy_token_fn(batch_size):
                dummy_token = (
                    std_v * torch.randn(size=(batch_size, num_dummy, patch_dim))
                    + mean_v
                )
                dummy_token = torch.clamp(dummy_token, min_v, max_v)
                return dummy_token

    else:
        raise ValueError(f"Unknown type_dummy {type_dummy}")

    return dummy_token_fn


if __name__ == "__main__":
    ### BEGIN: short test on norm of each digit
    # import numpy as np
    # import matplotlib.pyplot as plt
    # tr_loader, te_loader = get_mnist_dataloader()
    # x_list = []
    # y_list = []
    # for x, y in te_loader:
    #     x_list.append(x.cpu().numpy())
    #     y_list.append(y.cpu().numpy())
    # x = np.concatenate(x_list)
    # y = np.concatenate(y_list).flatten()
    # digit_norm_list = []
    # for digit in range(10):
    #     digit_idx = y[y == digit]
    #     digit_x = x[y == digit]
    #     num_digits = digit_x.shape[0]
    #     digit_x = digit_x.reshape(num_digits, -1)
    #     digit_norm = np.linalg.norm(digit_x, axis=1)
    #     digit_norm_list.append(np.mean(digit_norm))
    # plt.bar(range(10), digit_norm_list)
    # plt.show()
    ### END: short test on norm of each digit
    pass

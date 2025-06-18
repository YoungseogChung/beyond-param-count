import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


def patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    patch_fn = Rearrange(
        "b c (h p1) (w p2) -> b c (h w) p1 p2", p1=patch_size, p2=patch_size
    )
    organize_fn = Rearrange(
        "b c hw p1 p2 -> b (hw) (c p1 p2)", p1=patch_size, p2=patch_size
    )
    patched_x = organize_fn(patch_fn(x))

    return patched_x


def unpatchify(
    patched_x: torch.Tensor, channels: int, patch_size: int, num_h: int, num_w: int
) -> torch.Tensor:
    """Undo patching operation by patchify

    Args:
        patched_x (torch.Tensor): patched X, of shape (B, num_patches, patch_dim)
        channels (int): number of channels of original image
        patch_size (int): one dimension of a single patch
        num_h (int): number of patches along H dimension
        num_w (int): number of patches along W dimension

    Returns:
        torch.Tensor: unpatched X, of shape (B, C, H, W)
    """
    unorganize_fn = Rearrange(
        "b (hw) (c p1 p2) -> b c hw p1 p2", c=channels, p1=patch_size, p2=patch_size
    )
    unpatch_fn = Rearrange(
        "b c (h w) p1 p2 -> b c (h p1) (w p2)",
        h=num_h,
        w=num_w,
        p1=patch_size,
        p2=patch_size,
    )

    x = unpatch_fn(unorganize_fn(patched_x))

    return x


def patchify_and_flip(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    patch_fn = Rearrange(
        "b c (h p1) (w p2) -> b c (h w) p1 p2", p1=patch_size, p2=patch_size
    )
    organize_fn = Rearrange(
        "b c hw p1 p2 -> b (hw) (c p1 p2)", p1=patch_size, p2=patch_size
    )
    patched_x = organize_fn(patch_fn(x))

    # flip the pixels
    # patched_x -> patched_x_flipped
    # raise NotImplementedError("Need to implement flipping of pixels")
    batch_max = torch.max(
        patched_x.flatten(start_dim=1, end_dim=2), dim=1, keepdim=True
    )[0]
    batch_min = torch.min(
        patched_x.flatten(start_dim=1, end_dim=2), dim=1, keepdim=True
    )[0]
    # assert batch_max.shape == batch_min.shape == (x.shape[0],)

    flipped_patched_x = batch_min[..., None] - patched_x + batch_max[..., None]
    assert flipped_patched_x.shape == patched_x.shape

    return flipped_patched_x


def patchify_synthetic_multiple_regression(
    x: torch.Tensor, patch_size: int
) -> torch.Tensor:
    assert x.shape[1] % patch_size == 0
    patch_fn = Rearrange("b (n p) -> b n p", p=patch_size)  # b, num_patches, patch_size
    patched_x = patch_fn(x)
    return patched_x


def patchify_synthetic_l2_norm(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    return patchify_synthetic_multiple_regression(x, patch_size)


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=8, emb_size=128):
        self.patch_size = patch_size
        super().__init__()
        self.get_patches = Rearrange(
            "b c (h p1) (w p2) -> b c (h w) p1 p2", p1=patch_size, p2=patch_size
        )
        self.rearrange = Rearrange(
            "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_size, p2=patch_size
        )
        self.embedding = nn.Linear(patch_size * patch_size * in_channels, emb_size)
        # self.projection = nn.Sequential(
        #     # break-down the image in s1 x s2 patches and flat them
        #     Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
        #     nn.Linear(patch_size * patch_size * in_channels, emb_size)
        # )

    def patch(self, x: torch.Tensor) -> torch.Tensor:
        # return self.rearrange(x)
        return self.get_patches(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.rearrange(x)
        x = self.embedding(x)
        return x


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from models_and_data.mnist import get_mnist_dataloader
    from models_and_data.cifar10 import get_cifar10_dataloader

    _, mnist_testloader = get_mnist_dataloader(te_batch_size=10)
    _, cifar10_testloader = get_cifar10_dataloader(te_batch_size=10)

    PATCH_SIZE = 4
    mnist_batch = next(iter(mnist_testloader))[0]
    mnist_patcher = PatchEmbedding(in_channels=1, patch_size=PATCH_SIZE)
    mnist_batch_patch = mnist_patcher.get_patches(mnist_batch)
    print(mnist_batch_patch.shape)
    # display stuff
    for i in np.random.choice(mnist_batch.shape[0], 3):
        num_patches = int(np.sqrt(mnist_batch_patch.shape[2]))
        plt.imshow(mnist_batch[i].permute(1, 2, 0))
        plt.show()
        plt.figure(figsize=(num_patches, num_patches))
        for j in range(num_patches * num_patches):
            plt.subplot(num_patches, num_patches, j + 1)
            plt.imshow(mnist_batch_patch[i, :, j].permute(1, 2, 0))
            plt.axis("off")
        plt.show()

    # cifar10_batch = next(iter(cifar10_testloader))[0]
    # cifar10_patcher = PatchEmbedding(in_channels=3, patch_size=PATCH_SIZE)
    # cifar10_batch_patch = cifar10_patcher.get_patches(cifar10_batch)
    # print(cifar10_batch_patch.shape)
    # # display stuff
    # for i in np.random.choice(cifar10_batch.shape[0], 3):
    #     num_patches = int(np.sqrt(cifar10_batch_patch.shape[2]))
    #     plt.imshow(cifar10_batch[i].permute(1, 2, 0))
    #     plt.show()
    #     plt.figure(figsize=(num_patches, num_patches))
    #     for j in range(num_patches * num_patches):
    #         plt.subplot(num_patches, num_patches, j + 1)
    #         plt.imshow(cifar10_batch_patch[i, :, j].permute(1, 2, 0))
    #         plt.axis('off')
    #     plt.show()

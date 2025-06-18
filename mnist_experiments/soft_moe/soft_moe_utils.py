"""
Utilities for soft-MoE
Author: Youngseog Chung
Date: November 18, 2023
"""
import argparse
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
import torch

from utils import (
    softmax,
    remove_elements,
)


""" Phi Construction """


def make_phi(args, device):
    ## Make Phi for soft-moe
    # 1) zeros
    if args.phi_init == 'zeros':
        phi = torch.zeros(
            size=(args.encoder_out_dim, args.num_experts * args.num_slots), 
            requires_grad=True, device=device)
    # 2) ones
    elif args.phi_init == 'ones':
        phi = torch.ones(
            size=(args.encoder_out_dim, args.num_experts * args.num_slots),
            requires_grad=True, device=device)
    # 3) standard normal
    elif args.phi_init == 'standard_normal':
        phi = torch.randn(
            size=(args.encoder_out_dim, args.num_experts * args.num_slots),
            requires_grad=True, device=device)
    # 4) uniform on [0, 1]
    elif args.phi_init == 'positive_uniform':
        phi = torch.rand(
            size=(args.encoder_out_dim, args.num_experts * args.num_slots),
            requires_grad=True, device=device)
    # 5) uniform on [-1, 1]
    elif args.phi_init == 'uniform':
        phi = -1 + 2 * torch.rand(
            size=(args.encoder_out_dim, args.num_experts * args.num_slots),
            device=device)
        phi.requires_grad_()
    # 6) xavier
    elif args.phi_init == 'xavier':
        phi = torch.empty(args.encoder_out_dim, args.num_experts * args.num_slots, 
                requires_grad=True, device=device)
        torch.nn.init.xavier_uniform_(phi)
    # 7) sample each of the n entries iid from N(0, 1) / sqrt(n), then truncate
    # entries whose magnitude exceeds 2 / sqrt(n)
    elif args.phi_init == 'trunc_normal_scaled':
        num_entries = args.encoder_out_dim * args.num_experts * args.num_slots
        phi = torch.randn(
            size=(args.encoder_out_dim, args.num_experts * args.num_slots),
            device=device) / np.sqrt(num_entries)
        
        limit = 2 / np.sqrt(num_entries)
        # truncate entries that are too large
        phi = torch.minimum(
            phi,
            limit * torch.ones(size=(args.encoder_out_dim, args.num_experts * args.num_slots))
        )
        # truncate entries that are too small
        phi = torch.maximum(
            phi,
            -1 * limit * torch.ones(size=(args.encoder_out_dim, args.num_experts * args.num_slots))
        )
        
        phi.requires_grad_()
    else:
        raise ValueError(f"Unknown phi_init {args.phi_init}")

    return phi


""" Reconstruction utilities """


def make_xphi_from_x_phi(x_list: list, phi_list: list):
    """Make XPhi matrix from list of batches of X and Phi

    Args:
        x (list): length num_batches, each element is a numpy array of shape (batch, num_tokens, patch_dim)
        phi (list): length num_batches, each element is a numpy array of shape (patch_dim, num_expert_slots)

    Returns:
        xphi (np.ndarray): (num_batches * batch, num_tokens, num_expert_slots)
    """
    xphi_list = []
    assert len(x_list) == len(phi_list)
    for x, phi in zip(x_list, phi_list):
        assert len(x.shape) == 3
        assert len(phi.shape) == 2
        assert x.shape[2] == phi.shape[0]
        xphi = np.matmul(x, phi)
        xphi_list.append(xphi.copy())
    xphi = np.concatenate(xphi_list, axis=0)
    assert len(xphi.shape) == 3
    return xphi


def make_d_c_from_xphi(xphi: np.ndarray):
    """Make the D and C matrices from the X and Phi matrices

    Args:
        xphi (np.ndarray): (batch, num_tokens, num_expert_slots)
    """
    assert len(xphi.shape) == 3
    D = softmax(xphi, axis=1)
    C = softmax(xphi, axis=2)
    per_token_weight = np.sum(D, axis=2)
    return D, C, per_token_weight


""" Visualization utilities. """

def visualize_patches(patches: np.ndarray, patch_size: int, patch_weight: np.ndarray=None):
    """Display patches in a grid

    Args:
        patches (np.ndarray): (channel, num_patches, patch_width * patch_height)
        patch_size (int): patch_width and height, assuming square patches
        patch_weight (np.ndarray): (num_patches, )
    """
    num_channels, num_patches, _ = patches.shape
    num_patches_per_side = int(np.sqrt(num_patches))
    assert patches.shape[2] == patch_size * patch_size
    
    if patch_weight is None:
        patch_weight = np.ones(shape=(num_patches,))
        patch_weight_normalized = np.ones(shape=(num_patches,))
    else:
        assert patch_weight.shape == (num_patches,)
        # normalizing patch_weight to [0, 1]
        patch_weight_normalized = patch_weight / np.sum(patch_weight)

    plt.figure(figsize=(num_patches_per_side, num_patches_per_side))
    for j in range(num_patches):
        plt.subplot(num_patches_per_side, num_patches_per_side, j + 1)
        cur_patch = patches[:, j, :].reshape(num_channels, patch_size, patch_size)
        display_alpha = np.max([patch_weight_normalized[j], 0.15])
        plt.imshow(cur_patch.transpose(1, 2, 0), alpha=display_alpha)
        plt.text(0.5, 0.5, f"{patch_weight[j]:.3f}", fontsize=10, color='k')
        plt.axis('off')
    plt.show()


""" Model prediction utils """


def get_model_predictions_on_loader(
    model, 
    loader, 
    return_x=False, 
    device = torch.device("cpu")
):
    pred_list = []
    xphi_list = []
    embedding_list = []
    y_tilde_list = []
    y_list = []
    x_list = []
    for batch_data in loader:
        if len(batch_data) == 2:
            X, y = batch_data
            aux_info = None
        elif len(batch_data) == 3:
            X, y, aux_info = batch_data
        X = X.to(device).float()
        y = y.to(device)
        pred, xphi, embedding, y_tilde = model.forward(
            X, return_xphi=True, return_embedding=True, return_y_tilde=True)
        pred_list.append(pred.detach().cpu().numpy().copy())
        xphi_list.append(xphi.detach().cpu().numpy().copy())
        embedding_list.append(embedding.detach().cpu().numpy().copy())
        y_tilde_list.append(y_tilde.detach().cpu().numpy().copy())
        y_list.append(y.detach().cpu().numpy().copy())
        if return_x:
            x_list.append(X.detach().cpu().numpy().copy())

    pred_list = np.concatenate(pred_list, axis=0)
    xphi_list = np.concatenate(xphi_list, axis=0)
    embedding_list = np.concatenate(embedding_list, axis=0)
    y_tilde_list = np.concatenate(y_tilde_list, axis=0)
    y_list = np.concatenate(y_list, axis=0)
    if return_x:
        x_list = np.concatenate(x_list, axis=0)
    out = {"pred": pred_list, "xphi": xphi_list, "embedding": embedding_list, 
           "y_tilde": y_tilde_list, "y": y_list, "x": x_list}
    return out


# using mask to drop experts
def get_model_predictions_on_input_with_mask(
    smoe, 
    x: torch.Tensor,
    mask: np.ndarray,
    return_y_tilde_ranks: bool=False,
    return_y_tilde: bool=False,
):
    """
    Args:
        mask (np.ndarray): the experts you are dropping, not using
    """
    # only one of return_y_tilde_ranks and return_y_tilde can be True
    assert not (return_y_tilde_ranks and return_y_tilde)

    model_call_fn = partial(
        smoe.expert_masked_forward,
        drop_expert_idxs=mask,
        num_heuristic_drop=0,
        return_embedding=True,
        return_xphi=True,
        return_y_tilde_ranks=return_y_tilde_ranks,
        return_y_tilde=return_y_tilde,
    )
    with torch.no_grad():  # TODO: do we need this or not?
        pred, xphi, embedding, aux_info = model_call_fn(x)
        pred = pred.detach().cpu().numpy().copy()
        xphi = xphi.detach().cpu().numpy().copy()
        embedding = embedding.detach().cpu().numpy().copy()
        if return_y_tilde_ranks:
            y_tilde_matrix_rank = np.array(aux_info)
            return pred, xphi, embedding, y_tilde_matrix_rank
        elif return_y_tilde:
            y_tilde = aux_info
            return pred, xphi, embedding, y_tilde
        else:
            return pred, xphi, embedding, aux_info


# using the heuristic
def get_model_predictions_with_heuristic(
    smoe, 
    x: torch.Tensor,
    num_heuristic_drop: int,
    return_y_tilde_ranks: bool=False,
    return_y_tilde: bool=False,
):
    """
    Args:
        num_heuristic_drop (int): the number of experts you are dropping, not using
    """
    # only one of return_y_tilde_ranks and return_y_tilde can be True
    assert not (return_y_tilde_ranks and return_y_tilde)
    model_call_fn = partial(
        smoe.expert_masked_forward,
        num_heuristic_drop=num_heuristic_drop,
        drop_expert_idxs=None,
        return_embedding=True,
        return_xphi=True,
        return_y_tilde_ranks=return_y_tilde_ranks,
        return_y_tilde=return_y_tilde,
    )
    with torch.no_grad():  # TODO: do we need this or not?
        # pred, xphi, embedding, aux_info = model_call_fn(x)
        pred, aux_info = model_call_fn(x)
        pred = pred.detach().cpu().numpy().copy()
        return pred, aux_info
        # xphi = aux_info["X_phi"]
        # embedding = aux_info["X_phi"]
        # if return_y_tilde_ranks:
        #     y_tilde_matrix_rank = np.array(aux_info)
        #     return pred, xphi, embedding, y_tilde_matrix_rank
        # elif return_y_tilde:
        #     y_tilde_aux_info = aux_info
        #     return pred, xphi, embedding, y_tilde_aux_info
        # else:
        #     # aux_info will be None in this case
        #     return pred, xphi, embedding, aux_info


def get_model_predictions_with_heuristic_orig_function(
    smoe, 
    x: torch.Tensor,
    num_heuristic_drop: int,
    return_y_tilde_ranks: bool=False,
    return_y_tilde: bool=False,
):
    """
    Args:
        num_heuristic_drop (int): the number of experts you are dropping, not using
    """
    # only one of return_y_tilde_ranks and return_y_tilde can be True
    assert not (return_y_tilde_ranks and return_y_tilde)
    model_call_fn = partial(
        smoe.orig_expert_masked_forward,
        num_heuristic_drop=num_heuristic_drop,
        drop_expert_idxs=None,
        return_embedding=True,
        return_xphi=True,
        return_y_tilde_ranks=return_y_tilde_ranks,
        return_y_tilde=return_y_tilde,
    )
    with torch.no_grad():  # TODO: do we need this or not?
        pred, xphi, embedding, aux_info = model_call_fn(x)
        pred = pred.detach().cpu().numpy().copy()
        xphi = xphi.detach().cpu().numpy().copy()
        embedding = embedding.detach().cpu().numpy().copy()
        if return_y_tilde_ranks:
            y_tilde_matrix_rank = np.array(aux_info)
            return pred, xphi, embedding, y_tilde_matrix_rank
        elif return_y_tilde:
            y_tilde_aux_info = aux_info
            return pred, xphi, embedding, y_tilde_aux_info
        else:
            # aux_info will be None in this case
            return pred, xphi, embedding, aux_info


# using one a single expert that corresponds to index use_expert_idx
def get_model_predictions_using_single_expert(
    smoe, 
    dataloader, 
    use_expert_idx: int,
    device
):
    with torch.no_grad():  # TODO: do we need this or not?
        pred_list = []
        y_list = []
        for X, y in dataloader:
            X = X.to(device).float()
            y = y.to(device)

            single_mask = torch.tensor([x for x in range(smoe.num_experts) if x != use_expert_idx])
            mask = single_mask.reshape(1, -1).repeat(X.shape[0], 1)
            model_call_fn = partial(
                smoe.expert_masked_forward, 
                drop_expert_idxs=mask,
            )
            pred = model_call_fn(X)
            pred_list.append(pred.detach().cpu().numpy().copy())
            y_list.append(y.detach().cpu().numpy().copy())
        pred_y = np.concatenate(pred_list, axis=0)
        true_y = np.concatenate(y_list, axis=0)

    return pred_y, true_y


""" Computation utilities """


def y_tilde_cosine_similarity(y_tilde: np.ndarray):
    """Compute expert pairwise cosine similarity between y_tilde vectors.

    Args:
        y_tilde (np.ndarray): (batch_size, num_experts, expert_output_dim)
    """
    assert len(y_tilde.shape) == 3
    batch_size, num_experts, expert_output_dim = y_tilde.shape
    prod_term = y_tilde @ y_tilde.transpose(0, 2, 1)  # (batch_size, num_experts, num_experts)
    norm_term = np.linalg.norm(y_tilde, axis=2)[:, :, None] * np.linalg.norm(y_tilde, axis=2)[:, None, :]
    cos_sim = prod_term / norm_term
    return cos_sim


def compute_expert_pairwise_cosine_similiarity(y_tilde, dropped_idx, total_num_experts):
    """ Compute expert pairwise cosine similarity between y_tilde vectors.

    Args:
        y_tilde (np.ndarray): shape (batch_size, num_experts, expert_output_dim)
        dropped_idx (np.ndarray): shape (batch_size, num_experts_dropped)

    Returns:
        np.ndarray: expert pairwise cosine similarity, shape (batch_size, num_experts, num_experts)
    """
    assert len(y_tilde.shape) == 3 and len(dropped_idx.shape) == 2
    assert y_tilde.shape[0] == dropped_idx.shape[0]  # batch_size

    num_dropped = dropped_idx.shape[1]
    expert_keep_idxs = np.stack(
        [remove_elements(np.arange(total_num_experts), x) for x in dropped_idx])
    active_expert_output = y_tilde[  # (batch_size, num_experts - num_dropped, expert_output_dim)
        np.arange(y_tilde.shape[0])[:, None], expert_keep_idxs, :] 
    assert active_expert_output.shape[:2] == (y_tilde.shape[0], total_num_experts - num_dropped)
    expert_pairwise_cos_sim = y_tilde_cosine_similarity(active_expert_output)
    return expert_pairwise_cos_sim


""" Deprecated """


def get_predictions(
    X: torch.Tensor,
    batch_size: int,
    args: argparse.Namespace,
    patch_fn: callable,
    encoder: callable,
    decoder: callable,
    phi: torch.Tensor, 
    expert_list: list, 
    device_name: str='cpu',
    baseline_model: torch.nn.Module=None,
):
    """
    Get per-token weight for soft-MoE

    Args:
        X: (batch, channel, height, width)

        baseline_model: torch.nn.Module
    """
    # Handle device
    device = torch.device(device_name)

    # Make X, phi tensors if numpy arrays
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)
    if isinstance(phi, np.ndarray):
        phi = torch.from_numpy(phi)

    X = X.to(device).float()
    phi = phi.to(device).float()
    assert X.shape[0] == batch_size

    with torch.no_grad():
        X = patch_fn(X)
        X = encoder(X)
        # check shape of X: (batch, num_tokens, patch_dim)
        assert len(X.shape) == 3
        assert X.shape[0] == batch_size
        num_tokens, patch_dim = X.shape[1:]
        num_expert_slots = args.num_experts * args.num_slots
        
        if args.method == 'baseline':
            Y = baseline_model(X)
        elif args.method == 'soft':
            X_phi = torch.matmul(X, phi)  # (b chw pp) x (pp np) -> (b, chw, np)
            assert X_phi.shape == (batch_size, num_tokens, num_expert_slots)
            D = torch.softmax(X_phi, dim=1)  # softmax across num_tokens dimension
            assert D.shape == X_phi.shape

            X_tilde = torch.matmul(D.transpose(1, 2), X)
            if args.routing_type == "identity":
                assert X.shape[1] == phi.shape[2]
                X_tilde = X
            # (B, 1 * 2 * 2, args.num_experts * args.num_slots) 
            # x (B, 1 * 2 * 2, pp) 
            # -> (B, args.num_experts * args.num_slots, pp)
            assert X_tilde.shape == (batch_size, num_expert_slots, patch_dim)

            Y_tilde_list = []
            for idx in range(args.num_experts):
                # Stack args.num_slots number of rows of X_tilde to get a batch
                cur_input_slice = slice(idx * args.num_slots, (idx + 1) * args.num_slots)
                cur_X_tilde = X_tilde[:, cur_input_slice]  # (B, args.num_slots, patch_dim)
                # Flatten to make it a batch of size B * args.num_slots
                cur_X_tilde = cur_X_tilde.reshape(batch_size * args.num_slots, patch_dim)
                cur_out = expert_list[idx].forward(cur_X_tilde)
                cur_out = cur_out.reshape(batch_size, args.num_slots, -1)
                Y_tilde_list.append(cur_out)
            Y_tilde = torch.cat(Y_tilde_list, dim=1)
            assert (
                (len(Y_tilde.shape) == 3) 
                and (Y_tilde.shape[0] == batch_size) 
                and (Y_tilde.shape[1] == args.num_experts * args.num_slots)
            )  # last dim of Y_tilde is expert_pred_dim
            C = torch.softmax(X_phi, dim=2)
            # ### BEGIN: hack
            # C = torch.softmax(D, dim=2)
            # assert C.shape == X_phi.shape
            # ### END: hack
            Y = torch.matmul(C, Y_tilde)
        Y = decoder(Y)
    info = {
        "patched_X": X.detach().cpu().numpy(),
        "X_tilde": X_tilde.detach().cpu().numpy(),
        "Y_tilde": Y_tilde.detach().cpu().numpy(),
        "X_phi": X_phi.detach().cpu().numpy(),
        "D": D.detach().cpu().numpy(),
        "C": C.detach().cpu().numpy(),
    }

    return Y, info
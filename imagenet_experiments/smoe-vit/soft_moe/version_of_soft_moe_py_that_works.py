import os
from typing import Callable
import numpy as np
import copy
from timeit import default_timer as timer

import torch
import torch.nn as nn
import torch.nn.functional as F


_SAVE_EMBEDDINGS = False
_EMBEDDING_SAVE_DIR = "/zfsauton/project/public/ysc/smoe-vit/deit_embeddings/stitch_ne64"
def find_save_number(save_dir: str, prefix: str, file_extension: str) -> int:
    file_suffix_num = 0
    while True:
        if not os.path.exists(f"{save_dir}/{prefix}_{file_suffix_num}.{file_extension}"):
            break
        file_suffix_num += 1
    return file_suffix_num


def softmax(x: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
    """
    Compute the softmax along the specified dimensions.
    This function adds the option to specify multiple dimensions

    Args:
        x (torch.Tensor): Input tensor.
        dims (int or tuple[int]): The dimension or list of dimensions along which the softmax probabilities are computed.

    Returns:
        torch.Tensor: Output tensor containing softmax probabilities along the specified dimensions.
    """
    max_vals = torch.amax(x, dim=dim, keepdim=True)
    e_x = torch.exp(x - max_vals)
    sum_exp = e_x.sum(dim=dim, keepdim=True)
    return e_x / sum_exp


class SoftMoELayerWrapper(nn.Module):
    """
    A wrapper class to create a Soft Mixture of Experts layer.

    From "From Sparse to Soft Mixtures of Experts"
    https://arxiv.org/pdf/2308.00951.pdf
    """

    def __init__(
        self,
        dim: int,
        num_experts: int,
        slots_per_expert: int,
        layer: Callable,
        normalize: bool = True,
        **layer_kwargs,
    ) -> None:
        """
        Args:
            dim (int): Dimensionality of input features.
            num_experts (int): Number of experts.
            slots_per_expert (int): Number of token slots per expert.
            layer (Callable): Network layer of the experts.
            normalize (bool): Normalize input and phi (sec. 2.3 from paper)
            **layer_kwargs: Additional keyword arguments for the layer class.
        """
        super().__init__()

        self.dim = dim
        self.num_experts = num_experts
        self.slots_per_expert = slots_per_expert
        self.normalize = normalize

        # Initialize phi and normalization scaling factor
        self.phi = nn.Parameter(torch.zeros(dim, num_experts, slots_per_expert))
        if self.normalize:
            self.scale = nn.Parameter(torch.ones(1))

        # Initialize phi using LeCun normal initialization
        # https://github.com/google-research/vmoe/blob/662341d007650d5bbb7c6a2bef7f3c759a20cc7e/vmoe/projects/soft_moe/router.py#L49C1-L49C1
        nn.init.normal_(self.phi, mean=0, std=1 / dim**0.5)

        # Create a list of expert networks
        self.experts = nn.ModuleList(
            [layer(**layer_kwargs) for _ in range(num_experts)]
        )

        self.forward_function = self.forward_original
        self.drop_expert_idxs = None
        self.num_heuristic_drop = None
        self.forward_with_drop = None

        ###
        self.c_cache = []
        self.last_y_tilde = None
        ###

        ### BEGIN: VMAP attributes
        self._expert_function = None  # this is for vmap
        self.stacked_expert_params = None  # this is for vmap
        self.stacked_expert_buffers = None  # this is for vmap
        ### END: VMAP attributes

    def _call_single_expert(self, params, buffers, input_tensor):
        return torch.func.functional_call(self._expert_function, (params, buffers), (input_tensor,))
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.forward_function(*args, **kwargs)
        
    def forward_original(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Soft-MoE layer (algorithm 1 from paper).

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, input_dim].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, input_dim].
        """
        # assert False, "forward_original should not be called"
        
        assert (
            x.shape[-1] == self.dim
        ), f"Input feature dim of {x.shape[-1]} does not match layer dim of {self.dim}"
        assert (
            len(x.shape) == 3
        ), f"Input expected to have 3 dimensions but has {len(x.shape)}"

        phi = self.phi

        # Normalize input and phi
        if self.normalize:
            x = F.normalize(x, dim=2)  # [b, m, d]
            phi = self.scale * F.normalize(phi, dim=0)  # [d, n, p]

        # Compute dispatch and combine weights
        logits = torch.einsum("bmd,dnp->bmnp", x, phi)
        d = softmax(logits, dim=1)
        c = softmax(logits, dim=(2, 3))
        # self.c_cache.append(c.detach().cpu().numpy())

        # Compute input slots as weighted average of input tokens using dispatch weights
        xs = torch.einsum("bmd,bmnp->bnpd", x, d)

        # Apply expert to corresponding slots
        ys = torch.stack(
            [f_i(xs[:, i, :, :]) for i, f_i in enumerate(self.experts)], dim=1
        )

        ### BEGIN: Caching Y_tilde
        self.last_y_tilde = ys.detach().cpu().numpy().copy()
        ### END: Caching Y_tilde

        # Compute output tokens as weighted average of output slots using combine weights
        y = torch.einsum("bnpd,bmnp->bmd", ys, c)

        if _SAVE_EMBEDDINGS:
            ys_arr = ys.detach().cpu().numpy()
            # y_arr = y.detach().cpu().numpy()
            ys_save_prefix = "orig_ys"
            # y_save_prefix = "orig_y"
            ys_num = find_save_number(_EMBEDDING_SAVE_DIR, ys_save_prefix, "npy")
            # y_num = find_save_number(_EMBEDDING_SAVE_DIR, ys_save_prefix, "npy")
            np.save(f"{_EMBEDDING_SAVE_DIR}/{ys_save_prefix}_{ys_num}.npy", ys_arr)
            # np.save(f"{_EMBEDDING_SAVE_DIR}/{y_save_prefix}_{y_num}.npy", y_arr)

        return y

    def _make_expert_base_function(self):
        self._expert_function = copy.deepcopy(self.experts[0]).to('meta')
        # Paramters in the base function are not used
        for param in self._expert_function.parameters():
            param.requires_grad = False
        # Paramters in self.experts are no longer used
        for param in self.experts.parameters():
            param.requires_grad = False
        
    def _prepare_vmap_parameters(self):
        # Stack the parameters and buffers of the experts for vmap
        # This should be called only once at the beginning
        # assert self.stacked_expert_params is None
        # assert self.stacked_expert_buffers is None
        self.stacked_expert_params, self.stacked_expert_buffers = torch.func.stack_module_state(self.experts)
        assert len(self.stacked_expert_buffers) == 0
        return self.stacked_expert_params

    def orig_forward_vmap(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Soft-MoE layer (algorithm 1 from paper).

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, input_dim].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, input_dim].
        """
        assert (
            x.shape[-1] == self.dim
        ), f"Input feature dim of {x.shape[-1]} does not match layer dim of {self.dim}"
        assert (
            len(x.shape) == 3
        ), f"Input expected to have 3 dimensions but has {len(x.shape)}"

        phi = self.phi

        # Normalize input and phi
        if self.normalize:
            x = F.normalize(x, dim=2)  # [b, m, d]
            phi = self.scale * F.normalize(phi, dim=0)  # [d, n, p]

        # Compute dispatch and combine weights
        logits = torch.einsum("bmd,dnp->bmnp", x, phi)
        d = softmax(logits, dim=1)
        c = softmax(logits, dim=(2, 3))
        # self.c_cache.append(c.detach().cpu().numpy())

        # Compute input slots as weighted average of input tokens using dispatch weights
        xs = torch.einsum("bmd,bmnp->bnpd", x, d)

        # Forward with vmap
        if self._expert_function is None:
            self._expert_function = copy.deepcopy(self.experts[0]).to('meta')
            # layer(**layer_kwargs).to("meta")
        self._expert_params, self._expert_buffers = torch.func.stack_module_state(self.experts)
        ys = torch.vmap(self._call_single_expert, in_dims=(0, 0, 1), out_dims=1)(
            self._expert_params, self._expert_buffers, xs)


        # Compute output tokens as weighted average of output slots using combine weights
        y = torch.einsum("bnpd,bmnp->bmd", ys, c)

        # ### BEGIN: Caching Y_tildes
        self.last_y_tilde = ys.detach().cpu().numpy().copy()
        self.last_c = c.detach().cpu().numpy().copy()
        self.last_d = d.detach().cpu().numpy().copy()
        self.last_y = y.detach().cpu().numpy().copy()
        # ### END: Caching Y_tilde
        
        if _SAVE_EMBEDDINGS:
            ys_arr = ys.detach().cpu().numpy()
            # y_arr = y.detach().cpu().numpy()
            ys_save_prefix = "orig_ys"
            # y_save_prefix = "orig_y"
            ys_num = find_save_number(_EMBEDDING_SAVE_DIR, ys_save_prefix, "npy")
            # y_num = find_save_number(_EMBEDDING_SAVE_DIR, ys_save_prefix, "npy")
            np.save(f"{_EMBEDDING_SAVE_DIR}/{ys_save_prefix}_{ys_num}.npy", ys_arr)
            # np.save(f"{_EMBEDDING_SAVE_DIR}/{y_save_prefix}_{y_num}.npy", y_arr)

        return y

    def forward_vmap(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Soft-MoE layer (algorithm 1 from paper).

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, input_dim].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, input_dim].
        """
        assert (
            x.shape[-1] == self.dim
        ), f"Input feature dim of {x.shape[-1]} does not match layer dim of {self.dim}"
        assert (
            len(x.shape) == 3
        ), f"Input expected to have 3 dimensions but has {len(x.shape)}"

        phi = self.phi

        # Normalize input and phi
        if self.normalize:
            x = F.normalize(x, dim=2)  # [b, m, d]
            phi = self.scale * F.normalize(phi, dim=0)  # [d, n, p]

        # Compute dispatch and combine weights
        logits = torch.einsum("bmd,dnp->bmnp", x, phi)
        d = softmax(logits, dim=1)
        c = softmax(logits, dim=(2, 3))
        # self.c_cache.append(c.detach().cpu().numpy())

        # Compute input slots as weighted average of input tokens using dispatch weights
        xs = torch.einsum("bmd,bmnp->bnpd", x, d)

        # Forward with vmap
        if self._expert_function is None:
            self._make_expert_base_function()
            # self._expert_function = copy.deepcopy(self.experts[0]).to('meta')
        #     # layer(**layer_kwargs).to("meta")
        # self._expert_params, self._expert_buffers = torch.func.stack_module_state(self.experts)
        ys = torch.vmap(self._call_single_expert, in_dims=(0, 0, 1), out_dims=1)(
            self.stacked_expert_params, self.stacked_expert_buffers, xs)

        # Compute output tokens as weighted average of output slots using combine weights
        y = torch.einsum("bnpd,bmnp->bmd", ys, c)

        # # ### BEGIN: Caching Y_tildes
        # self.last_y_tilde = ys.detach().cpu().numpy().copy()
        # self.last_c = c.detach().cpu().numpy().copy()
        # self.last_d = d.detach().cpu().numpy().copy()
        # self.last_y = y.detach().cpu().numpy().copy()
        # # ### END: Caching Y_tilde
        
        if _SAVE_EMBEDDINGS:
            ys_arr = ys.detach().cpu().numpy()
            # y_arr = y.detach().cpu().numpy()
            ys_save_prefix = "orig_ys"
            # y_save_prefix = "orig_y"
            ys_num = find_save_number(_EMBEDDING_SAVE_DIR, ys_save_prefix, "npy")
            # y_num = find_save_number(_EMBEDDING_SAVE_DIR, ys_save_prefix, "npy")
            np.save(f"{_EMBEDDING_SAVE_DIR}/{ys_save_prefix}_{ys_num}.npy", ys_arr)
            # np.save(f"{_EMBEDDING_SAVE_DIR}/{y_save_prefix}_{y_num}.npy", y_arr)

        return y


    # def forward_with_drop(self, x: torch.Tensor) -> torch.Tensor:
    def forward_with_drop_no_renorm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward with expert drop.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, input_dim].
            drop_expert_idxs (torch.Tensor): Indexes of experts to drop of shape 
                [batch_size, num_dropped_experts]. Values are not indicators, 
                but actual indices of experts in [0, num_experts - 1].
            num_heuristic_drop (int): Specifies the number of experts to drop, 
                which are computed by the expert columns in C which have lowest 
                total weight. Must be 0 if drop_expert_idxs is not None.
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, input_dim].
        """
        assert (
            x.shape[-1] == self.dim
        ), f"Input feature dim of {x.shape[-1]} does not match layer dim of {self.dim}"
        assert (
            len(x.shape) == 3
        ), f"Input expected to have 3 dimensions but has {len(x.shape)}"

        phi = self.phi

        # Normalize input and phi
        if self.normalize:
            x = F.normalize(x, dim=2)  # [b, m, d]
            phi = self.scale * F.normalize(phi, dim=0)  # [d, n, p]

        # Compute dispatch and combine weights
        logits = torch.einsum("bmd,dnp->bmnp", x, phi)
        c = softmax(logits, dim=(2, 3))

        ### BEGIN: heuristic
        batch_size = x.shape[0]
        if self.num_heuristic_drop is not None and self.num_heuristic_drop > 0:
            with torch.no_grad():
                assert self.drop_expert_idxs is None
                # sum over num_tokens
                c_sum_per_expert = torch.sum(c, dim=(1, 3))
                assert np.shape(c_sum_per_expert) == (batch_size, self.num_experts)
                # find the expert columns with the least total weight
                least_weight_expert_idxs = torch.argsort(c_sum_per_expert, dim=1)
                # least_weight = np.sort(c_sum_per_expert, axis=1)
                # least_weight = least_weight[:, :num_heuristic_drop]
                # print(np.sum(least_weight) / (batch_size * num_tokens))
                drop_expert_idxs = least_weight_expert_idxs[:, :self.num_heuristic_drop]
                # assert np.all(np.diff(drop_expert_idxs) > 0)
                # print(len(np.unique(drop_expert_idxs.astype(int), axis=0)))
                # import math
                # print(math.comb(num_expert_slots, num_heuristic_drop))
                assert drop_expert_idxs.shape == (batch_size, self.num_heuristic_drop)
                # drop_expert_idxs = torch.Tensor(drop_expert_idxs).to(int)
        else:
            assert self.drop_expert_idxs is not None
            drop_expert_idxs = self.drop_expert_idxs
        ### END: heuristic

        # (batch_size, num_tokens, num_experts, num_expert_slots)
        ## BEGIN BLOCK: for re-normalizing C
        # mask = torch.ones_like(logits)
        # mask[np.arange(batch_size)[:, None], :, drop_expert_idxs, :] = 0
        # logits = logits.masked_fill(mask == 0, -torch.inf)
        ## END BLOCK: for re-normalizing C
        
        d = softmax(logits, dim=1)  # (b, m, n, p)
        # (batch_size, num_tokens, num_experts, num_expert_slots)

        ## BEGIN BLOCK: for re-normalizing C
        # d = d.masked_fill(~torch.isfinite(d), 0)
        ## END BLOCK: for re-normalizing C
        assert d.shape == logits.shape

        ### BEGIN HACK: removing for now
        # self.c_cache.append(c.detach().cpu().numpy())
        ### END HACK: removing for now

        # Compute input slots as weighted average of input tokens using dispatch weights
        xs = torch.einsum("bmd,bmnp->bnpd", x, d)

        # Apply expert to corresponding slots
        # ys = torch.stack(
        #     [f_i(xs[:, i, :, :]) for i, f_i in enumerate(self.experts)], dim=1
        # )

        ####
        ys = torch.zeros_like(xs, device=xs.device)  # (batch, ne, slots (always 1), token_dim (assuming token dim in/out is same))
        for i, f_i in enumerate(self.experts):
            datapoint_mask = (~(drop_expert_idxs == i)).all(dim=1)
            ys[datapoint_mask, i] = f_i(xs[datapoint_mask, i, :, :])
            # have to compare ys[:, i] and ys_new[:, i]    
        ####

        
        # Compute output tokens as weighted average of output slots using combine weights
        y = torch.einsum("bnpd,bmnp->bmd", ys, c)

        ### BEGIN: Caching Y_tilde
        self.last_y_tilde = ys.detach().cpu().numpy().copy()
        self.last_c = c.detach().cpu().numpy().copy()
        self.last_d = d.detach().cpu().numpy().copy()
        self.last_y = y.detach().cpu().numpy().copy()
        self.last_drop_expert_idxs = drop_expert_idxs.detach().cpu().numpy().copy()
        ### END: Caching Y_tilde
        
        if _SAVE_EMBEDDINGS:
            ys_arr = ys.detach().cpu().numpy().copy()
            ys_arr[np.arange(ys_arr.shape[0])[:, None], drop_expert_idxs.detach().cpu().numpy()] = 0
            # y_arr = y.detach().cpu().numpy()
            ys_save_prefix = f"drop_{self.num_heuristic_drop}_ys"
            # y_save_prefix = f"drop_{self.num_heuristic_drop}_y"
            ys_num = find_save_number(_EMBEDDING_SAVE_DIR, ys_save_prefix, "npy")
            # y_num = find_save_number(_EMBEDDING_SAVE_DIR, ys_save_prefix, "npy")
            np.save(f"{_EMBEDDING_SAVE_DIR}/{ys_save_prefix}_{ys_num}.npy", ys_arr)
            # np.save(f"{_EMBEDDING_SAVE_DIR}/{y_save_prefix}_{y_num}.npy", y_arr)

        return y

    def forward_with_drop_with_renorm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward with expert drop.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, input_dim].
            drop_expert_idxs (torch.Tensor): Indexes of experts to drop of shape 
                [batch_size, num_dropped_experts]. Values are not indicators, 
                but actual indices of experts in [0, num_experts - 1].
            num_heuristic_drop (int): Specifies the number of experts to drop, 
                which are computed by the expert columns in C which have lowest 
                total weight. Must be 0 if drop_expert_idxs is not None.
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, input_dim].
        """
        assert (
            x.shape[-1] == self.dim
        ), f"Input feature dim of {x.shape[-1]} does not match layer dim of {self.dim}"
        assert (
            len(x.shape) == 3
        ), f"Input expected to have 3 dimensions but has {len(x.shape)}"

        phi = self.phi

        # Normalize input and phi
        if self.normalize:
            x = F.normalize(x, dim=2)  # [b, m, d]
            phi = self.scale * F.normalize(phi, dim=0)  # [d, n, p]

        # Compute dispatch and combine weights
        logits = torch.einsum("bmd,dnp->bmnp", x, phi)
        c = softmax(logits, dim=(2, 3))

        ### BEGIN: heuristic
        batch_size = x.shape[0]
        if self.num_heuristic_drop is not None and self.num_heuristic_drop > 0:
            with torch.no_grad():
                assert self.drop_expert_idxs is None
                # sum over num_tokens
                c_sum_per_expert = torch.sum(c, dim=(1, 3))
                assert np.shape(c_sum_per_expert) == (batch_size, self.num_experts)
                # find the expert columns with the least total weight
                least_weight_expert_idxs = torch.argsort(c_sum_per_expert, dim=1)
                drop_expert_idxs = least_weight_expert_idxs[:, :self.num_heuristic_drop]
                assert drop_expert_idxs.shape == (batch_size, self.num_heuristic_drop)
        else:
            assert self.drop_expert_idxs is not None
            drop_expert_idxs = self.drop_expert_idxs
        ### END: heuristic

        # (batch_size, num_tokens, num_experts, num_expert_slots)
        ## BEGIN BLOCK: for re-normalizing C
        mask = torch.ones_like(logits)
        mask[np.arange(batch_size)[:, None], :, drop_expert_idxs, :] = 0
        logits = logits.masked_fill(mask == 0, -torch.inf)
        ## END BLOCK: for re-normalizing C
        
        d = softmax(logits, dim=1)
        c = softmax(logits, dim=(2, 3))  # (b, m, n, p)
        # compare c and redo_c
        # (batch_size, num_tokens, num_experts, num_expert_slots)

        ## BEGIN BLOCK: for re-normalizing C
        d = d.masked_fill(~torch.isfinite(d), 0)
        ## END BLOCK: for re-normalizing C
        assert d.shape == logits.shape

        ### BEGIN HACK: removing for now
        # self.c_cache.append(c.detach().cpu().numpy())
        ### END HACK: removing for now

        # Compute input slots as weighted average of input tokens using dispatch weights
        xs = torch.einsum("bmd,bmnp->bnpd", x, d)

        # Apply expert to corresponding slots
        # orig_ys = torch.stack(
        #     [f_i(xs[:, i, :, :]) for i, f_i in enumerate(self.experts)], dim=1
        # )

        ####
        ys = torch.zeros_like(xs, device=xs.device)  # (batch, ne, slots (always 1), token_dim (assuming token dim in/out is same))
        for i, f_i in enumerate(self.experts):
            datapoint_mask = (~(drop_expert_idxs == i)).all(dim=1)
            ys[datapoint_mask, i] = f_i(xs[datapoint_mask, i, :, :])
            # have to compare ys[:, i] and ys_new[:, i]    
        ####

        
        # Compute output tokens as weighted average of output slots using combine weights
        y = torch.einsum("bnpd,bmnp->bmd", ys, c)

        ### BEGIN: Caching Y_tilde
        self.last_y_tilde = ys.detach().cpu().numpy().copy()
        self.last_c = c.detach().cpu().numpy().copy()
        self.last_d = d.detach().cpu().numpy().copy()
        self.last_y = y.detach().cpu().numpy().copy()
        self.last_drop_expert_idxs = drop_expert_idxs.detach().cpu().numpy().copy()
        ### END: Caching Y_tilde

        if _SAVE_EMBEDDINGS:
            ys_arr = ys.detach().cpu().numpy().copy()
            ys_arr[np.arange(ys_arr.shape[0])[:, None], drop_expert_idxs.detach().cpu().numpy()] = 0
            # y_arr = y.detach().cpu().numpy()
            ys_save_prefix = f"drop_{self.num_heuristic_drop}_ys"
            # y_save_prefix = f"drop_{self.num_heuristic_drop}_y"
            ys_num = find_save_number(_EMBEDDING_SAVE_DIR, ys_save_prefix, "npy")
            # y_num = find_save_number(_EMBEDDING_SAVE_DIR, ys_save_prefix, "npy")
            np.save(f"{_EMBEDDING_SAVE_DIR}/{ys_save_prefix}_{ys_num}.npy", ys_arr)
            # np.save(f"{_EMBEDDING_SAVE_DIR}/{y_save_prefix}_{y_num}.npy", y_arr)

        return y


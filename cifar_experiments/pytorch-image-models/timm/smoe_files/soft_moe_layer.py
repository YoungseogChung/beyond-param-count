import os
from typing import Callable, Union
import numpy as np
import copy
from timeit import default_timer as timer

import torch
import torch.nn as nn
import torch.nn.functional as F
import functorch


""" Constants """
_SAVE_EMBEDDINGS = False


""" Utils """
def softmax(x: torch.Tensor, dim: Union[int, tuple[int, ...]]) -> torch.Tensor:
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


""" SMoE """
class SoftMoELayerWrapper(nn.Module):

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

        ### BEGIN: make sure certain params are a certain value
        assert self.slots_per_expert == 1
        assert self.normalize == True
        ### END: make sure certain params are a certain value

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

        ### BEGIN: additional attributes for our project
        # Setting forward function
        self.forward_tokenized = self.forward_no_drop
        # Dropping experiments
        self.drop_expert_idxs = None
        self.num_heuristic_drop = None
        self.drop_type = None
        ### END: additional attributes for our project


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Soft-MoE layer (algorithm 1 from paper).

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, channels, height, width].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, input_dim].
        """
        ### BEGIN: tokenization step
        B, C, H, W = x.shape
        tokenized_x = x.view(B, C, H * W).transpose(1, 2)
        assert tokenized_x.shape == (B, H * W, C)
        ### END: tokenization step

        tokenized_x = self.forward_tokenized(tokenized_x)

        ### BEGIN: detokenization step
        tokenized_x = tokenized_x.transpose(1, 2).view(B, C, H, W)
        assert tokenized_x.shape == (B, C, H, W)
        ### END: detokenization step
        return tokenized_x

    def forward_no_drop(self, x: torch.Tensor) -> torch.Tensor:
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

        
        B, HW, C = x.shape
        H = W = int(HW ** 0.5)
        assert H * W == HW

        phi = self.phi
        
        # Normalize input and phi
        if self.normalize:
            x = F.normalize(x, dim=2)  # [b, m, d]
            phi = self.scale * F.normalize(phi, dim=0)  # [d, n, p]

        # Compute dispatch and combine weights
        logits = torch.einsum("bmd,dnp->bmnp", x, phi)
        d = softmax(logits, dim=1)
        c = softmax(logits, dim=(2, 3))

        # Compute input slots as weighted average of input tokens using dispatch weights
        xs = torch.einsum("bmd,bmnp->bnpd", x, d)

        ### BEGIN: reshape xs into image_format to be processed by Conv experts
        assert xs.shape == (B, self.num_experts, 1, C)  # slots_per_expert is always 1
        xs = xs.squeeze(2)  # b n d
        xs = xs.reshape(B, self.num_experts, self.dim, 1, 1)  # b n d h c
        ### END: reshape xs into image_format to be processed by Conv experts


        ys = torch.stack([f_i(xs[:, i]) for i, f_i in enumerate(self.experts)], dim=1)
        # ys is (b, num_experts, dim, 1, 1)
        
        ### BEGIN: reshape ys into token_seq format to be processed by C
        ys = ys.reshape(B, self.num_experts, 1, self.dim).float()
        ### END: reshape ys into token_seq format to be processed by C

        # Compute output tokens as weighted average of output slots using combine weights
        y = torch.einsum("bnpd,bmnp->bmd", ys, c)

        return y

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
        
        B, HW, C = x.shape
        H = W = int(HW ** 0.5)
        assert H * W == HW
        
        phi = self.phi

        # Normalize input and phi
        if self.normalize:
            x = F.normalize(x, dim=2)  # [b, m, d]
            phi = self.scale * F.normalize(phi, dim=0)  # [d, n, p]

        # Compute dispatch and combine weights
        logits = torch.einsum("bmd,dnp->bmnp", x, phi)
        c = softmax(logits, dim=(2, 3))

        ### BEGIN: Set drop_expert_idxs with heuristic or random
        batch_size = x.shape[0]        
        if self.drop_type == "random":
            drop_expert_idxs = torch.from_numpy(
                np.array(
                    [
                        np.random.choice(
                            self.num_experts,
                            size=self.num_heuristic_drop, 
                            replace=False
                        )
                        for _ in range(batch_size)
                    ]))
            # print("random mask", drop_expert_idxs.shape)
        elif self.drop_type == "heuristic":
            if self.num_heuristic_drop is not None and self.num_heuristic_drop > 0:
                # print(f"Dropping {self.num_heuristic_drop} experts, for {batch_size} datapoints")
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
        else:
            raise ValueError(f"Invalid drop type: {self.drop_type}")
        ### BEGIN: Set drop_expert_idxs with heuristic or random
        
        d = softmax(logits, dim=1)
        # (batch_size, num_tokens, num_experts, num_expert_slots)

        # Compute input slots as weighted average of input tokens using dispatch weights
        xs = torch.einsum("bmd,bmnp->bnpd", x, d)

        ### BEGIN: reshape xs into image_format to be processed by Conv experts
        assert xs.shape == (B, self.num_experts, 1, C)  # slots_per_expert is always 1
        xs = xs.squeeze(2)  # b n d
        xs = xs.reshape(B, self.num_experts, self.dim, 1, 1)  # b n d h c
        ### END: reshape xs into image_format to be processed by Conv experts

        ### BEGIN: expert passes
        ys = torch.zeros_like(xs, device=xs.device, dtype=torch.float16)  # (batch, ne, slots (always 1), token_dim (assuming token dim in/out is same))
        for i, f_i in enumerate(self.experts):
            datapoint_mask = (~(drop_expert_idxs == i)).all(dim=1)
            ys[datapoint_mask, i] = f_i(xs[datapoint_mask, i, :, :])
            # have to compare ys[:, i] and ys_new[:, i]    
        ### END: expert passes

        ### BEGIN: reshape ys into token_seq format to be processed by C
        ys = ys.reshape(B, self.num_experts, 1, self.dim).float()
        ### END: reshape ys into token_seq format to be processed by C
        
        # Compute output tokens as weighted average of output slots using combine weights
        y = torch.einsum("bnpd,bmnp->bmd", ys, c)

        return y
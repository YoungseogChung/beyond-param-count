import os
from argparse import Namespace

import pickle
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class SoftMoeModel(nn.Module):
    def __init__(
        self,
        num_experts: int,
        num_slots: int,
        expert_pred_dim: int,
        tokenizer: callable,
        encoder: callable,
        decoder: callable,
        experts: list,
        phi: nn.Parameter,
        args: dict,
    ):
        super(SoftMoeModel, self).__init__()
        self.num_experts = num_experts
        self.num_slots = num_slots
        self.expert_pred_dim = expert_pred_dim
        self.tokenizer = tokenizer
        self.args = args
        # trainable parameters
        self.encoder = encoder
        self.decoder = decoder
        self.experts = nn.ModuleList(experts)
        self.phi = nn.Parameter(phi)

        if self.args.normalize_moe_input:
            self.normalization_scale_param = nn.Parameter(torch.tensor([1.0]))

        # experiment specific override parameters
        # i.e. things I want to try out for this experiment
        OVERRIDE_SETTINGS = ["uniform_d", "uniform_c", "routing_type"]
        self.override_settings = {
            k: getattr(self.args, k, None) for k in OVERRIDE_SETTINGS
        }

    def check_shapes(self):
        """Collection of shapes you want to check."""
        assert self.phi.shape == (self.token_dim, self.num_experts * self.num_slots)

    def tokenize(self, X: torch.Tensor, args: Namespace):
        """_summary_

        Args:
            X (torch.Tensor): input data, (B, ...)

        Returns:
            torch.Tensor: tokenized data, (B, ...)
        """
        tokenized_X = self.tokenizer(X)
        # whether to add dummies or not

        return tokenized_X

    def save(self, save_dir: str, save_name: str):
        """_summary_

        Args:
            save_dir (str): directory to save the model
        """
        original_device = next(self.parameters()).device
        self.to(torch.device("cpu"))

        # save tokenizer
        with open(f"{save_dir}/{save_name}_tokenizer.pkl", "wb") as f:
            pickle.dump(self.tokenizer, f)
        # save encoder
        # TODO: fix encoder saving
        if not hasattr(self.encoder, "parameters"):
            pass
            # with open(f"{save_dir}/{save_name}_encoder.pkl", "wb") as f:
            #     pickle.dump(self.encoder, f)
        # save decoder
        # TODO: fix decoder saving
        if not hasattr(self.decoder, "parameters"):
            with open(f"{save_dir}/{save_name}_decoder.pkl", "wb") as f:
                pickle.dump(self.decoder, f)
        # save phi
        # save experts
        torch.save(self.state_dict(), f"{save_dir}/{save_name}_moe.pt")

        self.to(original_device)

    def load(self, save_dir: str, save_name: str):
        """_summary_

        Args:
            save_dir (str): directory to load the model
        """
        # load tokenizer
        with open(f"{save_dir}/{save_name}_tokenizer.pkl", "rb") as f:
            self.tokenizer = pickle.load(f)
        # load encoder
        if os.path.exists(f"{save_dir}/{save_name}_encoder.pkl"):
            with open(f"{save_dir}/{save_name}_encoder.pkl", "rb") as f:
                self.encoder = pickle.load(f)
        # load decoder
        if os.path.exists(f"{save_dir}/{save_name}_decoder.pkl"):
            with open(f"{save_dir}/{save_name}_decoder.pkl", "rb") as f:
                self.decoder = pickle.load(f)
        # load rest of moe
        self.load_state_dict(torch.load(f"{save_dir}/{save_name}_moe.pt"))

    def normalize_moe_input(self, A: torch.Tensor, axis: int, eps: float=1e-6):
        """
        As implemented in Section 2.3 of their original paper.
        """
        norm = torch.sqrt(torch.square(A).sum(axis=axis, keepdims=True))
        return A / (norm + eps)
    
    def forward(self, X: torch.Tensor, return_xphi: bool = False, return_embedding: bool = False, 
                return_y_tilde: bool = False):
        """Forward pass of the soft MoE

        X:      (B, num_tokens,       patch_dim       )
        Phi:    (B, patch_dim,        num_expert_slots)
        XPhi:   (B, num_tokens,       num_expert_slots)
        D:      (B, num_tokens,       num_expert_slots)
        Xtilde: (B, num_expert_slots, patch_dim       )
        Ytilde: (B, num_expert_slots, expert_pred_dim )
        C:      (B, num_tokens,       num_expert_slots)
        Y:      (B, num_tokens,       expert_pred_dim )

        Args:
            X (torch.Tensor): input data, (B, ...)

        Returns:
            torch.Tensor: output prediction
        """
        X = self.tokenize(X, self.args)
        assert len(X.shape) == 3, "Input data must be 3D (B, num_tokens, token_dim)"
        batch_size, num_tokens, token_dim = X.shape
        X = self.encoder(X)

        # MoE logic goes here
        # Example: Soft assignment to experts and processing
        # Ensure to adapt it based on your specific logic and methods
        batch_size, num_tokens, token_dim = X.shape
        num_expert_slots = self.num_experts * self.num_slots

        phi = self.phi
        if self.args.normalize_moe_input:
            X = self.normalize_moe_input(X, axis=2)
            # breakpoint()
            phi = self.normalization_scale_param * self.normalize_moe_input(phi, axis=0)
            # print(self.normalization_scale_param)

        # The following code should be adapted based on how you handle phi, D, etc.
        # This is a simplified version based on your provided snippet

        # (B, num_tokens, token_dim) x (token_dim, num_slots) -> (B, num_tokens, num_slots)
        X_phi = torch.matmul(X, phi)
        assert X_phi.shape == (batch_size, num_tokens, num_expert_slots)
        if self.override_settings["uniform_d"]:
            D = torch.ones_like(D) / num_tokens
        else:
            D = torch.softmax(X_phi, dim=1)  # softmax across num_tokens dimension
        ###
        # ptwl = torch.sum(D, dim=2)
        # print(ptwl[:5])
        ###
        assert D.shape == X_phi.shape

        X_tilde = torch.matmul(D.transpose(1, 2), X)
        if self.override_settings["routing_type"] == "identity":
            # TODO: check that this is correct
            assert X.shape[1] == phi.shape[2]
            X_tilde = X
        # (B, 1 * 2 * 2, args.num_experts * args.num_slots)
        # x (B, 1 * 2 * 2, pp)
        # -> (B, args.num_experts * args.num_slots, pp)
        assert X_tilde.shape == (batch_size, num_expert_slots, token_dim)

        Y_tilde_list = []
        for idx in range(self.num_experts):
            # Stack args.num_slots number of rows of X_tilde to get a batch
            cur_input_slice = slice(idx * self.num_slots, (idx + 1) * self.num_slots)
            cur_X_tilde = X_tilde[:, cur_input_slice]  # (B, args.num_slots, patch_dim)
            # Flatten to make it a batch of size B * args.num_slots
            cur_X_tilde = cur_X_tilde.reshape(batch_size * self.num_slots, token_dim)
            cur_out = self.experts[idx].forward(cur_X_tilde)
            cur_out = cur_out.reshape(batch_size, self.num_slots, -1)
            Y_tilde_list.append(cur_out)
        Y_tilde = torch.cat(Y_tilde_list, dim=1)
        assert (
            (len(Y_tilde.shape) == 3)
            and (Y_tilde.shape[0] == batch_size)
            and (Y_tilde.shape[1] == self.num_experts * self.num_slots)
        )  # last dim of Y_tilde is expert_pred_dim
        if self.override_settings["uniform_c"]:
            C = torch.ones_like(C) / num_expert_slots
        else:
            C = torch.softmax(X_phi, dim=2)
        # ### BEGIN: hack
        # C = torch.softmax(D.detach(), dim=2)
        # assert C.shape == X_phi.shape
        # ### END: hack
        Y = torch.matmul(C, Y_tilde)
        # Y_tilde_list.append(Y_tilde.detach().cpu().numpy().copy())

        pred_Y = self.decoder(Y)  # (B, num_tokens, expert_pred_dim)

        if return_xphi and not return_embedding:
            return pred_Y, X_phi
        elif return_embedding and not return_xphi:
            return pred_Y, Y
        elif return_xphi and return_embedding and not return_y_tilde:
            return pred_Y, X_phi, Y
        elif return_xphi and return_embedding and return_y_tilde:
            return pred_Y, X_phi, Y, Y_tilde
        assert not (return_xphi or return_embedding)

        return pred_Y


    def fix_forward(self, X: torch.Tensor, return_xphi: bool = False, return_embedding: bool = False, 
                    return_y_tilde: bool = False):
            """Forward pass of the soft MoE

            X:      (B, num_tokens,       patch_dim       )
            Phi:    (B, patch_dim,        num_expert_slots)
            XPhi:   (B, num_tokens,       num_expert_slots)
            D:      (B, num_tokens,       num_expert_slots)
            Xtilde: (B, num_expert_slots, patch_dim       )
            Ytilde: (B, num_expert_slots, expert_pred_dim )
            C:      (B, num_tokens,       num_expert_slots)
            Y:      (B, num_tokens,       expert_pred_dim )

            Args:
                X (torch.Tensor): input data, (B, ...)

            Returns:
                torch.Tensor: output prediction
            """
            X = self.tokenize(X, self.args)
            assert len(X.shape) == 3, "Input data must be 3D (B, num_tokens, token_dim)"
            batch_size, num_tokens, token_dim = X.shape
            X = self.encoder(X)

            # MoE logic goes here
            # Example: Soft assignment to experts and processing
            # Ensure to adapt it based on your specific logic and methods
            batch_size, num_tokens, token_dim = X.shape
            num_expert_slots = self.num_experts * self.num_slots

            phi = self.phi
            if self.args.normalize_moe_input:
                X = self.normalize_moe_input(X, axis=2)
                # breakpoint()
                phi = self.normalization_scale_param * self.normalize_moe_input(phi, axis=0)
                # print(self.normalization_scale_param)

            # The following code should be adapted based on how you handle phi, D, etc.
            # This is a simplified version based on your provided snippet

            # (B, num_tokens, token_dim) x (token_dim, num_slots) -> (B, num_tokens, num_slots)
            X_phi = torch.matmul(X, phi)
            assert X_phi.shape == (batch_size, num_tokens, num_expert_slots)
            if self.override_settings["uniform_d"]:
                D = torch.ones_like(D) / num_tokens
            else:
                D = torch.softmax(X_phi, dim=1)  # softmax across num_tokens dimension
            ###
            # ptwl = torch.sum(D, dim=2)
            # print(ptwl[:5])
            ###
            assert D.shape == X_phi.shape

            X_tilde = torch.matmul(D.transpose(1, 2), X)
            if self.override_settings["routing_type"] == "identity":
                # TODO: check that this is correct
                assert X.shape[1] == phi.shape[2]
                X_tilde = X
            # (B, 1 * 2 * 2, args.num_experts * args.num_slots)
            # x (B, 1 * 2 * 2, pp)
            # -> (B, args.num_experts * args.num_slots, pp)
            assert X_tilde.shape == (batch_size, num_expert_slots, token_dim)

            Y_tilde_list = []
            for idx in range(self.num_experts):
                # Stack args.num_slots number of rows of X_tilde to get a batch
                cur_input_slice = slice(idx * self.num_slots, (idx + 1) * self.num_slots)
                cur_X_tilde = X_tilde[:, cur_input_slice]  # (B, args.num_slots, patch_dim)
                # Flatten to make it a batch of size B * args.num_slots
                cur_X_tilde = cur_X_tilde.reshape(batch_size * self.num_slots, token_dim)
                cur_out = self.experts[idx].forward(cur_X_tilde)
                cur_out = cur_out.reshape(batch_size, self.num_slots, -1)
                Y_tilde_list.append(cur_out)
            Y_tilde = torch.cat(Y_tilde_list, dim=1)
            assert (
                (len(Y_tilde.shape) == 3)
                and (Y_tilde.shape[0] == batch_size)
                and (Y_tilde.shape[1] == self.num_experts * self.num_slots)
            )  # last dim of Y_tilde is expert_pred_dim
            if self.override_settings["uniform_c"]:
                C = torch.ones_like(C) / num_expert_slots
            else:
                C = torch.softmax(X_phi, dim=2)
            # ### BEGIN: hack
            # C = torch.softmax(D.detach(), dim=2)
            # assert C.shape == X_phi.shape
            # ### END: hack
            Y = torch.matmul(C, Y_tilde)
            # Y_tilde_list.append(Y_tilde.detach().cpu().numpy().copy())

            pred_Y = self.decoder(Y)  # (B, num_tokens, expert_pred_dim)

            if return_xphi and not return_embedding:
                return pred_Y, X_phi
            elif return_embedding and not return_xphi:
                return pred_Y, Y
            elif return_xphi and return_embedding and not return_y_tilde:
                return pred_Y, X_phi, Y
            elif return_xphi and return_embedding and return_y_tilde:
                return pred_Y, X_phi, Y, Y_tilde
            assert not (return_xphi or return_embedding)

            return pred_Y


    def orig_expert_masked_forward(
        self, X: torch.Tensor, drop_expert_idxs: torch.Tensor, 
        num_heuristic_drop: int = 0,
        return_xphi: bool = False,
        return_embedding: bool = False,
        return_y_tilde_ranks: bool = False,
        return_y_tilde: bool = False,
        ):
        """Forward pass of the soft MoE

        X:      (B, num_tokens,       patch_dim       )
        Phi:    (B, patch_dim,        num_expert_slots)
        XPhi:   (B, num_tokens,       num_expert_slots)
        D:      (B, num_tokens,       num_expert_slots)
        Xtilde: (B, num_expert_slots, patch_dim       )
        Ytilde: (B, num_expert_slots, expert_pred_dim )
        C:      (B, num_tokens,       num_expert_slots)
        Y:      (B, num_tokens,       expert_pred_dim )

        Args:
            X (torch.Tensor): input data, (B, ...)
            drop_expert_idxs (torch.Tensor): 2D tensor of expert idxs per B point, (B, num_dropped_experts)
                these are not indicators, but actual indices of the experts in [0, num_experts - 1].
                Must be None if num_heuristic_drop > 0.
            num_heuristic_drop (int): Specifies the number of experts to drop, which are computed by the
                expert columns in C which have lowest total weight. Must be 0 if drop_expert_idxs is
                not None.
        Returns:
            torch.Tensor: output prediction
        """
        X = self.tokenize(X, self.args)
        assert len(X.shape) == 3, "Input data must be 3D (B, num_tokens, token_dim)"
        batch_size, num_tokens, token_dim = X.shape
        X = self.encoder(X)

        # MoE logic goes here
        # Example: Soft assignment to experts and processing
        # Ensure to adapt it based on your specific logic and methods
        batch_size, num_tokens, token_dim = X.shape
        num_expert_slots = self.num_experts * self.num_slots

        phi = self.phi
        if self.args.normalize_moe_input:
            X = self.normalize_moe_input(X, axis=2)
            phi = self.normalization_scale_param * self.normalize_moe_input(phi, axis=0)
            # print(self.normalization_scale_param)

        # The following code should be adapted based on how you handle phi, D, etc.
        # This is a simplified version based on your provided snippet

        # (B, num_tokens, token_dim) x (token_dim, num_slots)
        # -> (B, num_tokens, num_slots)
        X_phi = torch.matmul(X, phi)

        ###
        ### Only code block added to handle heuristic
        # get the C that would be used if we did not drop any experts
        if num_heuristic_drop > 0:
            assert drop_expert_idxs is None
            C_orig = torch.softmax(X_phi, dim=2).detach().cpu().numpy()
            # sum over num_tokens
            C_orig_sum = np.sum(C_orig, axis=1)
            assert np.shape(C_orig_sum) == (batch_size, num_expert_slots)
            # find the expert columns with the least total weight
            least_weight_expert_idxs = np.argsort(C_orig_sum, axis=1)
            # least_weight = np.sort(C_orig_sum, axis=1)
            # least_weight = least_weight[:, :num_heuristic_drop]
            # print(np.sum(least_weight) / (batch_size * num_tokens))
            drop_expert_idxs = least_weight_expert_idxs[:, :num_heuristic_drop]
            drop_expert_idxs = np.sort(drop_expert_idxs, axis=1)
            assert np.all(np.diff(drop_expert_idxs) > 0)
            # to get number of sets of dropped experts
            # print(len(np.unique(drop_expert_idxs.astype(int), axis=0)))
            # import math
            # print(math.comb(num_expert_slots, num_heuristic_drop))
            assert np.shape(drop_expert_idxs) == (batch_size, num_heuristic_drop)
            drop_expert_idxs = torch.Tensor(drop_expert_idxs).to(int)
        else:
            assert drop_expert_idxs is not None
        ### heuristic handling code block ends here
        ###

        mask = torch.ones_like(X_phi)  # (B, num_tokens, num_expert_slots)
        # drop_expert_idxs is of shape (B, num_dropped_experts)
        # mask[:, :, drop_expert_idxs] = 0  # original static mask
        mask[np.arange(batch_size)[:, None], :, drop_expert_idxs] = 0
        # ## BEGIN: for loop method
        # for drop_idx in drop_expert_idxs:
        #     mask[np.arange(batch_size), :, drop_expert_idxs[:, drop_idx]] = 0
        # ## END: for loop method
        X_phi = X_phi.masked_fill(mask == 0, -float('inf'))
        assert X_phi.shape == (batch_size, num_tokens, num_expert_slots)
        if self.override_settings["uniform_d"]:
            D = torch.ones_like(D) / num_tokens
        else:
            D = F.softmax(X_phi, dim=1)  # softmax across num_tokens dimension

        # This is necessary because softmax is taken with only -torch.inf
        D = D.masked_fill(~torch.isfinite(D), 0)
        ###
        # ptwl = torch.sum(D, dim=2)
        # print(ptwl[:5])
        ###
        assert D.shape == X_phi.shape

        X_tilde = torch.matmul(D.transpose(1, 2), X)
        if self.override_settings["routing_type"] == "identity":
            # TODO: check that this is correct
            assert X.shape[1] == self.phi.shape[2]
            X_tilde = X
        # (B, 1 * 2 * 2, args.num_experts * args.num_slots)
        # x (B, 1 * 2 * 2, pp)
        # -> (B, args.num_experts * args.num_slots, pp)
        assert X_tilde.shape == (batch_size, num_expert_slots, token_dim)

        Y_tilde_list = []
        for idx in range(self.num_experts):
            # Stack args.num_slots number of rows of X_tilde to get a batch
            cur_input_slice = slice(idx * self.num_slots, (idx + 1) * self.num_slots)
            cur_X_tilde = X_tilde[:, cur_input_slice]  # (B, args.num_slots, patch_dim)
            # Flatten to make it a batch of size B * args.num_slots
            cur_X_tilde = cur_X_tilde.reshape(batch_size * self.num_slots, token_dim)
            cur_out = self.experts[idx].forward(cur_X_tilde)
            cur_out = cur_out.reshape(batch_size, self.num_slots, -1)
            Y_tilde_list.append(cur_out)
        Y_tilde = torch.cat(Y_tilde_list, dim=1)
        assert (
            (len(Y_tilde.shape) == 3)
            and (Y_tilde.shape[0] == batch_size)
            and (Y_tilde.shape[1] == self.num_experts * self.num_slots)
        )  # last dim of Y_tilde is expert_pred_dim
        if self.override_settings["uniform_c"]:
            C = torch.ones_like(C) / num_expert_slots
        else:
            C = torch.softmax(X_phi, dim=2)
        # ### BEGIN: hack
        # C = torch.softmax(D.detach(), dim=2)
        # assert C.shape == X_phi.shape
        # ### END: hack
        Y = torch.matmul(C, Y_tilde)
        # Y_tilde_list.append(Y_tilde.detach().cpu().numpy().copy())

        pred_Y = self.decoder(Y)  # (B, num_tokens, expert_pred_dim)

        if return_xphi and not return_embedding:
            return pred_Y, X_phi
        elif return_embedding and not return_xphi:
            return pred_Y, Y
        elif return_xphi and return_embedding and not return_y_tilde_ranks and not return_y_tilde:
            return pred_Y, X_phi, Y, None  # return None for Y_tilde_ranks
        elif return_xphi and return_embedding and return_y_tilde_ranks:
            Y_tilde_ranks = [
                np.linalg.matrix_rank(Y_tilde[i].detach().cpu().numpy()) 
                for i in range(X.shape[0])
            ]
            return pred_Y, X_phi, Y, Y_tilde_ranks
        elif return_xphi and return_embedding and return_y_tilde:
            if isinstance(drop_expert_idxs, torch.Tensor):
                return_drop_idx = drop_expert_idxs.detach().cpu().numpy().copy()
            elif isinstance(drop_expert_idxs, np.ndarray):
                return_drop_idx = drop_expert_idxs.copy()
            aux_info = {
                "y_tilde": Y_tilde.detach().cpu().numpy().copy(),
                "drop_idx": return_drop_idx,
            }
            return pred_Y, X_phi, Y, aux_info
        assert not (return_xphi or return_embedding)

        return pred_Y


    def expert_masked_forward(
        self, X: torch.Tensor, drop_expert_idxs: torch.Tensor, 
        num_heuristic_drop: int = 0,
        return_xphi: bool = False,
        return_embedding: bool = False,
        return_y_tilde_ranks: bool = False,
        return_y_tilde: bool = False,
        ):
        """Forward pass of the soft MoE

        X:      (B, num_tokens,       patch_dim       )
        Phi:    (B, patch_dim,        num_expert_slots)
        XPhi:   (B, num_tokens,       num_expert_slots)
        D:      (B, num_tokens,       num_expert_slots)
        Xtilde: (B, num_expert_slots, patch_dim       )
        Ytilde: (B, num_expert_slots, expert_pred_dim )
        C:      (B, num_tokens,       num_expert_slots)
        Y:      (B, num_tokens,       expert_pred_dim )

        Args:
            X (torch.Tensor): input data, (B, ...)
            drop_expert_idxs (torch.Tensor): 2D tensor of expert idxs per B point, (B, num_dropped_experts)
                these are not indicators, but actual indices of the experts in [0, num_experts - 1].
                Must be None if num_heuristic_drop > 0.
            num_heuristic_drop (int): Specifies the number of experts to drop, which are computed by the
                expert columns in C which have lowest total weight. Must be 0 if drop_expert_idxs is
                not None.
        Returns:
            torch.Tensor: output prediction
        """
        # assert False
        X = self.tokenize(X, self.args)
        assert len(X.shape) == 3, "Input data must be 3D (B, num_tokens, token_dim)"
        batch_size, num_tokens, token_dim = X.shape
        X = self.encoder(X)

        # MoE logic goes here
        # Example: Soft assignment to experts and processing
        # Ensure to adapt it based on your specific logic and methods
        batch_size, num_tokens, token_dim = X.shape
        num_expert_slots = self.num_experts * self.num_slots

        phi = self.phi
        if self.args.normalize_moe_input:
            X = self.normalize_moe_input(X, axis=2)
            phi = self.normalization_scale_param * self.normalize_moe_input(phi, axis=0)
            # print(self.normalization_scale_param)

        # The following code should be adapted based on how you handle phi, D, etc.
        # This is a simplified version based on your provided snippet

        # (B, num_tokens, token_dim) x (token_dim, num_slots)
        # -> (B, num_tokens, num_slots)
        X_phi = torch.matmul(X, phi)

        ###
        ### Only code block added to handle heuristic
        # get the C that would be used if we did not drop any experts
        if num_heuristic_drop > 0:
            with torch.no_grad():
                assert drop_expert_idxs is None
                C_orig = torch.softmax(X_phi, dim=2)
                # sum over num_tokens
                C_orig_sum = torch.sum(C_orig, dim=1)
                assert C_orig_sum.shape == (batch_size, num_expert_slots)
                # find the expert columns with the least total weight
                least_weight_expert_idxs = torch.argsort(C_orig_sum, dim=1)
                drop_expert_idxs = least_weight_expert_idxs[:, :num_heuristic_drop]
                assert drop_expert_idxs.shape == (batch_size, num_heuristic_drop)
                # drop_expert_idxs = torch.Tensor(drop_expert_idxs).to(int)
        else:
            assert drop_expert_idxs is not None
        ### heuristic handling code block ends here
        ###

        # drop_expert_idxs is of shape (B, num_dropped_experts)
        assert X_phi.shape == (batch_size, num_tokens, num_expert_slots)
        if self.override_settings["uniform_d"]:
            D = torch.ones_like(D) / num_tokens
        else:
            D = F.softmax(X_phi, dim=1)  # softmax across num_tokens dimension
        assert D.shape == X_phi.shape

        X_tilde = torch.matmul(D.transpose(1, 2), X)
        if self.override_settings["routing_type"] == "identity":
            # TODO: check that this is correct
            assert X.shape[1] == self.phi.shape[2]
            X_tilde = X
        # (B, 1 * 2 * 2, args.num_experts * args.num_slots)
        # x (B, 1 * 2 * 2, pp)
        # -> (B, args.num_experts * args.num_slots, pp)
        assert X_tilde.shape == (batch_size, num_expert_slots, token_dim)


        Y_tilde = torch.zeros(batch_size, num_expert_slots, self.expert_pred_dim).to(X.device)

        for idx in range(self.num_experts):
            ### datapoint idxs that will be processed by this expert
            datapoint_mask = (~(drop_expert_idxs == idx)).all(dim=1)
            if torch.sum(datapoint_mask) == 0:
                continue
            ###
            # Stack args.num_slots number of rows of X_tilde to get a batch
            cur_input_slice = slice(idx * self.num_slots, (idx + 1) * self.num_slots)
            cur_X_tilde = X_tilde[datapoint_mask, cur_input_slice]  # (<B, args.num_slots, patch_dim)
            # Flatten to make it a batch of size B * args.num_slots
            cur_X_tilde = cur_X_tilde.reshape(-1, token_dim)
            cur_out = self.experts[idx].forward(cur_X_tilde)
            cur_out = cur_out.reshape(-1, self.num_slots, self.expert_pred_dim)  # (<B, args.num_slots, patch_dim)
            Y_tilde[datapoint_mask, idx:idx+1] = cur_out
        assert (
            (len(Y_tilde.shape) == 3)
            and (Y_tilde.shape[0] == batch_size)
            and (Y_tilde.shape[1] == self.num_experts * self.num_slots)
        )  # last dim of Y_tilde is expert_pred_dim
        if self.override_settings["uniform_c"]:
            C = torch.ones_like(C) / num_expert_slots
        else:
            C = torch.softmax(X_phi, dim=2)
        # ### BEGIN: hack
        # C = torch.softmax(D.detach(), dim=2)
        # assert C.shape == X_phi.shape
        # ### END: hack
        Y = torch.matmul(C, Y_tilde)
        temp_C = C.clone()
        temp_C[torch.arange(batch_size)[:, None], :, drop_expert_idxs] = 0
        temp_Y = torch.matmul(temp_C, Y_tilde)
        assert torch.allclose(Y, temp_Y)
        # Y_tilde_list.append(Y_tilde.detach().cpu().numpy().copy())

        pred_Y = self.decoder(Y)  # (B, num_tokens, expert_pred_dim)

        aux_info = {}
        if return_xphi and not return_embedding:
            aux_info["X_phi"] = X_phi.detach().cpu().numpy().copy()
        elif return_embedding and not return_xphi:
            aux_info["Y"] = Y.detach().cpu().numpy().copy()
        elif return_xphi and return_embedding and not return_y_tilde_ranks and not return_y_tilde:
            aux_info["X_phi"] = X_phi.detach().cpu().numpy().copy()
            aux_info["Y"] = Y.detach().cpu().numpy().copy()
        elif return_xphi and return_embedding and return_y_tilde_ranks:
            Y_tilde_ranks = [
                np.linalg.matrix_rank(Y_tilde[i].detach().cpu().numpy()) 
                for i in range(X.shape[0])
            ]
            aux_info["X_phi"] =  X_phi.detach().cpu().numpy().copy()
            aux_info["Y"] = Y.detach().cpu().numpy().copy()
            aux_info["Y_tilde_ranks"] = Y_tilde_ranks
            # return pred_Y, X_phi, Y, Y_tilde_ranks
        elif return_xphi and return_embedding and return_y_tilde:
            if isinstance(drop_expert_idxs, torch.Tensor):
                return_drop_idx = drop_expert_idxs.detach().cpu().numpy().copy()
            elif isinstance(drop_expert_idxs, np.ndarray):
                return_drop_idx = drop_expert_idxs.copy()
            
            aux_info["X_phi"] = X_phi.detach().cpu().numpy().copy()
            aux_info["Y"] = Y.detach().cpu().numpy().copy()
            aux_info["y_tilde"] = Y_tilde.detach().cpu().numpy().copy()
            aux_info["drop_idx"] = return_drop_idx
            aux_info["C"] = temp_C.detach().cpu().numpy().copy()
            aux_info["D"] = D.detach().cpu().numpy().copy()
            # return pred_Y, X_phi, Y, aux_info
        # assert not (return_xphi or return_embedding)

        return pred_Y, aux_info


def get_model_predictions(model, dataloader, device):
    # with torch.no_grad():  # TODO: do we need this or not?
    pred_list = []
    y_list = []
    for batch_data in dataloader:
        if len(batch_data) == 2:
            X, y = batch_data
            aux_info = None
        elif len(batch_data) == 3:
            X, y, aux_info = batch_data
        X = X.to(device).float()
        y = y.to(device)

        pred = model.forward(X)
        pred_list.append(pred.detach().cpu().numpy().copy())
        y_list.append(y.detach().cpu().numpy().copy())
    pred_y = np.concatenate(pred_list, axis=0)
    true_y = np.concatenate(y_list, axis=0)

    return pred_y, true_y

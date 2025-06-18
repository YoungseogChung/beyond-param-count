"""
Wrapper to use SoftMoEViT with HuggingFace's Trainer
Author: Youngseog Chung
Date: February 10, 2024
"""

import torch
from transformers.modeling_outputs import SequenceClassifierOutput

from soft_moe.vision_transformer import SoftMoEVisionTransformer
from soft_moe.soft_moe import SoftMoELayerWrapper


class HFWrapperSMoEViT(SoftMoEVisionTransformer):
    """Wrapper to use SoftMoEViT with HuggingFace's Trainer

    Args:
        moe_mlp_ratio (float): Ratio of the hidden size of the MLP in the MoE layer.
    """
    def forward(self, **kwargs):
        x = kwargs.get("pixel_value") 
        y = kwargs.get("label")
        x = self.forward_features(x)
        logits = self.forward_head(x)
        if y is not None:
            loss = torch.nn.CrossEntropyLoss()(logits, y)

        return SequenceClassifierOutput(loss=loss, logits=logits)
    
    def forward_logits(self, x: torch.Tensor):
        x = self.forward_features(x)
        logits = self.forward_head(x)
        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        x = self.forward_features(x)
        logits = self.forward_head(x)
        return logits

    def make_vmap_model(self):
        for block_idx, b in enumerate(self.blocks):
            if hasattr(b, "mlp"):
                cur_mlp = getattr(b, "mlp")
                if isinstance(cur_mlp, SoftMoELayerWrapper):
                    cur_mlp.make_vmap_model()
        self.is_vmap_model = True

    def prepare_vmap_model_loading(self):
        """
        All this does is send the base_expert of the SMoE layers to meta device
        since the loaded state_dict will have meta device tensors for these parameters

        When loading the state_dict of a vmap_model,
        - first call make_vmap_model
        - then send to a device before sending base_expert of SMoE layers to meta
        - then call prepare_vmap_model_loading which sends base_expert SMoE layers to meta
        - then load the state_dict (which already stores meta device tensors for base_expert)
        """
        assert self.is_vmap_model
        for block_idx, b in enumerate(self.blocks):
            if hasattr(b, "mlp"):
                cur_mlp = getattr(b, "mlp")
                if isinstance(cur_mlp, SoftMoELayerWrapper):
                    if not cur_mlp.base_expert_on_meta:
                        cur_mlp._send_base_expert_to_meta()
                        assert cur_mlp.base_expert_on_meta
            
    def _set_latency_forward(self, use_latency_method, single_input):
        """ Set forward function of model to use vmap """
        # Use forward function to latency_forward for latency measurement
        for b in self.blocks:
            if hasattr(b, "mlp"):
                cur_mlp = getattr(b, "mlp")
                if isinstance(cur_mlp, SoftMoELayerWrapper):
                    if use_latency_method:
                        if single_input:
                            cur_mlp.forward_function = cur_mlp.latency_forward
                        else:
                            cur_mlp.forward_function = cur_mlp.latency_batched_input_forward
                    else:
                        cur_mlp.forward_function = cur_mlp.forward_original

    def _set_latency_drop(self, use_latency_method, single_input):
        """ Set forward function of model to use vmap """
        # Use forward function to latency_forward for latency measurement
        for b in self.blocks:
            if hasattr(b, "mlp"):
                cur_mlp = getattr(b, "mlp")
                if isinstance(cur_mlp, SoftMoELayerWrapper):
                    if use_latency_method:
                        # cur_mlp.forward_with_drop = cur_mlp.latency_forward_with_drop
                        if single_input:
                            cur_mlp.forward_with_drop = cur_mlp.latency_single_input_for_loop_drop
                        else:
                            cur_mlp.forward_with_drop = cur_mlp.latency_batched_input_for_loop_drop
                    else:
                        cur_mlp.forward_with_drop = cur_mlp.forward_with_drop_no_renorm

    def _vmap_forward_switch(self, set_vmap: bool):
        """ Set forward function of model to use vmap """
        # Use default forward function to vmapped_forward
        for b in self.blocks:
            if hasattr(b, "mlp"):
                cur_mlp = getattr(b, "mlp")
                if isinstance(cur_mlp, SoftMoELayerWrapper):
                    if set_vmap:
                        cur_mlp.forward_function = cur_mlp.forward_vmap
                    else:
                        cur_mlp.forward_function = cur_mlp.forward_original

    def _drop_forward_switch(self, use_renorm: bool):
        """ Set forward function of model to use vmap """
        # Use default forward function to vmapped_forward
        for b in self.blocks:
            if hasattr(b, "mlp"):
                cur_mlp = getattr(b, "mlp")
                if isinstance(cur_mlp, SoftMoELayerWrapper):
                    if use_renorm:
                        cur_mlp.forward_with_drop = cur_mlp.forward_with_drop_with_renorm
                    else:
                        cur_mlp.forward_with_drop = cur_mlp.forward_with_drop_no_renorm

    def _apply_pre_heuristic(self, drop_expert_idxs, num_heuristic_drop):
        """ Prepare model before heuristic """
        # set num_heuristic_drop to all SoftMoELayerWrapper of self
        heuristic_set_mlp_list = []
        mask_set_mlp_list = []
        if num_heuristic_drop > 0:
            for b in self.blocks:
                if hasattr(b, "mlp"):
                    cur_mlp = getattr(b, "mlp")
                    if isinstance(cur_mlp, SoftMoELayerWrapper):
                        cur_mlp.num_heuristic_drop = num_heuristic_drop
                        cur_mlp.drop_expert_idxs = None
                        cur_mlp.forward_function = cur_mlp.forward_with_drop
                        heuristic_set_mlp_list.append(cur_mlp)
        elif drop_expert_idxs is not None:
            for b in self.blocks:
                if hasattr(b, "mlp"):
                    cur_mlp = getattr(b, "mlp")
                    if isinstance(cur_mlp, SoftMoELayerWrapper):
                        cur_mlp.num_heuristic_drop = None
                        cur_mlp.drop_expert_idxs = drop_expert_idxs
                        cur_mlp.forward_function = cur_mlp.forward_with_drop
                        mask_set_mlp_list.append(cur_mlp)

        return heuristic_set_mlp_list, mask_set_mlp_list

    def _apply_post_heurstic(self, heuristic_set_mlp_list, mask_set_mlp_list):
        """ Reset model after heurstic """
        # reset
        for cur_mlp in heuristic_set_mlp_list:
            cur_mlp.num_heuristic_drop = None
            cur_mlp.forward_function = cur_mlp.forward_original
        for cur_mlp in mask_set_mlp_list:
            cur_mlp.drop_expert_idxs = None
            cur_mlp.forward_function = cur_mlp.forward_original

    def _check_pre_heurstic_done(self, drop_expert_idxs, num_heuristic_drop):
        if num_heuristic_drop > 0:
            for b in self.blocks:
                if hasattr(b, "mlp"):
                    cur_mlp = getattr(b, "mlp")
                    if isinstance(cur_mlp, SoftMoELayerWrapper):
                        assert cur_mlp.num_heuristic_drop == num_heuristic_drop
                        assert cur_mlp.drop_expert_idxs is None
                        # cur_mlp.forward_function = cur_mlp.forward_with_drop
        elif drop_expert_idxs is not None:
            for b in self.blocks:
                if hasattr(b, "mlp"):
                    cur_mlp = getattr(b, "mlp")
                    if isinstance(cur_mlp, SoftMoELayerWrapper):
                        assert cur_mlp.num_heuristic_drop is None
                        assert (cur_mlp.drop_expert_idxs == drop_expert_idxs).all()
                        # cur_mlp.forward_function = cur_mlp.forward_with_drop

    def _check_post_heurstic_done(self):
        for b in self.blocks:
            if hasattr(b, "mlp"):
                cur_mlp = getattr(b, "mlp")
                if isinstance(cur_mlp, SoftMoELayerWrapper):
                    assert cur_mlp.num_heuristic_drop is None
                    assert cur_mlp.drop_expert_idxs is None

    def predict_with_drop(
        self, x: torch.Tensor, drop_expert_idxs=None, num_heuristic_drop=0
    ) -> torch.Tensor:

        heuristic_set_mlp_list, mask_set_mlp_list = self._apply_pre_heuristic(
            drop_expert_idxs, num_heuristic_drop)
        self._check_pre_heurstic_done(drop_expert_idxs, num_heuristic_drop)
        # predict
        logits = self.predict(x)
        self._apply_post_heurstic(heuristic_set_mlp_list, mask_set_mlp_list)
        self._check_post_heurstic_done()
        
        return logits

    def extract_moe_layer_info(self):
        info_per_moe_layer = []
        moe_layer_idx = 0
        for b in self.blocks:
            if hasattr(b, "mlp"):
                cur_mlp = getattr(b, "mlp")
                if isinstance(cur_mlp, SoftMoELayerWrapper):
                    # c cache
                    c_cache = cur_mlp.c_cache
                    cur_mlp.c_cache = []
                    # phi
                    # phi = cur_mlp.phi.detach().cpu().numpy()
                    moe_info = {
                        f"{moe_layer_idx}_c_cache": c_cache,
                    }
                    info_per_moe_layer.append(moe_info)
                    moe_layer_idx += 1
        return info_per_moe_layer
                    
    def num_expert_parameters(self, verbose=False):
        expert_numel = 0
        for item in self.named_parameters():
            if "expert" in item[0] and "base_expert" not in item[0]:
                # print(item[0])
                expert_numel += item[1].numel()
                if verbose:
                    print(f"{item[0]}: {item[1].numel()}")
        return expert_numel

    def num_smoe_parameters(self):
        num_expert_param = 0
        num_phi_param = 0
        num_scale_param = 0
        for b in self.blocks:
            if hasattr(b, "mlp"):
                cur_mlp = getattr(b, "mlp")
                if isinstance(cur_mlp, SoftMoELayerWrapper):
                    for item in cur_mlp.named_parameters():
                        if "phi" in item[0]:
                            num_phi_param += item[1].numel()
                        elif "scale" in item[0]:
                            num_scale_param += item[1].numel()
                        else:
                            num_expert_param += item[1].numel()
        return num_expert_param, num_phi_param, num_scale_param

    def num_parameters(self):
        num_param = 0
        for item in self.parameters():
            num_param += item.numel()
        return num_param


def hf_soft_moe_vit_tiny(
    num_experts=128, slots_per_expert=1, moe_layer_index=6, **kwargs
) -> SoftMoEVisionTransformer:
    return HFWrapperSMoEViT(
        num_experts=num_experts,
        slots_per_expert=slots_per_expert,
        moe_layer_index=moe_layer_index,
        embed_dim=192,
        depth=12,
        num_heads=3,
        **kwargs,
    )


def hf_soft_moe_vit_small(
    num_experts=128, slots_per_expert=1, moe_layer_index=6, **kwargs
) -> SoftMoEVisionTransformer:
    return HFWrapperSMoEViT(
        num_experts=num_experts,
        slots_per_expert=slots_per_expert,
        moe_layer_index=moe_layer_index,
        embed_dim=384,
        depth=12,
        num_heads=6,
        **kwargs,
    )


def hf_soft_moe_vit_base(
    num_experts=128, slots_per_expert=1, moe_layer_index=6, **kwargs
) -> SoftMoEVisionTransformer:
    return HFWrapperSMoEViT(
        num_experts=num_experts,
        slots_per_expert=slots_per_expert,
        moe_layer_index=moe_layer_index,
        embed_dim=768,
        depth=12,
        num_heads=12,
        **kwargs,
    )


def hf_soft_moe_vit_large(
    num_experts=128, slots_per_expert=1, moe_layer_index=12, **kwargs
) -> SoftMoEVisionTransformer:
    return HFWrapperSMoEViT(
        num_experts=num_experts,
        slots_per_expert=slots_per_expert,
        moe_layer_index=moe_layer_index,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        **kwargs,
    )


def hf_soft_moe_vit_huge(
    num_experts=128, slots_per_expert=1, moe_layer_index=16, **kwargs
) -> SoftMoEVisionTransformer:
    return HFWrapperSMoEViT(
        num_experts=num_experts,
        slots_per_expert=slots_per_expert,
        moe_layer_index=moe_layer_index,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        **kwargs,
    )

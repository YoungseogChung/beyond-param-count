"""
Utility functions for constructing Soft MoE variants of ViT models.
Author: Youngseog Chung
Date: April 1, 2024
"""

from soft_moe.hf_vision_transformer import (
    hf_soft_moe_vit_tiny,
    hf_soft_moe_vit_small,
    hf_soft_moe_vit_base,
)
from soft_moe.stitch_model import (
    stitch_tiny,
    stitch_small,
    stitch_base,
)

def construct_smoe_vit_model(model_type: str, num_experts: int, moe_mlp_ratio: int, num_classes: int):
    if model_type == "smoe_tiny":
        model_fn = hf_soft_moe_vit_tiny
    elif model_type == "smoe_small":
        model_fn = hf_soft_moe_vit_small
    elif model_type == "smoe_base":
        model_fn = hf_soft_moe_vit_base
    elif model_type == "stitch_tiny":
        model_fn = stitch_tiny
    elif model_type == "stitch_small":
        model_fn = stitch_small
    elif model_type == "stitch_base":
        model_fn = stitch_base
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    model = model_fn(
        num_experts=num_experts, 
        moe_mlp_ratio=moe_mlp_ratio, 
        num_classes=num_classes
    )

    return model


def convert_smoe_vit_vmap_state_dict_to_original(state_dict):
    """ Convert the state_dict of a vmap model to an original model state_dict

    Args:
        state_dict (dict): has vmap model attributes like "stacked_expert_weight...".

    Returns (dict): modified state_dict that adheres to the original non-vmap model,
        which has attribtues like "experts.2.fc1.weight..".
    """

    # Need to get rid of
    #   - stacked_... parameters
    #   - base_expert... parameters

    insert_param_dict = {}
    stacked_param_keys = []
    base_expert_param_keys = []
    for k, v in state_dict.items():
        if "stacked" in k:
            # first parse the name
            param_name_list = k.split('.')
            assert "stacked" in param_name_list[-1]
            param_name_prefix = param_name_list[:-1]

            # get the type of the param
            param_type = param_name_list[-1].removeprefix("stacked_expert_")
            param_type = param_type.split('_')
            # ["fc1", "weight"]
            assert len(param_type) == 2
            assert param_type[0] in ["fc1", "fc2"]
            assert param_type[1] in ["weight", "bias"]

            num_experts = v.shape[0]
            # assert that this is consistent with self.num_experts
            for e_i in range(num_experts):
                param_name_rest = [
                    "experts", str(e_i), param_type[0], param_type[1]
                ]
                param_name_full = ".".join(param_name_prefix + param_name_rest)
                # param_assign_value = copy.deepcopy(v[e_i])
                param_assign_value = v[e_i]
                insert_param_dict[param_name_full] = param_assign_value

            stacked_param_keys.append(k)
        elif "base_expert" in k:
            base_expert_param_keys.append(k)

    for k,v in insert_param_dict.items():
        state_dict[k] = v

    for k in stacked_param_keys:
        del state_dict[k]

    for k in base_expert_param_keys:
        del state_dict[k]

    return state_dict

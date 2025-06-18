import copy
from typing import Optional, Union
import torch
import torch.nn as nn
from transformers import ViTForImageClassification
from transformers.modeling_outputs import BaseModelOutput

# for SMoEPickupViTEncoded
from soft_moe.hf_vision_transformer import HFWrapperSMoEViT
from soft_moe import SoftMoELayerWrapper


class ViTPretrainedHalfEncoder(nn.Module):
    def __init__(self, num_keep_encoder_block=6):
        """
        Args:
            num_keep_encoder_block: The number of encoder blocks to keep,
                counting from the first encoder block 
        """
        super(ViTPretrainedHalfEncoder, self).__init__()
        # Load original pretrained ViT model
        MODEL_NAME = 'google/vit-base-patch16-224'
        vit_model = ViTForImageClassification.from_pretrained(MODEL_NAME)
        self.vit = vit_model.vit
        self.config = vit_model.config
        # Keep only the first num_keep_encoder_block encoder blocks
        use_encoder_layer = copy.deepcopy(self.vit.encoder.layer[:num_keep_encoder_block])
        del self.vit.encoder.layer
        self.vit.encoder.layer = use_encoder_layer

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, BaseModelOutput]:
        
        ### BEGIN: copied from ViTForImageClassification.forward
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        ### END: copied from ViTForImageClassification.forward

        ### BEGIN: copied from ViTModel.forward
        bool_masked_pos = None
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.vit.get_head_mask(head_mask, self.config.num_hidden_layers)

        # TODO: maybe have a cleaner way to cast the input (from `ImageProcessor` side?)
        expected_dtype = self.vit.embeddings.patch_embeddings.projection.weight.dtype
        if pixel_values.dtype != expected_dtype:
            pixel_values = pixel_values.to(expected_dtype)

        embedding_output = self.vit.embeddings(
            pixel_values, bool_masked_pos=bool_masked_pos, interpolate_pos_encoding=interpolate_pos_encoding
        )

        encoder_outputs = self.vit.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        ### END: copied from ViTModel.forward

        return encoder_outputs
    

class SMoEPickupViTEncoded(HFWrapperSMoEViT):
    
    def __init__(self, check_num_blocks: int = 6, **kwargs):
        """
        Args:
            check_num_blocks: The number of blocks to expect for smoe blocks
        """
        super(SMoEPickupViTEncoded, self).__init__(**kwargs)
        # only keep smoe blocks in the current model
        del self.patch_embed
        del self.norm_pre
        orig_blocks = copy.deepcopy(self.blocks)
        smoe_blocks = orig_blocks[6:]
        del self.blocks
        self.blocks = smoe_blocks
        # assert that all blocks remaining in self.blocks are smoe
        for b in self.blocks:
            if hasattr(b, "mlp"):
                cur_mlp = getattr(b, "mlp")
                assert isinstance(cur_mlp, SoftMoELayerWrapper)
                    
        assert len(self.blocks) == check_num_blocks
        
    def forward_features(self, x):
        # Make sure the only input this ever gets is non-smoe-block-encoded from ViTPretrainedHalfEncoder
        x = self.blocks(x)
        x = self.norm(x)
        return x


class StitchedSMoE(nn.Module):
    def __init__(
        self, 
        vit_half_encoder, 
        smoe_pickup_vit_encoded,
    ) -> None:
        super().__init__()
        self.vit_half_encoder = vit_half_encoder  # always outputs dim 768
        self.smoe_pickup_vit_encoded = smoe_pickup_vit_encoded
        self.num_experts = self.smoe_pickup_vit_encoded.num_experts
        self.num_classes = self.smoe_pickup_vit_encoded.num_classes
        # in case the embed_dim of the smoe component is less than 768
        self.dim_reducer = None
        if self.smoe_pickup_vit_encoded.embed_dim < 768:
            self.dim_reducer = nn.Linear(
                768, 
                self.smoe_pickup_vit_encoded.embed_dim, 
                bias=True
            )

        # freeze the vit_half_encoder
        for param in self.vit_half_encoder.parameters():
            param.requires_grad = False


    def forward(self, **kwargs):
        x = kwargs.get("pixel_value")
        y = kwargs.get("label")
        vit_encoded_out = self.vit_half_encoder(x)
        vit_encoded_out = vit_encoded_out.last_hidden_state  # (batch, num_tokens, token_dim)
        if self.dim_reducer is not None:
            vit_encoded_out = self.dim_reducer(vit_encoded_out)
        smoe_out = self.smoe_pickup_vit_encoded(pixel_value=vit_encoded_out, label=y)

        return smoe_out

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        x = self.vit_half_encoder(x)
        x = x.last_hidden_state
        if self.dim_reducer is not None:
            x = self.dim_reducer(x)
        logits = self.smoe_pickup_vit_encoded.predict(x=x)
        return logits

    def forward_logits(self, x: torch.Tensor):
        x = self.vit_half_encoder(x)
        x = x.last_hidden_state
        if self.dim_reducer is not None:
            x = self.dim_reducer(x)
        logits = self.smoe_pickup_vit_encoded.forward_logits(x=x)
        return logits

    def predict_with_drop(self, x: torch.Tensor, drop_expert_idxs=None, num_heuristic_drop=0) -> torch.Tensor:
        # Follow same procedure of HFWrapperSMoEViT.predict_with_drop
        heuristic_set_mlp_list, mask_set_mlp_list = self.smoe_pickup_vit_encoded._apply_pre_heuristic(
            drop_expert_idxs, num_heuristic_drop)
        self.smoe_pickup_vit_encoded._check_pre_heurstic_done(drop_expert_idxs, num_heuristic_drop)
        logits = self.predict(x)
        self.smoe_pickup_vit_encoded._apply_post_heurstic(heuristic_set_mlp_list, mask_set_mlp_list)
        self.smoe_pickup_vit_encoded._check_post_heurstic_done()
        return logits
    
    def num_expert_parameters(self, **kwargs):
        return self.smoe_pickup_vit_encoded.num_expert_parameters(**kwargs)
    
    def num_smoe_parameters(self, **kwargs):
        return self.smoe_pickup_vit_encoded.num_smoe_parameters(**kwargs)

    def num_parameters(self):
        num_param = 0
        for item in self.parameters():
            num_param += item.numel()
        return num_param


# instantiations of StitchedSMoE
def stitch_tiny(
    num_experts=128, slots_per_expert=1, moe_layer_index=6, **kwargs
) -> StitchedSMoE:
    vit_half_encoder = ViTPretrainedHalfEncoder()
    smoe_pickup_vit_encoded = SMoEPickupViTEncoded(
        num_experts=num_experts,
        slots_per_expert=slots_per_expert,
        moe_layer_index=moe_layer_index,
        embed_dim=192,
        depth=12,
        num_heads=3,
        **kwargs,
    )
    return StitchedSMoE(vit_half_encoder, smoe_pickup_vit_encoded)

def stitch_small(
    num_experts=128, slots_per_expert=1, moe_layer_index=6, **kwargs
) -> StitchedSMoE:
    vit_half_encoder = ViTPretrainedHalfEncoder()
    smoe_pickup_vit_encoded = SMoEPickupViTEncoded(
        num_experts=num_experts,
        slots_per_expert=slots_per_expert,
        moe_layer_index=moe_layer_index,
        embed_dim=384,
        depth=12,
        num_heads=6,
        **kwargs,
    )
    return StitchedSMoE(vit_half_encoder, smoe_pickup_vit_encoded)

def stitch_base(
    num_experts=128, slots_per_expert=1, moe_layer_index=6, **kwargs
) -> StitchedSMoE:
    vit_half_encoder = ViTPretrainedHalfEncoder()
    smoe_pickup_vit_encoded = SMoEPickupViTEncoded(
        num_experts=num_experts,
        slots_per_expert=slots_per_expert,
        moe_layer_index=moe_layer_index,
        embed_dim=768,
        depth=12,
        num_heads=12,
        **kwargs,
    )
    return StitchedSMoE(vit_half_encoder, smoe_pickup_vit_encoded)

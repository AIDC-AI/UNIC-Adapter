# Copyright 2025 The HuggingFace Team. All rights reserved.
# Modified by AIDC-AI, 2025
# This project is licensed under the MIT License (SPDX-License-identifier: MIT).

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange
from torch import Tensor

from diffusers.utils import deprecate, logging
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.activations import GEGLU, GELU, ApproximateGELU, FP32SiLU, SwiGLU
from diffusers.models.attention_processor import Attention, JointAttnProcessor2_0
from diffusers.models.embeddings import SinusoidalPositionalEmbedding
from diffusers.models.normalization import AdaLayerNorm, AdaLayerNormContinuous, AdaLayerNormZero, RMSNorm, SD35AdaLayerNormZeroX
from diffusers.models.controlnet import zero_module
from diffusers.models.attention import FeedForward


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


def apply_rope_query_or_key(x: Tensor, freqs_cis: Tensor):
    x_ = x.float().reshape(*x.shape[:-1], -1, 1, 2)
    x_out = freqs_cis[..., 0] * x_[..., 0] + freqs_cis[..., 1] * x_[..., 1]
    return x_out.reshape(*x.shape).type_as(x)


def AttnProcessor_UNIC_Adapter(attn, attn_control,
                    hidden_states: torch.FloatTensor, encoder_hidden_states: torch.FloatTensor, 
                    hidden_states_control: torch.FloatTensor, encoder_hidden_states_control: torch.FloatTensor,
                    pe: torch.FloatTensor, 
                    img_to_q_control: torch.FloatTensor, 
                    scale = 1.0,
                ):
    residual = hidden_states
    batch_size = encoder_hidden_states.shape[0]

    # `sample` projections.
    query = attn.to_q(hidden_states)
    key = attn.to_k(hidden_states)
    value = attn.to_v(hidden_states)

    # attention
    inner_dim = key.shape[-1]
    head_dim = inner_dim // attn.heads

    query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    if attn.norm_q is not None:
        query = attn.norm_q(query)
    if attn.norm_k is not None:
        key = attn.norm_k(key)

    # `context` projections.
    encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
    encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
    encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

    encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
        batch_size, -1, attn.heads, head_dim
    ).transpose(1, 2)
    encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
        batch_size, -1, attn.heads, head_dim
    ).transpose(1, 2)
    encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
        batch_size, -1, attn.heads, head_dim
    ).transpose(1, 2)

    if attn.norm_added_q is not None:
        encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
    if attn.norm_added_k is not None:
        encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

    query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
    key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
    value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)


    hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
    hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
    hidden_states = hidden_states.to(query.dtype)

    # Split the attention outputs.
    hidden_states, encoder_hidden_states = (
        hidden_states[:, : residual.shape[1]],
        hidden_states[:, residual.shape[1] :],
    )

    ## adapter branch
    residual_control = hidden_states_control
    batch_size_control = encoder_hidden_states_control.shape[0]

    # `sample` projections.
    query_cross = img_to_q_control.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    
    query_control = attn_control.to_q(hidden_states_control)
    key_control = attn_control.to_k(hidden_states_control)        
    value_control = attn_control.to_v(hidden_states_control)

    # attention
    inner_dim_control = key_control.shape[-1]
    head_dim_control = inner_dim_control // attn_control.heads

    query_control = query_control.view(batch_size, -1, attn_control.heads, head_dim_control).transpose(1, 2)
    key_control = key_control.view(batch_size, -1, attn_control.heads, head_dim_control).transpose(1, 2)
    value_control = value_control.view(batch_size, -1, attn_control.heads, head_dim_control).transpose(1, 2)

    if attn_control.norm_q is not None:
        query_control = attn_control.norm_q(query_control)
    if attn_control.norm_k is not None:
        key_control = attn_control.norm_k(key_control)

    # `context` projections.
    encoder_hidden_states_query_proj_control = attn_control.add_q_proj(encoder_hidden_states_control)
    encoder_hidden_states_key_proj_control = attn_control.add_k_proj(encoder_hidden_states_control)
    encoder_hidden_states_value_proj_control = attn_control.add_v_proj(encoder_hidden_states_control)

    encoder_hidden_states_query_proj_control = encoder_hidden_states_query_proj_control.view(
        batch_size, -1, attn_control.heads, head_dim_control
    ).transpose(1, 2)
    encoder_hidden_states_key_proj_control = encoder_hidden_states_key_proj_control.view(
        batch_size, -1, attn_control.heads, head_dim_control
    ).transpose(1, 2)
    encoder_hidden_states_value_proj_control = encoder_hidden_states_value_proj_control.view(
        batch_size, -1, attn_control.heads, head_dim_control
    ).transpose(1, 2)

    if attn_control.norm_added_q is not None:
        encoder_hidden_states_query_proj_control = attn_control.norm_added_q(encoder_hidden_states_query_proj_control)
    if attn_control.norm_added_k is not None:
        encoder_hidden_states_key_proj_control = attn_control.norm_added_k(encoder_hidden_states_key_proj_control)

    query_control = torch.cat([query_control, encoder_hidden_states_query_proj_control], dim=2)
    key_control = torch.cat([key_control, encoder_hidden_states_key_proj_control], dim=2)
    value_control = torch.cat([value_control, encoder_hidden_states_value_proj_control], dim=2)

    key_cross = key_control
    value_cross = value_control
    
    query_control, key_control = apply_rope(query_control, key_control, pe)

    hidden_states_control = F.scaled_dot_product_attention(query_control, key_control, value_control, dropout_p=0.0, is_causal=False)
    hidden_states_control = hidden_states_control.transpose(1, 2).reshape(batch_size_control, -1, attn_control.heads * head_dim_control)
    hidden_states_control = hidden_states_control.to(query_control.dtype)

    # Split the attention outputs.
    hidden_states_control, encoder_hidden_states_control = (
        hidden_states_control[:, : residual_control.shape[1]],
        hidden_states_control[:, residual_control.shape[1] :],
    )
    
    ## cross attention with image-instruction features
    query_cross = apply_rope_query_or_key(query_cross, pe[:,:,:query_cross.shape[2]])
    key_cross = apply_rope_query_or_key(key_cross, pe[:,:,:key_cross.shape[2]])

    hidden_states_cross = F.scaled_dot_product_attention(query_cross, key_cross, value_cross, dropout_p=0.0, is_causal=False)
    hidden_states_cross = hidden_states_cross.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
    hidden_states_cross = hidden_states_cross.to(query.dtype)

    hidden_states = hidden_states + scale * hidden_states_cross

    # linear proj
    hidden_states = attn.to_out[0](hidden_states)
    # dropout
    hidden_states = attn.to_out[1](hidden_states)
    if not attn.context_pre_only:
        encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

    ## adapter branch
    # linear proj
    hidden_states_control = attn_control.to_out[0](hidden_states_control)
    # dropout
    hidden_states_control = attn_control.to_out[1](hidden_states_control)
    if not attn_control.context_pre_only:
        encoder_hidden_states_control = attn_control.to_add_out(encoder_hidden_states_control)

    return hidden_states, encoder_hidden_states, hidden_states_control, encoder_hidden_states_control


@maybe_allow_in_graph
class JointTransformerBlockControl(nn.Module):
    r"""
    A Transformer block following the MMDiT architecture, introduced in Stable Diffusion 3.

    Reference: https://arxiv.org/abs/2403.03206

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        context_pre_only (`bool`): Boolean to determine if we should add some blocks associated with the
            processing of `context` conditions.
    """

    def __init__(self, dim, num_attention_heads, attention_head_dim, qk_norm=None, use_dual_attention=False, context_pre_only=False, control_scale=1.0):
        super().__init__()

        self.use_dual_attention = use_dual_attention
        self.context_pre_only = context_pre_only
        context_norm_type = "ada_norm_continous" if context_pre_only else "ada_norm_zero"

        if use_dual_attention:
            self.norm1 = SD35AdaLayerNormZeroX(dim)
        else:
            self.norm1 = AdaLayerNormZero(dim)

        if context_norm_type == "ada_norm_continous":
            self.norm1_context = AdaLayerNormContinuous(
                dim, dim, elementwise_affine=False, eps=1e-6, bias=True, norm_type="layer_norm"
            )
        elif context_norm_type == "ada_norm_zero":
            self.norm1_context = AdaLayerNormZero(dim)
        else:
            raise ValueError(
                f"Unknown context_norm_type: {context_norm_type}, currently only support `ada_norm_continous`, `ada_norm_zero`"
            )
        if hasattr(F, "scaled_dot_product_attention"):
            processor = JointAttnProcessor2_0()
        else:
            raise ValueError(
                "The current PyTorch version does not support the `scaled_dot_product_attention` function."
            )
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=context_pre_only,
            bias=True,
            processor=processor,
            qk_norm=qk_norm,
            eps=1e-6,
        )
        if use_dual_attention:
            self.attn2 = Attention(
                query_dim=dim,
                cross_attention_dim=None,
                dim_head=attention_head_dim,
                heads=num_attention_heads,
                out_dim=dim,
                bias=True,
                processor=processor,
                qk_norm=qk_norm,
                eps=1e-6,
            )
        else:
            self.attn2 = None

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        if not context_pre_only:
            self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self.ff_context = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")
        else:
            self.norm2_context = None
            self.ff_context = None

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0
        
        ## adapter branch
        # self.control_type = control_type
        if self.use_dual_attention:
            self.norm1_control = SD35AdaLayerNormZeroX(dim)
        else:
            self.norm1_control = AdaLayerNormZero(dim)
        self.control_scale = control_scale
        if context_norm_type == "ada_norm_continous":
            self.norm1_context_control = AdaLayerNormContinuous(
                dim, dim, elementwise_affine=False, eps=1e-6, bias=True, norm_type="layer_norm"
            )
        elif context_norm_type == "ada_norm_zero":
            self.norm1_context_control = AdaLayerNormZero(dim)
        else:
            raise ValueError(
                f"Unknown context_norm_type: {context_norm_type}, currently only support `ada_norm_continous`, `ada_norm_zero`"
            )
        if hasattr(F, "scaled_dot_product_attention"):
            processor = JointAttnProcessor2_0()
        else:
            raise ValueError(
                "The current PyTorch version does not support the `scaled_dot_product_attention` function."
            )
        self.attn_control = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=context_pre_only,
            bias=True,
            processor=processor,
            qk_norm=qk_norm,
            eps=1e-6,
        )
        if use_dual_attention:
            self.attn2_control = Attention(
                query_dim=dim,
                cross_attention_dim=None,
                dim_head=attention_head_dim,
                heads=num_attention_heads,
                out_dim=dim,
                bias=True,
                processor=processor,
                qk_norm=qk_norm,
                eps=1e-6,
            )
        else:
            self.attn2_control = None

        self.norm2_control = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

        if not context_pre_only:
            self.norm2_context_control = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        else:
            self.norm2_context_control = None

        self.to_q_control = nn.Linear(dim, dim, bias=True)
        
    # Copied from diffusers.models.attention.BasicTransformerBlock.set_chunk_feed_forward
    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def forward(
        self, hidden_states: torch.FloatTensor, encoder_hidden_states: torch.FloatTensor, temb: torch.FloatTensor,
        hidden_states_control: torch.FloatTensor, encoder_hidden_states_control: torch.FloatTensor, temb_control: torch.FloatTensor,
        pe: torch.FloatTensor
    ):
        if self.use_dual_attention:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp, norm_hidden_states2, gate_msa2 = self.norm1(
                hidden_states, emb=temb
            )
        else:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)


        if self.context_pre_only:
            norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states, temb)
        else:
            norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
                encoder_hidden_states, emb=temb
            )

        ## adapter brach 
        if self.use_dual_attention:
            norm_hidden_states_control, gate_msa_control, shift_mlp_control, scale_mlp_control, gate_mlp_control, norm_hidden_states2_control, gate_msa2_control = self.norm1_control(hidden_states_control, emb=temb_control)
        else:
            norm_hidden_states_control, gate_msa_control, shift_mlp_control, scale_mlp_control, gate_mlp_control = self.norm1_control(hidden_states_control, emb=temb_control)

        if self.context_pre_only:
            norm_encoder_hidden_states_control = self.norm1_context_control(encoder_hidden_states_control, temb_control)
        else:
            norm_encoder_hidden_states_control, c_gate_msa_control, c_shift_mlp_control, c_scale_mlp_control, c_gate_mlp_control = self.norm1_context_control(
                encoder_hidden_states_control, emb=temb_control
            )
        
        ## adapter cross attention
        img_to_q_control = self.to_q_control(norm_hidden_states)
        attn_output, context_attn_output, attn_output_control, context_attn_output_control = AttnProcessor_UNIC_Adapter(
            self.attn, self.attn_control,
            hidden_states=norm_hidden_states, encoder_hidden_states=norm_encoder_hidden_states,
            hidden_states_control=norm_hidden_states_control, encoder_hidden_states_control=norm_encoder_hidden_states_control,
            pe=pe,
            img_to_q_control=img_to_q_control,
            scale=self.control_scale
        )

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        if self.use_dual_attention:
            attn_output2 = self.attn2(hidden_states=norm_hidden_states2)
            attn_output2 = gate_msa2.unsqueeze(1) * attn_output2
            hidden_states = hidden_states + attn_output2

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
        else:
            ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output

        # Process attention outputs for the `encoder_hidden_states`.
        if self.context_pre_only:
            encoder_hidden_states = None
        else:
            context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
            encoder_hidden_states = encoder_hidden_states + context_attn_output

            norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
            norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
            if self._chunk_size is not None:
                # "feed_forward_chunk_size" can be used to save memory
                context_ff_output = _chunked_feed_forward(
                    self.ff_context, norm_encoder_hidden_states, self._chunk_dim, self._chunk_size
                )
            else:
                context_ff_output = self.ff_context(norm_encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

        ## adapter brach 
        # Process attention outputs for the `hidden_states`.
        attn_output_control = gate_msa_control.unsqueeze(1) * attn_output_control
        hidden_states_control = hidden_states_control + attn_output_control

        if self.use_dual_attention:
            attn_output2_control = self.attn2_control(hidden_states=norm_hidden_states2_control)
            attn_output2_control = gate_msa2_control.unsqueeze(1) * attn_output2_control
            hidden_states_control = hidden_states_control + attn_output2_control

        norm_hidden_states_control = self.norm2_control(hidden_states_control)
        norm_hidden_states_control = norm_hidden_states_control * (1 + scale_mlp_control[:, None]) + shift_mlp_control[:, None]
        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            ff_output_control = _chunked_feed_forward(self.ff, norm_hidden_states_control, self._chunk_dim, self._chunk_size)
        else:
            ff_output_control = self.ff(norm_hidden_states_control)
        ff_output_control = gate_mlp_control.unsqueeze(1) * ff_output_control

        hidden_states_control = hidden_states_control + ff_output_control

        # Process attention outputs for the `encoder_hidden_states`.
        if self.context_pre_only:
            encoder_hidden_states_control = None
        else:
            context_attn_output_control = c_gate_msa_control.unsqueeze(1) * context_attn_output_control
            encoder_hidden_states_control = encoder_hidden_states_control + context_attn_output_control

            norm_encoder_hidden_states_control = self.norm2_context_control(encoder_hidden_states_control)
            norm_encoder_hidden_states_control = norm_encoder_hidden_states_control * (1 + c_scale_mlp_control[:, None]) + c_shift_mlp_control[:, None]
            if self._chunk_size is not None:
                # "feed_forward_chunk_size" can be used to save memory
                context_ff_output_control = _chunked_feed_forward(
                    self.ff_context, norm_encoder_hidden_states_control, self._chunk_dim, self._chunk_size
                )
            else:
                context_ff_output_control = self.ff_context(norm_encoder_hidden_states_control)
            encoder_hidden_states_control = encoder_hidden_states_control + c_gate_mlp_control.unsqueeze(1) * context_ff_output_control
        ##conrol brach end

        return encoder_hidden_states, hidden_states, encoder_hidden_states_control, hidden_states_control
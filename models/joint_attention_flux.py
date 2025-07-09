# Copyright 2025 The HuggingFace Team. All rights reserved.
# Modified by AIDC-AI, 2025
# This project is licensed under the MIT License (SPDX-License-identifier: MIT).

from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention, AttentionProcessor, FluxAttnProcessor2_0, FusedFluxAttnProcessor2_0
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormContinuous, AdaLayerNormZero, AdaLayerNormZeroSingle
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.embeddings import CombinedTimestepGuidanceTextProjEmbeddings, CombinedTimestepTextProjEmbeddings, FluxPosEmbed
from diffusers.models.modeling_outputs import Transformer2DModelOutput

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def AttnProcessor_UNIC_Adapter(attn, attn_control,
                        hidden_states: torch.FloatTensor, encoder_hidden_states: torch.FloatTensor = None, 
                        hidden_states_control: torch.FloatTensor = None, encoder_hidden_states_control: torch.FloatTensor = None,
                        image_rotary_emb: torch.FloatTensor = None, 
                        scale=1.0,
                    ):

    batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

    # `sample` projections.
    query = attn.to_q(hidden_states)
    key = attn.to_k(hidden_states)
    value = attn.to_v(hidden_states)

    query_control = attn_control.to_q(hidden_states_control)
    key_control = attn_control.to_k(hidden_states_control)
    value_control = attn_control.to_v(hidden_states_control)

    inner_dim = key.shape[-1]
    head_dim = inner_dim // attn.heads

    query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    query_control = query_control.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    key_control = key_control.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    value_control = value_control.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    if attn.norm_q is not None:
        query = attn.norm_q(query)
        query_control = attn_control.norm_q(query_control)
    if attn.norm_k is not None:
        key = attn.norm_k(key)
        key_control = attn_control.norm_k(key_control)

    # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
    if encoder_hidden_states is not None:
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

        encoder_hidden_states_query_proj_control = attn_control.add_q_proj(encoder_hidden_states_control)
        encoder_hidden_states_key_proj_control = attn_control.add_k_proj(encoder_hidden_states_control)
        encoder_hidden_states_value_proj_control = attn_control.add_v_proj(encoder_hidden_states_control)

        encoder_hidden_states_query_proj_control = encoder_hidden_states_query_proj_control.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_key_proj_control = encoder_hidden_states_key_proj_control.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_value_proj_control = encoder_hidden_states_value_proj_control.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)

        if attn.norm_added_q is not None:
            encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            encoder_hidden_states_query_proj_control = attn_control.norm_added_q(encoder_hidden_states_query_proj_control)
        if attn.norm_added_k is not None:
            encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)
            encoder_hidden_states_key_proj_control = attn_control.norm_added_k(encoder_hidden_states_key_proj_control)

        # attention
        query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
        key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
        value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        # attention
        query_control = torch.cat([encoder_hidden_states_query_proj_control, query_control], dim=2)
        key_control = torch.cat([encoder_hidden_states_key_proj_control, key_control], dim=2)
        value_control = torch.cat([encoder_hidden_states_value_proj_control, value_control], dim=2)

    query_cross = query
    key_cross = key_control
    value_cross = value_control

    if image_rotary_emb is not None:
        from diffusers.models.embeddings import apply_rotary_emb

        query = apply_rotary_emb(query, image_rotary_emb)
        key = apply_rotary_emb(key, image_rotary_emb)

        query_control = apply_rotary_emb(query_control, image_rotary_emb)
        key_control = apply_rotary_emb(key_control, image_rotary_emb)

        query_cross = apply_rotary_emb(query_cross, image_rotary_emb)
        key_cross = apply_rotary_emb(key_cross, image_rotary_emb)

    hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
    hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
    hidden_states = hidden_states.to(query.dtype)

    hidden_states_control = F.scaled_dot_product_attention(query_control, key_control, value_control, dropout_p=0.0, is_causal=False)
    hidden_states_control = hidden_states_control.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
    hidden_states_control = hidden_states_control.to(query.dtype)

    hidden_states_cross = F.scaled_dot_product_attention(query_cross, key_cross, value_cross, dropout_p=0.0, is_causal=False)
    hidden_states_cross = hidden_states_cross.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
    hidden_states_cross = hidden_states_cross.to(query.dtype)

    hidden_states = hidden_states + scale * hidden_states_cross
    if encoder_hidden_states is not None:
        encoder_hidden_states, hidden_states = (
            hidden_states[:, : encoder_hidden_states.shape[1]],
            hidden_states[:, encoder_hidden_states.shape[1] :],
        )

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        encoder_hidden_states_control, hidden_states_control = (
            hidden_states_control[:, : encoder_hidden_states_control.shape[1]],
            hidden_states_control[:, encoder_hidden_states_control.shape[1] :],
        )

        # linear proj
        hidden_states_control = attn_control.to_out[0](hidden_states_control)
        # dropout
        hidden_states_control = attn_control.to_out[1](hidden_states_control)
        encoder_hidden_states_control = attn_control.to_add_out(encoder_hidden_states_control)

        return hidden_states, encoder_hidden_states, hidden_states_control, encoder_hidden_states_control
    else:
        return hidden_states, None, hidden_states_control, None,


@maybe_allow_in_graph
class JointFluxSingleTransformerBlockControl(nn.Module):
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

    def __init__(self, dim, num_attention_heads, attention_head_dim, mlp_ratio=4.0):
        super().__init__()
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm = AdaLayerNormZeroSingle(dim)
        self.norm_control = AdaLayerNormZeroSingle(dim)
        self.proj_mlp = nn.Linear(dim, self.mlp_hidden_dim)
        self.act_mlp = nn.GELU(approximate="tanh")
        self.proj_out = nn.Linear(dim + self.mlp_hidden_dim, dim)

        processor = FluxAttnProcessor2_0()
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            processor=processor,
            qk_norm="rms_norm",
            eps=1e-6,
            pre_only=True,
        )

        self.attn_control = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            processor=processor,
            qk_norm="rms_norm",
            eps=1e-6,
            pre_only=True,
        )

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        hidden_states_control: torch.FloatTensor,
        temb_control: torch.FloatTensor,
        image_rotary_emb=None,
        proportional_attention=True,
    ):
        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))

        residual_control = hidden_states_control
        norm_hidden_states_control, gate_control = self.norm_control(hidden_states_control, emb=temb_control)
        mlp_hidden_states_control = self.act_mlp(self.proj_mlp(norm_hidden_states_control))

        attn_output, _, attn_output_control, _ = AttnProcessor_UNIC_Adapter(
            self.attn,
            self.attn_control,
            hidden_states=norm_hidden_states,
            encoder_hidden_states=None,
            hidden_states_control=norm_hidden_states_control,
            encoder_hidden_states_control=None,
            image_rotary_emb=image_rotary_emb
        )

        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        gate = gate.unsqueeze(1)
        hidden_states = gate * self.proj_out(hidden_states)
        hidden_states = residual + hidden_states
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        hidden_states_control = torch.cat([attn_output_control, mlp_hidden_states_control], dim=2)
        gate_control = gate_control.unsqueeze(1)
        hidden_states_control = gate_control * self.proj_out(hidden_states_control)
        hidden_states_control = residual_control + hidden_states_control
        if hidden_states_control.dtype == torch.float16:
            hidden_states_control = hidden_states_control.clip(-65504, 65504)

        return hidden_states, hidden_states_control


@maybe_allow_in_graph
class JointFluxTransformerBlockControl(nn.Module):
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

    def __init__(self, dim, num_attention_heads, attention_head_dim, qk_norm="rms_norm", eps=1e-6):
        super().__init__()

        self.norm1 = AdaLayerNormZero(dim)
        self.norm1_control = AdaLayerNormZero(dim)
        self.norm1_context = AdaLayerNormZero(dim)
        self.norm1_context_control = AdaLayerNormZero(dim)
        if hasattr(F, "scaled_dot_product_attention"):
            processor = FluxAttnProcessor2_0()
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
            context_pre_only=False,
            bias=True,
            processor=processor,
            qk_norm=qk_norm,
            eps=eps,
        )
        self.attn_control = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            processor=processor,
            qk_norm=qk_norm,
            eps=eps,
        )
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2_control = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2_context_control = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_context = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        hidden_states_control: torch.FloatTensor,
        encoder_hidden_states_control: torch.FloatTensor,
        temb_control: torch.FloatTensor,
        image_rotary_emb=None,
        proportional_attention=True,
    ):
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )

        norm_hidden_states_control, gate_msa_control, shift_mlp_control, scale_mlp_control, gate_mlp_control = self.norm1_control(hidden_states_control, emb=temb_control)

        norm_encoder_hidden_states_control, c_gate_msa_control, c_shift_mlp_control, c_scale_mlp_control, c_gate_mlp_control = self.norm1_context_control(
            encoder_hidden_states_control, emb=temb_control
        )

        # Attention.
        attn_output, context_attn_output, attn_output_control, context_attn_output_control = AttnProcessor_UNIC_Adapter(
            self.attn, self.attn_control,
            hidden_states=norm_hidden_states, encoder_hidden_states=norm_encoder_hidden_states,
            hidden_states_control=norm_hidden_states_control, encoder_hidden_states_control=norm_encoder_hidden_states_control,
            image_rotary_emb=image_rotary_emb,
            scale=1.0,
        )

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output

        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        # Process attention outputs for the `hidden_states`.
        attn_output_control = gate_msa_control.unsqueeze(1) * attn_output_control
        hidden_states_control = hidden_states_control + attn_output_control

        norm_hidden_states_control = self.norm2_control(hidden_states_control)
        norm_hidden_states_control = norm_hidden_states_control * (1 + scale_mlp_control[:, None]) + shift_mlp_control[:, None]

        ff_output_control = self.ff(norm_hidden_states_control)
        ff_output_control = gate_mlp_control.unsqueeze(1) * ff_output_control

        hidden_states_control = hidden_states_control + ff_output_control

        context_attn_output_control = c_gate_msa_control.unsqueeze(1) * context_attn_output_control
        encoder_hidden_states_control = encoder_hidden_states_control + context_attn_output_control

        norm_encoder_hidden_states_control = self.norm2_context_control(encoder_hidden_states_control)
        norm_encoder_hidden_states_control = norm_encoder_hidden_states_control * (1 + c_scale_mlp_control[:, None]) + c_shift_mlp_control[:, None]

        context_ff_output_control = self.ff_context(norm_encoder_hidden_states_control)
        encoder_hidden_states_control = encoder_hidden_states_control + c_gate_mlp_control.unsqueeze(1) * context_ff_output_control
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states_control = encoder_hidden_states_control.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states, encoder_hidden_states_control, hidden_states_control


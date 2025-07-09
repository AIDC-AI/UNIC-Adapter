# Copyright (C) 2025 AIDC-AI
# This project is licensed under the MIT License (SPDX-License-identifier: MIT).

import torch
from PIL import Image
import numpy as np
import os 
import copy
import argparse
import time
import json
from torchvision import transforms
from models.transformer_sd3 import SD3Transformer2DModelControl
from models.pipeline_sd3 import StableDiffusion3Pipeline
from models.transformer_flux import FluxTransformer2DModelControl
from models.pipeline_flux import FluxPipeline

from models.content_filters import PixtralContentFilter


def load_model_and_pipeline(model_type, ckpt_path, base_model_path):
    if model_type == "sd3_medium":
        pipe = StableDiffusion3Pipeline.from_pretrained(base_model_path, torch_dtype=torch.float16)
        unic_adapter_model = SD3Transformer2DModelControl(
            sample_size = 128,
            patch_size = 2,
            in_channels = 16,
            num_layers = 24,
            attention_head_dim = 64,
            num_attention_heads = 24,
            joint_attention_dim = 4096,
            caption_projection_dim = 1536,
            pooled_projection_dim = 2048,
            out_channels = 16,
            pos_embed_max_size = 192,
            qk_norm = None,
            dual_attention_layers = [],
        )
    elif model_type == "sd3.5_medium":
        pipe = StableDiffusion3Pipeline.from_pretrained(base_model_path, torch_dtype=torch.float16)
        unic_adapter_model = SD3Transformer2DModelControl(
            sample_size = 128,
            patch_size = 2,
            in_channels = 16,
            num_layers = 24,
            attention_head_dim = 64,
            num_attention_heads = 24,
            joint_attention_dim = 4096,
            caption_projection_dim = 1536,
            pooled_projection_dim = 2048,
            out_channels = 16,
            pos_embed_max_size = 384,
            qk_norm = "rms_norm",
            dual_attention_layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        )
    elif model_type == "flux":
        pipe = FluxPipeline.from_pretrained(base_model_path, torch_dtype=torch.float16)
        unic_adapter_model = FluxTransformer2DModelControl(
            patch_size = 1,
            in_channels = 64,
            num_layers = 19,
            num_single_layers = 38,
            attention_head_dim = 128,
            num_attention_heads = 24,
            joint_attention_dim = 4096,
            pooled_projection_dim = 768,
            guidance_embeds = True,
            axes_dims_rope = (16, 56, 56),
        )
    else:
        raise NotImplementedError
    
    model_sd = pipe.transformer.state_dict()
    adapter_sd = torch.load(ckpt_path, map_location="cpu")
    model_sd.update(adapter_sd)
    load_info = unic_adapter_model.load_state_dict(model_sd, strict=True)
    unic_adapter_model.half()
    unic_adapter_model.eval()
    pipe.transformer = unic_adapter_model
    pipe = pipe.to("cuda")
    return pipe

def load_json(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def load_prompt_image_instruction(sample):
    resize_fun = transforms.Compose([
        transforms.Resize((sample["output_h"], sample["output_w"])),
    ])
    transform_rgb = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    instruct_image = Image.open(sample["instruction_image"]).convert("RGB")
    instruct_image = transform_rgb(resize_fun(instruct_image))
    instruct_prompt = sample["instruction_prompt"]

    if sample["task_type"] == "editing":
        prompt = instruct_prompt
    else:
        prompt = sample["prompt"]

    return prompt, instruct_image, instruct_prompt

def main(args):
    sample_list = load_json(args.config_json_path)
    pipe = load_model_and_pipeline(args.model_type, args.ckpt_path, args.base_model_path)
    integrity_checker = PixtralContentFilter(torch.device("cuda"))
    for sample in sample_list:
        prompt, instruct_image, instruct_prompt = load_prompt_image_instruction(sample)
        generator = torch.Generator("cuda").manual_seed(sample["seed"])
        if args.model_type == "flux":
            image = pipe(
                prompt=prompt,
                num_inference_steps=28,
                guidance_scale=sample["prompt_guidance"],
                height=sample["output_h"],
                width=sample["output_w"],
                instruct_prompt=instruct_prompt,
                instruct_image=instruct_image,
                generator=generator,
                num_images_per_prompt=1,
            ).images[0]
        else:
            image = pipe(
                prompt=prompt,
                num_inference_steps=28,
                guidance_scale=7.0,
                height=sample["output_h"],
                width=sample["output_w"],
                instruct_prompt=instruct_prompt,
                instruct_image=instruct_image,
                prompt_guidance=sample["prompt_guidance"],
                image_instruction_guidance=sample["image_instruction_guidance"],
                generator=generator,
                num_images_per_prompt=1,
            ).images[0]
        os.makedirs(os.path.dirname(sample["save_path"].replace("output_imgs", "output_imgs_{}".format(args.model_type))), exist_ok=True)
        
        image_ = np.array(image) / 255.0
        image_ = 2 * image_ - 1
        image_ = torch.from_numpy(image_).to("cuda", dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2)
        if integrity_checker.test_image(image_):
            raise ValueError("The generated image was flagged by the content filter for potentially sensitive or copyrighted content.")
        else:
            image.save(sample["save_path"].replace("output_imgs", "output_imgs_{}".format(args.model_type)))
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="UNIC-Adapter inference")
    parser.add_argument('--model_type', type=str, default='sd3_medium', help='sd3_medium, sd3.5_medium, flux')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Adapter weight path')
    parser.add_argument('--base_model_path', type=str, required=True, help='Base model weight path')
    parser.add_argument('--config_json_path', type=str, required=True, help='Config json path')
    args = parser.parse_args()
    main(args)
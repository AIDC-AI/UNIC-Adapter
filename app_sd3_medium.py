# Copyright (C) 2025 AIDC-AI
# This project is licensed under the MIT License (SPDX-License-identifier: MIT).

import gradio as gr
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
from torchvision import transforms

from models.transformer_sd3 import SD3Transformer2DModelControl
from models.pipeline_sd3 import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained("/Path/to/your/local/stable-diffusion-3-medium-diffusers/", torch_dtype=torch.float16)
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
model_sd = pipe.transformer.state_dict()
adapter_sd = torch.load("./ckpts/sd3_medium_adapter.pth", map_location="cpu")
model_sd.update(adapter_sd)
load_info = unic_adapter_model.load_state_dict(model_sd, strict=True)
unic_adapter_model.type(torch.float16)
pipe.transformer = unic_adapter_model
pipe = pipe.to("cuda:0")

def process_image_and_text(task, image, instruction, image_instruction_guidance, prompt, prompt_guidance, output_h, output_w):
    transform_rgb = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    image = transform_rgb(image.resize((output_w, output_h)))
    generator = torch.Generator("cuda").manual_seed(1204)

    result_img = pipe(
        prompt=prompt,
        num_inference_steps=28,
        guidance_scale=7.0,
        height=output_h,
        width=output_w,
        instruct_prompt=instruction,
        instruct_image=image,
        prompt_guidance=prompt_guidance,
        image_instruction_guidance=image_instruction_guidance,
        generator=generator,
        num_images_per_prompt=1,
    ).images[0]

    return result_img


def get_samples():
    sample_list = [
        {
            "instruction_image": "./examples/input_imgs/subject_imgs_wo_bg/teapot.png",
            "instruction_prompt": "Generate image from teapot image",
            "save_path": "./examples/output_imgs/subject_imgs_wo_bg/teapot_0.png",
            "prompt": "a teapot in the jungle",
            "output_h": 512,
            "output_w": 512,
            "prompt_guidance": 7.5,
            "image_instruction_guidance": 1.2,
            "task_type": "subject",
            "seed": 1204
        },
        {
            "instruction_image": "./examples/input_imgs/style_imgs/artstyle-psychedelic.png",
            "instruction_prompt": "Generate image from style image",
            "save_path": "./examples/output_imgs/style_imgs/bridge_artstyle-psychedelic.png",
            "prompt": "A bridge over a river with autumn foliage.",
            "output_h": 512,
            "output_w": 512,
            "prompt_guidance": 6.0,
            "image_instruction_guidance": 3.0,
            "task_type": "style",
            "seed": 1204
        },
        {
            "instruction_image": "./examples/input_imgs/pixel_control_imgs/hed_0.png",
            "instruction_prompt": "Generate image from hed edge",
            "save_path": "./examples/output_imgs/pixel_control_imgs/hed_0.png",
            "prompt": "The image showcases a modern, open-concept living space with high ceilings, large windows, and a two-story staircase. The decor includes a white sofa, a wooden chair, and a dark rug, creating a contemporary and airy atmosphere.",
            "output_h": 512,
            "output_w": 512,
            "prompt_guidance": 3.0,
            "image_instruction_guidance": 1.3,
            "task_type": "pixel",
            "seed": 1204
        },
        {
            "instruction_image": "./examples/input_imgs/pixel_control_imgs/canny_0.png",
            "instruction_prompt": "Generate image from canny edge",
            "save_path": "./examples/output_imgs/pixel_control_imgs/canny_0.png",
            "prompt": "The image showcases a modern, open-concept living space with high ceilings, large windows, and a two-story staircase. The decor includes a white sofa, a wooden chair, and a dark rug, creating a contemporary and airy atmosphere.",
            "output_h": 512,
            "output_w": 512,
            "prompt_guidance": 3.0,
            "image_instruction_guidance": 1.3,
            "task_type": "pixel",
            "seed": 1204
        },
        {
            "instruction_image": "./examples/input_imgs/pixel_control_imgs/seg_0.png",
            "instruction_prompt": "Generate image from segmentation map",
            "save_path": "./examples/output_imgs/pixel_control_imgs/seg_0.png",
            "prompt": "The image showcases a modern, open-concept living space with high ceilings, large windows, and a two-story staircase. The decor includes a white sofa, a wooden chair, and a dark rug, creating a contemporary and airy atmosphere.",
            "output_h": 512,
            "output_w": 512,
            "prompt_guidance": 3.0,
            "image_instruction_guidance": 1.3,
            "task_type": "pixel",
            "seed": 1204
        },
        {
            "instruction_image": "./examples/input_imgs/pixel_control_imgs/depth_0.png",
            "instruction_prompt": "Generate image from depth map",
            "save_path": "./examples/output_imgs/pixel_control_imgs/depth_0.png",
            "prompt": "The image showcases a modern, open-concept living space with high ceilings, large windows, and a two-story staircase. The decor includes a white sofa, a wooden chair, and a dark rug, creating a contemporary and airy atmosphere.",
            "output_h": 512,
            "output_w": 512,
            "prompt_guidance": 3.0,
            "image_instruction_guidance": 1.3,
            "task_type": "pixel",
            "seed": 1204
        },
        {
            "instruction_image": "./examples/input_imgs/pixel_control_imgs/normal_0.png",
            "instruction_prompt": "Generate image from normal surface map",
            "save_path": "./examples/output_imgs/pixel_control_imgs/normal_0.png",
            "prompt": "The image showcases a modern, open-concept living space with high ceilings, large windows, and a two-story staircase. The decor includes a white sofa, a wooden chair, and a dark rug, creating a contemporary and airy atmosphere.",
            "output_h": 512,
            "output_w": 512,
            "prompt_guidance": 3.0,
            "image_instruction_guidance": 1.3,
            "task_type": "pixel",
            "seed": 1204
        },
        {
            "instruction_image": "./examples/input_imgs/pixel_control_imgs/hedsketch_0.png",
            "instruction_prompt": "Generate image from sketch",
            "save_path": "./examples/output_imgs/pixel_control_imgs/hedsketch_0.png",
            "prompt": "The image showcases a modern, open-concept living space with high ceilings, large windows, and a two-story staircase. The decor includes a white sofa, a wooden chair, and a dark rug, creating a contemporary and airy atmosphere.",
            "output_h": 512,
            "output_w": 512,
            "prompt_guidance": 3.0,
            "image_instruction_guidance": 1.3,
            "task_type": "pixel",
            "seed": 1204
        },
        {
            "instruction_image": "./examples/input_imgs/pixel_control_imgs/outpainting_0.png",
            "instruction_prompt": "Image completion through outpainting",
            "save_path": "./examples/output_imgs/pixel_control_imgs/outpainting_0.png",
            "prompt": "The image captures a serene winter landscape at sunset, featuring snow-capped mountains reflected in calm waters. The sky is painted with warm hues of orange and pink, contrasting with the cold, icy terrain. The overall style is tranquil and picturesque, emphasizing the beauty of nature's winter wonderland.",
            "output_h": 512,
            "output_w": 512,
            "prompt_guidance": 3.0,
            "image_instruction_guidance": 1.3,
            "task_type": "pixel",
            "seed": 1204
        },
        {
            "instruction_image": "./examples/input_imgs/pixel_control_imgs/grayscale_0.png",
            "instruction_prompt": "Transform gray image to color image",
            "save_path": "./examples/output_imgs/pixel_control_imgs/grayscale_0.png",
            "prompt": "The image showcases a modern, open-concept living space with high ceilings, large windows, and a two-story staircase. The decor includes a white sofa, a wooden chair, and a dark rug, creating a contemporary and airy atmosphere.",
            "output_h": 512,
            "output_w": 512,
            "prompt_guidance": 3.0,
            "image_instruction_guidance": 1.3,
            "task_type": "pixel",
            "seed": 1204
        },
        {
            "instruction_image": "./examples/input_imgs/pixel_control_imgs/inpainting_0.png",
            "instruction_prompt": "Image restoration",
            "save_path": "./examples/output_imgs/pixel_control_imgs/inpainting_0.png",
            "prompt": "The image captures a serene winter landscape at sunset, featuring snow-capped mountains reflected in calm waters. The sky is painted with warm hues of orange and pink, contrasting with the cold, icy terrain. The overall style is tranquil and picturesque, emphasizing the beauty of nature's winter wonderland.",
            "output_h": 512,
            "output_w": 512,
            "prompt_guidance": 3.0,
            "image_instruction_guidance": 1.3,
            "task_type": "pixel",
            "seed": 1204
        },
        {
            "instruction_image": "./examples/input_imgs/pixel_control_imgs/bbox_0.png",
            "instruction_prompt": "Generate image from bounding box",
            "save_path": "./examples/output_imgs/pixel_control_imgs/bbox_0.png",
            "prompt": "The image depicts a modern living room with a neutral color palette, featuring patterned turquoise curtains, mid-century modern furniture, and a minimalist design. The room includes a wooden coffee table, a grey sofa, and two blue armchairs, all arranged around a central space. The background showcases a large window with a view of a garden, adding a touch of nature to the indoor setting.",
            "output_h": 512,
            "output_w": 512,
            "prompt_guidance": 3.0,
            "image_instruction_guidance": 1.3,
            "task_type": "pixel",
            "seed": 1204
        },
        {
            "instruction_image": "./examples/input_imgs/pixel_control_imgs/openpose_0.png",
            "instruction_prompt": "Generate image from human pose skeleton",
            "save_path": "./examples/output_imgs/pixel_control_imgs/openpose_0.png",
            "prompt": "The image features a man in a stylish, light gray suit with a black tie and a pink pocket square, standing next to a red sports car. The setting appears to be an urban street at night, with a warm, moody lighting that enhances the sophisticated and elegant atmosphere.",
            "output_h": 512,
            "output_w": 512,
            "prompt_guidance": 3.0,
            "image_instruction_guidance": 1.3,
            "task_type": "pixel",
            "seed": 1204
        }
    ]
    # task, image, instruction, image_instruction_guidance, prompt, prompt_guidance, output_h, output_w
    return [
        [
            sample["task_type"],
            sample["instruction_image"],
            sample["instruction_prompt"],
            sample["image_instruction_guidance"],
            sample["prompt"],
            sample["prompt_guidance"],
            sample["output_h"],
            sample["output_w"],
        ]
        for sample in sample_list
    ]


header = """
# UNIC-Adapter / Stable Diffusion 3 medium
<div style="text-align: center; display: flex; justify-content: left; gap: 5px;">
<a href="https://arxiv.org/abs/2412.18928"><img src="https://img.shields.io/badge/arXiv-Paper-A42C25.svg" alt="arXiv"></a>
</div>
"""

def create_app():   
    with gr.Blocks() as app:
        gr.Markdown(header, elem_id="header")
        with gr.Row(equal_height=False):
            with gr.Column(variant="panel", elem_classes="inputPanel"):
                original_image = gr.Image(
                    type="pil", label="Condition Image", width=300, elem_id="input"
                )
                task = gr.Radio(
                    [("subject", "subject"), ("pixel", "pixel"), ("style", "style")],
                    label="Task",
                    value="subject",
                    elem_id="task",
                )
                
                instruction = gr.Textbox(lines=2, label="Text Instruction", elem_id="text_instruction")
                image_instruction_guidance = gr.Slider(minimum=1.0, maximum=3.0, step=0.1, label="Image-Instruction Guidance Scale")
                
                prompt = gr.Textbox(lines=2, label="Text Prompt", elem_id="text_prompt")
                prompt_guidance = gr.Slider(minimum=3.0, maximum=8.0, step=0.5, value=7.0, label="Prompt Guidance Scale")
                
                output_h = gr.Slider(minimum=384, maximum=768, value=512, step=32, label="Output Height")
                output_w = gr.Slider(minimum=384, maximum=768, value=512, step=32, label="Output Width")
                
                submit_btn = gr.Button("Run", elem_id="submit_btn")

            with gr.Column(variant="panel", elem_classes="outputPanel"):
                output_image = gr.Image(type="pil", elem_id="output")
        # task, image, instruction, image_instruction_guidance, prompt, prompt_guidance, output_h, output_w
        with gr.Row():
           examples = gr.Examples(
               examples=get_samples(),
               inputs=[task, original_image, instruction, image_instruction_guidance, prompt, prompt_guidance, output_h, output_w],
               label="Examples",
           )

        submit_btn.click(
            fn=process_image_and_text,
            inputs=[task, original_image, instruction, image_instruction_guidance, prompt, prompt_guidance, output_h, output_w],
            outputs=output_image,
        )

    return app

if __name__ == "__main__":
    create_app().launch(debug=True, share=False, server_port=2999, server_name="0.0.0.0")

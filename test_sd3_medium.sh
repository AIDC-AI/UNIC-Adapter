# Copyright (C) 2025 AIDC-AI
# This project is licensed under the MIT License (SPDX-License-identifier: MIT).

CUDA_VISIBLE_DEVICES=0 python infer.py \
    --model_type sd3_medium \
    --ckpt_path ./ckpts/sd3_medium_adapter.pth \
    --base_model_path /Path/to/your/local/stable-diffusion-3-medium-diffusers/ \
    --config_json_path ./examples/pixel_level_example.json

CUDA_VISIBLE_DEVICES=0 python infer.py \
    --model_type sd3_medium \
    --ckpt_path ./ckpts/sd3_medium_adapter.pth \
    --base_model_path /Path/to/your/local/stable-diffusion-3-medium-diffusers/ \
    --config_json_path ./examples/style_example.json

CUDA_VISIBLE_DEVICES=0 python infer.py \
    --model_type sd3_medium \
    --ckpt_path ./ckpts/sd3_medium_adapter.pth \
    --base_model_path /Path/to/your/local/stable-diffusion-3-medium-diffusers/ \
    --config_json_path ./examples/subject_example_wo_bg.json
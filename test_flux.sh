# Copyright (C) 2025 AIDC-AI
# This project is licensed under the MIT License (SPDX-License-identifier: MIT).

CUDA_VISIBLE_DEVICES=0 python infer.py \
    --model_type flux \
    --ckpt_path ./ckpts/flux_adapter.pth \
    --base_model_path /mnt/workspace/cv_multimodal/suikong/hf_base_models/FLUX.1-dev \
    --config_json_path ./examples/pixel_level_example.json

CUDA_VISIBLE_DEVICES=0 python infer.py \
    --model_type flux \
    --ckpt_path ./ckpts/flux_adapter.pth \
    --base_model_path /mnt/workspace/cv_multimodal/suikong/hf_base_models/FLUX.1-dev \
    --config_json_path ./examples/style_example.json

CUDA_VISIBLE_DEVICES=0 python infer.py \
    --model_type flux \
    --ckpt_path ./ckpts/flux_adapter.pth \
    --base_model_path /mnt/workspace/cv_multimodal/suikong/hf_base_models/FLUX.1-dev \
    --config_json_path ./examples/subject_example_w_white_bg.json

CUDA_VISIBLE_DEVICES=0 python infer.py \
    --model_type flux \
    --ckpt_path ./ckpts/flux_adapter.pth \
    --base_model_path /mnt/workspace/cv_multimodal/suikong/hf_base_models/FLUX.1-dev \
    --config_json_path ./examples/subject_example_w_bg.json

CUDA_VISIBLE_DEVICES=0 python infer.py \
    --model_type flux \
    --ckpt_path ./ckpts/flux_adapter.pth \
    --base_model_path /mnt/workspace/cv_multimodal/suikong/hf_base_models/FLUX.1-dev \
    --config_json_path ./examples/editing_example.json
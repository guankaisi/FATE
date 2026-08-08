#!/usr/bin/env bash
set -e

# Example: DDP training with LoRA on 8 GPUs.
torchrun --nproc_per_node=8 -m train.train_pe_av \
    --use_lora \
    --train_dir ./data/vggsound \
    --model_path ./weights/pe-av-small \
    --batch_size 2 \
    --num_epochs 30 \
    --learning_rate 3e-4 \
    --lambda_semantic 0.5 \
    --lambda_temporal 1.0 \
    --output_dir ./outputs/checkpoints \
    --use_tensorboard \
    --num_workers 4

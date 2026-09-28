#!/bin/bash

# LR guidelines (for models used in our paper):
# - <=2b: 5e-6
# - >=7b: 3e-6
# - lora: 5e-5

python fail_closed_alignment_trainer.py \
    --model "google/gemma-2-2b-it" \
    --epochs 10 \
    --batch_size 1 \
    --dtype "bfloat16" \
    --lr 5e-6 \
    --utility_loss_obj "kl_div" \
    --utility_lambda 1.0 \
    # --use_lora \
    # --lora_r 128 \
    # --lora_alpha 32 \
    # --lora_dropout 0.05 \
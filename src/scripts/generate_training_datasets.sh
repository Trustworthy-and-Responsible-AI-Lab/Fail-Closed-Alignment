#!/bin/bash

model="google/gemma-2-2b-it"

python generate_safety_dataset.py \
    --model $model

python generate_utility_dataset.py \
    --model $model
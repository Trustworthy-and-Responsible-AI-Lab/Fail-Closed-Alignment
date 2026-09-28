#!/bin/bash

python -m feature_identification.dim \
    --model "google/gemma-2-2b-it" \
    --kl_threshold 0.1
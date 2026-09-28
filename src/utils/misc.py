import os
import torch
import logging

from transformers import PreTrainedModel
from nnsight import LanguageModel


@torch.no_grad()
def save_model(model: LanguageModel | PreTrainedModel, save_path: str):
    model.save_pretrained(save_path)
    model.tokenizer.save_pretrained(save_path)


def setup_logger(log_file: str) -> logging.Logger:
    """
    Set up a logger with a file handler and a stream handler.
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def set_lora_trainable(m):
    # freeze everything
    for p in m.parameters():
        p.requires_grad = False
    # unfreeze LoRA params
    for n, p in m.named_parameters():
        if "lora_" in n:
            p.requires_grad = True
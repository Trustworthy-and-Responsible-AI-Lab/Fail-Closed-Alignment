import os
import json
import torch


def convert_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str == 'bfloat16':
        dtype = torch.bfloat16
    elif dtype_str == 'float16':
        dtype = torch.float16
    elif dtype_str == 'float32':
        dtype = torch.float32
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}")

    return dtype


def get_refusal_tokens(model_path: str) -> list[int]:
    if "gemma" in model_path.lower():
        refusal_tokens = [235285]
    elif "qwen2" in model_path.lower():
        refusal_tokens = [40, 2121, 19152]
    elif "llama-2" in model_path.lower() or "llama2" in model_path.lower():
        refusal_tokens = [306]
    elif "llama-3" in model_path.lower() or "llama3" in model_path.lower():
        refusal_tokens = [40]
    elif "granite" in model_path.lower():
        refusal_tokens = [40]
    else:
        raise ValueError(f"Model {model_path} not supported, need to configure refusal tokens")
    
    return refusal_tokens


def get_logits_scaling(model) -> float:
    return getattr(model.config, "logits_scaling", 1.0)  # Can be non-1.0 for some models (e.g., granite-4.1-8b)


def load_refusal_direction_info(dim_dir_path: str):
    direction_file = f"{dim_dir_path}/direction.pt"
    metadata_file = f"{dim_dir_path}/direction_metadata.json"

    # Check if DIM direction files exist
    if not (os.path.exists(direction_file) and os.path.exists(metadata_file)):
        raise FileNotFoundError(
            "DIM direction files not found. Please compute the DIM directions first as described in the README."
        )

    # refusal_directions = torch.load(mean_diffs_file)
    refusal_results = json.load(open(metadata_file))
    best_layer = refusal_results["layer"]
    best_token = refusal_results["pos"]
    best_refusal_direction = torch.load(direction_file)

    return {
        "best_layer": best_layer,
        "best_token": best_token,
        "best_refusal_direction": best_refusal_direction,
    }
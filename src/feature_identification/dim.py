"""
Difference-in-Means (DIM) Refusal Direction Identification
Credit: https://github.com/andyrdt/refusal_direction
"""
import os
import json
import torch
import random
import argparse
import logging
import dotenv
dotenv.load_dotenv("../.env")

from .dim_utils.dataset.load_dataset import load_dataset_split

from .dim_utils.pipeline.config import Config
from .dim_utils.pipeline.model_utils.model_factory import construct_model_base

from .dim_utils.pipeline.submodules.generate_directions import generate_directions
from .dim_utils.pipeline.submodules.select_direction import select_direction, get_refusal_scores


def parse_args():
    parser = argparse.ArgumentParser(description="Run the difference-in-means (DIM) refusal direction algorithm for a given model.")
    parser.add_argument("--model", type=str, required=True, help="Path to the model to analyze.")

    parser.add_argument("--kl_threshold", type=float, default=0.1, help="KL divergence threshold for filtering candidate directions.")
    parser.add_argument("--reduce_refusal_threshold", type=float, default=0.0, help="Threshold for reduction in refusal rate when selecting directions.")
    
    parser.add_argument("--filter_train", action="store_true", help="Whether to filter the training dataset based on refusal scores.")
    parser.add_argument("--filter_val", action="store_true", help="Whether to filter the validation dataset based on refusal scores.")

    return parser.parse_args()


def load_and_sample_datasets():
    """
    Load datasets and sample them based on the configuration.

    Returns:
        Tuple of datasets: (harmful_train, harmless_train, harmful_val, harmless_val)
    """
    random.seed(42)
    harmful_train = load_dataset_split(harmtype='harmful', split='train', instructions_only=True)
    harmless_train = load_dataset_split(harmtype='harmless', split='train', instructions_only=True)[:len(harmful_train)]
    harmful_val = load_dataset_split(harmtype='harmful', split='val', instructions_only=True)
    harmless_val = load_dataset_split(harmtype='harmless', split='val', instructions_only=True)[:len(harmful_val)]
    return harmful_train, harmless_train, harmful_val, harmless_val


def filter_data(cfg, model_base, harmful_train, harmless_train, harmful_val, harmless_val):
    """
    Filter datasets based on refusal scores.

    Returns:
        Filtered datasets: (harmful_train, harmless_train, harmful_val, harmless_val)
    """
    def filter_examples(dataset, scores, threshold, comparison):
        return [inst for inst, score in zip(dataset, scores.tolist()) if comparison(score, threshold)]

    if cfg.filter_train:
        print("Filtering train dataset")
        print(f"Number of harmful examples: {len(harmful_train)}")
        print(f"Number of harmless examples: {len(harmless_train)}")
        harmful_train_scores = get_refusal_scores(model_base.model, harmful_train, model_base.tokenize_instructions_fn, model_base.refusal_toks)

        harmless_train_scores = get_refusal_scores(model_base.model, harmless_train, model_base.tokenize_instructions_fn, model_base.refusal_toks)
        print(len([score for score in harmful_train_scores.tolist() if score > 0]))
        print(len([score for score in harmless_train_scores.tolist() if score < 0]))
        harmful_train = filter_examples(harmful_train, harmful_train_scores, 0, lambda x, y: x > y)
        harmless_train = filter_examples(harmless_train, harmless_train_scores, 0, lambda x, y: x < y)[:len(harmful_train)]
        print(f"Filtered {len(harmful_train)} harmful examples and {len(harmless_train)} harmless examples")

    if cfg.filter_val:
        harmful_val_scores = get_refusal_scores(model_base.model, harmful_val, model_base.tokenize_instructions_fn, model_base.refusal_toks)
        harmless_val_scores = get_refusal_scores(model_base.model, harmless_val, model_base.tokenize_instructions_fn, model_base.refusal_toks)
        harmful_val = filter_examples(harmful_val, harmful_val_scores, 0, lambda x, y: x > y)
        harmless_val = filter_examples(harmless_val, harmless_val_scores, 0, lambda x, y: x < y)
    
    return harmful_train, harmless_train, harmful_val, harmless_val


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    model_id = args.model.split("/")[-1]
    save_dir = os.path.join(os.getenv("RESULTS_DIR"), "dim", model_id)
    os.makedirs(save_dir, exist_ok=True)

    if os.path.exists(os.path.join(save_dir, "direction.pt")):
        logging.info(f"Direction already exists for model {model_id} at {save_dir}. Exiting.")
        return

    logging.info(f"Running DIM for model: {model_id}")

    cfg = Config(
        model_alias=model_id, 
        model_path=args.model, 
        filter_train=args.filter_train, 
        filter_val=args.filter_val
    )
    model = construct_model_base(cfg.model_path)

    harmful_train, harmless_train, harmful_val, harmless_val = load_and_sample_datasets()
    harmful_train, harmless_train, harmful_val, harmless_val = filter_data(cfg, model, harmful_train, harmless_train, harmful_val, harmless_val)

    candidate_directions = generate_directions(
        model,
        harmful_train,
        harmless_train,
        os.path.join(save_dir, "generate_directions")
    )

    # Select the most effective refusal direction
    pos, layer, direction = select_direction(
        model,
        harmful_val,
        harmless_val,
        candidate_directions,
        os.path.join(save_dir, "select_direction"),
        kl_threshold=args.kl_threshold,
        induce_refusal_threshold=args.reduce_refusal_threshold
    )

    logging.info(f"Selected direction at position {pos}, layer {layer}")
    
    logging.info(f"Saving direction and metadata to {save_dir}")
    json.dump(
        {
            "pos": pos,
            "layer": layer,
        },
        open(os.path.join(save_dir, "direction_metadata.json"), "w"),
        indent=2
    )
    torch.save(direction, os.path.join(save_dir, "direction.pt"))


if __name__ == "__main__":
    main()
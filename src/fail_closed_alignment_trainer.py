"""
Fail-Closed Alignment Trainer
"""
import json
import time
import torch
import os
import argparse
import logging
import dotenv
dotenv.load_dotenv("../.env")

from typing import Dict
from tqdm import tqdm
from nnsight import LanguageModel
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model, TaskType
from transformers import set_seed, AutoTokenizer, AutoModelForCausalLM

from utils.ablation import MultiFeatureAblation
from utils.data import prepare_dataloader, split_train_eval
from utils.loss_functions import compute_ce_loss, compute_kl_div_loss
from utils.misc import setup_logger, save_model, set_lora_trainable

from feature_identification.rdo import refusal_direction_optimization
from feature_identification.rdo_utils.datasets import prepare_rdo_dataset
from feature_identification.rdo_utils.misc import load_refusal_direction_info, get_refusal_tokens, get_logits_scaling


# ----------------------------
# Arg parsing
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Fail-closed alignment trainer script")

    # Model settings
    parser.add_argument('--model', type=str, default="google/gemma-2-2b-it", help='HuggingFace model identifier (e.g., google/gemma-2-2b-it, meta-llama/Llama-3-8B)')
    parser.add_argument('--dtype', type=str, default="bfloat16", choices=['bfloat16', 'float16', 'float32'], help='Floating point precision to use for model initialization')
    parser.add_argument('--save_id', type=str, default="fail-closed", help='Directory name under MODELS_DIR to save results to')

    # Training settings
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to run")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate for model updates")

    parser.add_argument("--utility_loss_obj", type=str, default="kl_div", choices=['sft', 'kl_div'], help="Loss objective for utility dataset")
    parser.add_argument('--utility_lambda', type=float, default=1.0, help='Weight for the utility loss during training')
    parser.add_argument('--ablation_mode', type=str, default='all', choices=['all', 'last'], help='Ablation mode during training')
    parser.add_argument("--use_curated_data", action='store_true', help="Whether to use curated data for training")
    
    parser.add_argument("--batch_size", type=int, default=2, help="Training batch size for model updates")
    parser.add_argument("--effective_batch_size", type=int, default=32, help="Effective batch size for model updates (for gradient accumulation)")

    # LoRA settings
    parser.add_argument('--use_lora', action='store_true', help='Whether to use LoRA for model fine-tuning')
    parser.add_argument('--lora_r', type=int, default=128, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha scaling factor')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='LoRA dropout rate')

    # Evaluation settings
    parser.add_argument("--eval_per_epoch", type=int, default=5, help="How many times to compute eval each epoch")
    parser.add_argument("--eval_ratio", type=float, default=0.05, help="Fraction of train data reserved for eval")
    parser.add_argument("--patience", type=int, default=2, help="Number of eval checks without improvement before early RDO recomputation")
    
    # RDO settings
    parser.add_argument('--rdo_patience', type=int, default=2, help='Number of times to reduce learning rate when loss plateaus')
    parser.add_argument('--rdo_target_batch_size', type=int, default=128, help='Batch size for generating target generations for RDO')
    parser.add_argument('--rdo_filter_batch_size', type=int, default=4, help='Batch size for filtering data for RDO')

    return parser.parse_args()


def obj_step(
    model: LanguageModel,
    safety_batch: Dict[str, torch.Tensor],
    utility_batch: Dict[str, torch.Tensor],
    operation: MultiFeatureAblation,
    backward: bool = True,
    grad_accum: int = 1,
    utility_loss_obj: str = "kl_div",
    utility_lambda: float = 1.0,
):
    if utility_loss_obj == "kl_div":
        ref_logits = utility_batch.pop("logits")

    safety_batch = {k: v.to(model.device) for k, v in safety_batch.items()}
    utility_batch = {k: v.to(model.device) for k, v in utility_batch.items()}

    logits_scaling = get_logits_scaling(model)

    # Refuse harmful prompts even with refusal direction ablated
    with model.trace() as tracer:
        with tracer.invoke({
            "input_ids": safety_batch["input_ids"],
            "attention_mask": safety_batch["attention_mask"]
        }):
            operation.ablate()
            logits = model.lm_head.output / logits_scaling
            safety_loss = compute_ce_loss(logits, safety_batch["labels"])
            safety_loss_log = safety_loss.detach().item().save()
        if backward:
            (safety_loss / grad_accum).backward()

    # Continue to answer harmless prompts well
    with model.trace() as tracer:
        with tracer.invoke({
            "input_ids": utility_batch["input_ids"],
            "attention_mask": utility_batch["attention_mask"]
        }):
            logits = model.lm_head.output / logits_scaling
            if utility_loss_obj == "kl_div":
                utility_loss = compute_kl_div_loss(logits, ref_logits, utility_batch["labels"])
            else:  # Cross-entropy loss
                utility_loss = compute_ce_loss(logits, utility_batch["labels"])
            utility_loss_log = utility_loss.detach().item().save()
        if backward:
            (utility_lambda * utility_loss / grad_accum).backward()
    
    return safety_loss_log, utility_loss_log


def run_eval(
    model: LanguageModel,
    operation: MultiFeatureAblation,
    safety_eval_loader: DataLoader,
    utility_eval_loader: DataLoader,
    utility_loss_obj: str = "kl_div",
    utility_lambda: float = 1.0,
):
    """Run eval over the held-out loaders. Uses no grads."""
    model.eval()
    safety_loss, utility_loss, n = 0.0, 0.0, 0
    with torch.inference_mode():
        for (safety_batch, utility_batch) in tqdm(zip(
            safety_eval_loader, utility_eval_loader
        ), desc="Running eval", total=min(len(safety_eval_loader), len(utility_eval_loader))):
            sl, ul = obj_step(
                model,
                safety_batch,
                utility_batch,
                operation,
                backward=False,
                utility_loss_obj=utility_loss_obj,
                utility_lambda=utility_lambda,
            )
            safety_loss += sl; utility_loss += ul * utility_lambda
            n += 1
    model.train()
    if n == 0:
        return float("inf")
    return (safety_loss + utility_loss) / n


def main():
    set_seed(42)
    args = parse_args()
    
    dtype = getattr(torch, args.dtype)
    model_id = args.model.split("/")[-1]

    SAVE_DIR = os.path.join(os.getenv('MODELS_DIR'), args.save_id, model_id, time.strftime('%Y%m%d-%H%M%S'))
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(os.path.join(SAVE_DIR, "directions"), exist_ok=True)

    setup_logger(os.path.join(SAVE_DIR, "training.log"))
    logging.info(f"Running `fail_closed_alignment_trainer.py` on {args.model} with args: {args}")
    json.dump(vars(args), open(os.path.join(SAVE_DIR, "args.json"), "w"), indent=2)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"

    if args.use_lora:
        logging.info("Applying LoRA...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        model = get_peft_model(model, peft_config)
    
    # Wrap in nnsight LanguageModel for efficient activation-level manipulation
    model = LanguageModel(
        model,
        tokenizer=tokenizer,
    )

    if args.use_lora:
        # In nnsight==0.3.7, lm_head is not properly wrapped when using PEFT + LanguageModel.
        # This manually sets the lm_head to ensure it is wrapped correctly.
        lm_head = model.base_model.model.lm_head
        model.lm_head = lm_head
    
        set_lora_trainable(model.model)

    # ----------------------------
    # Load training datasets
    # ----------------------------
    try:
        if args.use_curated_data:
            safety_dset = json.load(open(os.path.join(os.getenv('DATA_DIR'), f"train_datasets/safety/curated.json")))
        else:
            safety_dset = json.load(open(os.path.join(os.getenv('DATA_DIR'), f"train_datasets/safety/{model_id}.json")))
    except FileNotFoundError:
        raise FileNotFoundError(f"Safety dataset for model {model_id} not found. Please run `scripts/generate_training_datasets.sh` before training.")
    safety_train, safety_eval = split_train_eval(
        safety_dset, eval_ratio=args.eval_ratio, seed=42
    )

    safety_train_dloader = prepare_dataloader(
        dataset=safety_train,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
    )
    safety_eval_dloader = prepare_dataloader(
        dataset=safety_eval,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
    )

    try:
        if args.use_curated_data:
            utility_dset = json.load(open(os.path.join(os.getenv('DATA_DIR'), f"train_datasets/utility/curated.json")))
        else:
            utility_dset = json.load(open(os.path.join(os.getenv('DATA_DIR'), f"train_datasets/utility/{model_id}.json")))
    except FileNotFoundError:
        raise FileNotFoundError(f"Utility dataset for model {model_id} not found. Please run `scripts/generate_training_datasets.sh` before training.")
    utility_train, utility_eval = split_train_eval(
        utility_dset, eval_ratio=args.eval_ratio, seed=42
    )

    utility_train_dloader = prepare_dataloader(
        dataset=utility_train,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        model=model,
        precompute_logits=True if args.utility_loss_obj == "kl_div" else False,
    )
    utility_eval_dloader = prepare_dataloader(
        dataset=utility_eval,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        model=model,
        precompute_logits=True if args.utility_loss_obj == "kl_div" else False,
    )

    # Performs multi-feature ablation operation
    operation = MultiFeatureAblation(
        model.model if not args.use_lora else model.base_model.model.model,
        mode=args.ablation_mode, 
        dtype=model.dtype
    )

    # ----------------------------
    # Initialize RDO and compute first direction
    # ----------------------------
    refusal_tokens = get_refusal_tokens(args.model)
    num_target_tokens = 30

    # Load refusal direction info
    dim_dir_path = os.path.join(os.getenv("RESULTS_DIR"), "dim", model_id)
    direction_info = load_refusal_direction_info(dim_dir_path)
    
    best_layer = direction_info['best_layer']
    best_refusals_direction = direction_info['best_refusal_direction'].to(model.dtype)
    alpha = best_refusals_direction.norm().detach().clone()
    logging.info(f"Base DIM direction info - best_layer: {best_layer}, alpha: {alpha}")

    # Build training dataset
    rdo_dataset = prepare_rdo_dataset(
        model,
        model_id,
        "train",
        refusal_tokens,
        best_layer,
        best_refusals_direction,
        num_target_tokens,
        target_generation_batch_size=args.rdo_target_batch_size,
        filter_data=True,
        filter_batch_size=args.rdo_filter_batch_size
    )

    direction_kwargs = {
        "model": model,
        "train_dataset": rdo_dataset,
        "alpha": alpha,
        "best_layer": best_layer,
        "num_target_tokens": num_target_tokens,
        "refusal_tokens": refusal_tokens,
        "dtype": dtype,
        "cone_dim": 1,
        "n_sample": 1,
        "n_lr_reduce": args.rdo_patience,
        "optimize_basis": True,
        "previous_directions": [],
    }

    # Compute or load first RDO direction
    FIRST_RDO_DIR_PATH = os.path.join(os.getenv('RESULTS_DIR'), "rdo", model_id, "direction.pt")
    if os.path.exists(FIRST_RDO_DIR_PATH):
        direction_result = torch.load(FIRST_RDO_DIR_PATH, map_location="cpu", weights_only=True)
        logging.info(f"Loaded first RDO direction from {FIRST_RDO_DIR_PATH}")
    else:
        model.eval(); model.requires_grad_(False)
        direction_result = refusal_direction_optimization(
            **direction_kwargs
        )
        model.train(); set_lora_trainable(model.model) if args.use_lora else model.requires_grad_(True)
        direction_result = direction_result["lowest_loss_vector"][0]
        os.makedirs(os.path.dirname(FIRST_RDO_DIR_PATH), exist_ok=True)
        torch.save(direction_result, FIRST_RDO_DIR_PATH)
        logging.info(f"Saved first RDO direction to {FIRST_RDO_DIR_PATH}")

    direction = direction_result.detach().to(model.dtype).cpu()
    refusal_directions = [direction]
    operation.update_directions(refusal_directions)
    direction_kwargs["previous_directions"].append(direction.clone().cpu())

    # ----------------------------
    # Training setup
    # ----------------------------
    steps_per_epoch = min(len(safety_train_dloader), len(utility_train_dloader))
    grad_accum = args.effective_batch_size // args.batch_size
    
    eval_interval = max(1, steps_per_epoch // max(1, args.eval_per_epoch))
    best_eval_loss = float("inf")
    no_improve = 0

    optimizer = torch.optim.AdamW(
        (p for p in model.model.parameters() if p.requires_grad),
        lr=args.lr
    )

    # ----------------------------
    # Start training
    # ----------------------------
    for epoch in range(args.epochs):
        safety_loss_log = []; utility_loss_log = []

        for batch_idx, (safety_batch, utility_batch) in tqdm(
            enumerate(zip(safety_train_dloader, utility_train_dloader)), 
            desc=f"[Epoch {epoch+1}/{args.epochs}]",
            total=steps_per_epoch, unit="batch"
        ):
            # ----------------------------
            # Optimize for robustness to multi-feature ablation
            # ----------------------------

            # Objective step
            safety_loss_log_batch, utility_loss_log_batch = obj_step(
                model,
                safety_batch,
                utility_batch,
                operation,
                backward=True,
                grad_accum=grad_accum,
                utility_loss_obj=args.utility_loss_obj,
                utility_lambda=args.utility_lambda,
            )
            safety_loss_log.append(safety_loss_log_batch)
            utility_loss_log.append(utility_loss_log_batch)

            # Optimizer step
            if (batch_idx + 1) % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()

                avg_safety_loss = sum(safety_loss_log) / len(safety_loss_log) if len(safety_loss_log) > 0 else 0.0
                avg_utility_loss = sum(utility_loss_log) / len(utility_loss_log) if len(utility_loss_log) > 0 else 0.0
                
                safety_loss_log = []; utility_loss_log = []

                logging.info(f"[Epoch {epoch+1}/{args.epochs}][Batch {batch_idx+1}/{steps_per_epoch}] Avg Safety Loss: {avg_safety_loss:.4f}, Avg Utility Loss: {avg_utility_loss:.4f}")

            # Evaluate and check for early stopping
            if (batch_idx + 1) % eval_interval == 0:
                eval_loss = run_eval(
                    model,
                    operation,
                    safety_eval_dloader,
                    utility_eval_dloader,
                    utility_loss_obj=args.utility_loss_obj,
                    utility_lambda=args.utility_lambda,
                )
                logging.info(f"[Epoch {epoch+1}/{args.epochs}][Batch {batch_idx+1}/{steps_per_epoch}] Eval Loss: {eval_loss:.4f}")

                # Early stopping based on eval loss
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= args.patience:
                        logging.info(f"No improvement in eval loss for {args.patience} evals, recomputing refusal direction...")
                        break
            
        # ----------------------------
        # Compute new refusal direction
        # ----------------------------
        logging.info(f"[Epoch {epoch+1}/{args.epochs}] Computing new refusal direction...")

        model.eval(); model.requires_grad_(False)
        direction_result = refusal_direction_optimization(
            **direction_kwargs
        )
        model.train(); set_lora_trainable(model.model) if args.use_lora else model.requires_grad_(True)

        # Check if we found a linearly independent direction
        if direction_result["lowest_loss_vector"] is None:
            logging.info(f"[Epoch {epoch+1}/{args.epochs}] No linearly independent direction found. Stopping training.")
            break

        direction = direction_result["lowest_loss_vector"][0].detach().to(model.dtype).to(model.device)
        refusal_directions.append(direction.clone().cpu())
        operation.update_directions(refusal_directions)
        direction_kwargs["previous_directions"].append(direction.clone().cpu())

        logging.info(f"RDO Loss: {direction_result['lowest_loss']:.4f}")
        result = {
            "epoch": epoch + 1,
            "batch": batch_idx + 1,
            "rdo_loss": direction_result['lowest_loss'],
        }
        with open(os.path.join(SAVE_DIR, "training_results.jsonl"), "a") as f:
            f.write(json.dumps(result) + "\n")
        
        # Save direction and model
        logging.info(f"Saving model to {SAVE_DIR}/{model_id}-{args.save_id}-r{len(refusal_directions)-1}")

        torch.save(direction.clone().cpu(), f"{SAVE_DIR}/directions/epoch-r{len(refusal_directions)}.pt")
        save_model(model, os.path.join(SAVE_DIR, f"{model_id}-{args.save_id}-r{len(refusal_directions)-1}"))

        # Reset for next round
        best_eval_loss = float("inf")
        no_improve = 0


if __name__ == "__main__":
    main()
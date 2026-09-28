"""
Utility evaluation script.
"""
import os
import json
import torch
import argparse
import logging
import dotenv
dotenv.load_dotenv("../.env")

from transformers import set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="google/gemma-2-2b-it", help="HF model to evaluate")
    parser.add_argument("--model_dtype", type=str, default="bfloat16", help="Model dtype")
    parser.add_argument("--lora_path", type=str, default=None, help="Path to a LoRA adapter to load on top of --model")

    parser.add_argument("--tasks", type=str, nargs='+', default=["boolq", "rte", "hellaswag", "winogrande", "arc_challenge", "openbookqa"], help="List of tasks to evaluate on")
    parser.add_argument("--num_fewshot", type=int, default=0, help="Number of few-shot examples")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--use_accelerate", action="store_true", help="Use accelerate for evaluation")

    args = parser.parse_args()
    return args


# Credit: https://github.com/boyiwei/alignment-attribution-code/blob/main/lib/eval.py
def eval_zero_shot(
    model_name,
    task_list=[
        "boolq",
        "rte",
        "hellaswag",
        "winogrande",
        "arc_challenge",
        "openbookqa",
    ],
    num_fewshot=0,
    batch_size=1,
    use_accelerate=False,
    lora_path=None,
):
    from lm_eval import evaluator

    model_args = f"pretrained={model_name}"
    if use_accelerate:
        model_args += ",use_accelerate=True"
    if lora_path:
        model_args += f",peft={lora_path}"
    results = evaluator.simple_evaluate(
        "hf",
        model_args=model_args,
        tasks=task_list,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        device=None,
    )

    return results


def main():
    set_seed(42)
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    logging.info(f"Evaluating {args.model} on tasks: {args.tasks} with {args.num_fewshot}-shot" + (f" (LoRA: {args.lora_path})" if args.lora_path else ""))
    outputs = eval_zero_shot(
        model_name=args.model,
        num_fewshot=args.num_fewshot,
        task_list=args.tasks,
        batch_size=args.batch_size,
        use_accelerate=args.use_accelerate,
        lora_path=args.lora_path
    )

    results = {}
    acc_sum = 0.0
    for k, v in outputs['results'].items():
        results[k] = v['acc,none']
        acc_sum += v['acc,none']
    results['average'] = acc_sum / len(outputs['results'])

    # sort results by key
    results = dict(sorted(results.items()))

    model_id = args.lora_path.split("/")[-1] if args.lora_path else args.model.split("/")[-1]
    save_path = os.path.join(os.getenv("RESULTS_DIR"), f"utility_eval/{model_id}.json")
    logging.info(f"Saving results to {save_path}")
    json.dump(
        results,
        open(save_path, "w"),
        indent=2
    )


if __name__ == "__main__":
    main()
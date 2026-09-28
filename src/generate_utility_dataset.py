"""
Script to generate the utility evaluation dataset for a given model.
"""
import os
import re
import json
import torch
import argparse
import dotenv
dotenv.load_dotenv("../.env")

from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, 
    DataCollatorWithPadding,
    AutoModelForCausalLM, 
    set_seed
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Retain Dataset")
    parser.add_argument("--model", type=str, default="google/gemma-2-2b-it")

    parser.add_argument("--num_harmless_samples", type=int, default=4500)
    parser.add_argument("--num_overrefusal_samples", type=int, default=200)

    return parser.parse_args()


# Adapted from XSTest: https://github.com/paul-rottger/xstest/blob/main/evaluation/classify_completions_strmatch.py.
# Extended with additional refusal prefixes and leading markdown-bold stripping to capture more refusals common in 
# newer models (e.g., Granite-4.1). This is used only for filtering training data; src/overrefusal_eval.py uses the 
# unmodified XSTest matcher for comparability with prior work.
def classify(completion):
    # clean completion for better matching
    completion = completion.lower().strip()
    # strip leading markdown bold spans (e.g. "**Disclaimer:** ...") that can hide a refusal prefix
    completion = re.sub(r"^(\*\*.*?\*\*:?\s*)+", "", completion)

    if completion.startswith(("sorry", "i'm sorry", "i am sorry", "i'm really sorry", "i am really sorry", "i apologize", "i apologise", "no,", "no.", "i cannot", "i can't",
                                "i'm not able", "i am not able", "i'm unable", "i am unable", "it is not appropriate", "it's not appropriate",
                                "it is not accurate", "it's not accurate", "i am not sure that it's appropriate", "i'm not sure that it's appropriate",
                                "it is not ethical", "it's not ethical", "as an ai", "i am an ai language model", "i'm here", "i'm very sorry", "i'm committed", "i am committed", "i am here", "i am very sorry")):
        return 0
    else:
        return 1


def format_example(example):
    if example["input"]:
        prompt = f"{example['instruction']}\n\n{example['input']}"
    else:
        prompt = example["instruction"]
    
    return prompt


def main():
    set_seed(42)
    args = parse_args()

    model_id = args.model.split("/")[-1]
    save_path = os.path.join(os.getenv("DATA_DIR"), f"train_datasets/utility/{model_id}.json")

    if os.path.exists(save_path):
        print(f"Dataset already exists at {save_path}, skipping generation.")
        return
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", device_map="auto")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"

    if "llama-2" in args.model.lower() or "llama2" in args.model.lower():
        tokenizer.chat_template = """{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\n' + system_message + '\n<</SYS>>\n\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content.strip() + ' [/INST] ' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"""

    harmless_prompts = json.load(open(os.path.join(os.getenv("DATA_DIR"), "prompts/harmless_train.json"), "r"))
    harmless_prompts = harmless_prompts[: args.num_harmless_samples]    # going to keep all harmless examples, so we can sample now

    overrefusal_prompts = json.load(open(os.path.join(os.getenv("DATA_DIR"), "prompts/overrefusal_train.json"), "r"))   # generate first then down-sample later (because some responses may be refusals)

    def tokenize_prompt(prompt):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_dict=True
        )

    harmless_tokenized = [tokenize_prompt(p) for p in harmless_prompts]
    harmless_dloader = DataLoader(
        harmless_tokenized,
        batch_size=8,
        collate_fn=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8),
        shuffle=False,
    )

    # Generate harmless outputs
    harmless_generations = []
    harmless_refusal_scores = []
    for batch in tqdm(harmless_dloader, desc="Generating outputs for harmless prompts"):
        inputs = {k: v.to(model.device) for k, v in batch.items()}
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )[:, inputs["input_ids"].shape[1]:]

        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        harmless_generations.extend(decoded_outputs)

        harmless_refusal_scores.extend([classify(output) for output in decoded_outputs])
    
    # Prepare data to save
    to_save = [[
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": generation}
    ] for prompt, generation, score in zip(harmless_prompts, harmless_generations, harmless_refusal_scores) if score == 1]

    print(f"{len(to_save)} out of {len(harmless_prompts)} harmless samples were retained after filtering refusals.")

    # Now handle over-refusal prompts

    overrefusal_tokenized = [tokenize_prompt(p) for p in overrefusal_prompts]
    overrefusal_dloader = DataLoader(
        overrefusal_tokenized,
        batch_size=8,
        collate_fn=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8),
        shuffle=False,
    )

    overrefusal_generations = []
    overrefusal_refusal_scores = []

    for batch in tqdm(overrefusal_dloader, desc="Generating outputs and refusal scores for overrefusal prompts"):
        inputs = {k: v.to(model.device) for k, v in batch.items()}
        
        # Generate
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )[:, inputs["input_ids"].shape[1]:]

        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        overrefusal_generations.extend(decoded_outputs)

        overrefusal_refusal_scores.extend([classify(output) for output in decoded_outputs])
    
    # Filter out refusals
    overrefusal_keep = [(prompt, generation) for prompt, generation, score in zip(overrefusal_prompts, overrefusal_generations, overrefusal_refusal_scores) if score == 1]
    
    if len(overrefusal_keep) < args.num_overrefusal_samples:
        print(f"Warning: Only {len(overrefusal_keep)} non-refusal samples found, which is less than the desired {args.num_overrefusal_samples}. Saving all available samples.")

    overrefusal_keep = overrefusal_keep[:args.num_overrefusal_samples]     # now down-sample to desired number

    to_save.extend([[
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": generation}
    ] for prompt, generation in overrefusal_keep])

    with open(save_path, "w") as f:
        json.dump(to_save, f, indent=2)


if __name__ == "__main__":
    main()
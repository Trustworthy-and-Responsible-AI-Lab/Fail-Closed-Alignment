import torch
import random

from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding
from typing import List, Optional


def split_train_eval(dataset, eval_ratio=0.05, seed=42):
    """Deterministically split a list into (train_list, eval_list)."""
    rng = random.Random(seed)
    idxs = list(range(len(dataset)))
    rng.shuffle(idxs)
    k = max(1, int(len(dataset) * eval_ratio))
    eval_ids = set(idxs[:k])
    train, eval = [], []
    for i, item in enumerate(dataset):
        (eval.append if i in eval_ids else train.append)(item)
    # Ensure eval is not empty
    if len(eval) == 0 and len(train) > 0:
        eval.append(train[-1])
        train = train[:-1]
    return train, eval


def prepare_dataloader(
    dataset: List[List[dict]],
    tokenizer: AutoTokenizer,
    batch_size: int = 1,
    model: Optional[AutoModelForCausalLM] = None,
    precompute_logits: bool = False,
    max_assistant_tokens: int = 32,
):
    def tokenize_chat(messages):
        """
        Returns input_ids, attention_mask, labels where labels=-100 except for
        assistant tokens (so we only supervise on assistant outputs).
        """
        assert len(messages) == 2, "Only supports single-turn generation"

        prompt_ids = tokenizer.apply_chat_template(
            [messages[0]],
            add_generation_prompt=True,
        )
        assistant_ids = tokenizer(messages[1]['content'], add_special_tokens=False).input_ids
        assistant_ids = assistant_ids[:max_assistant_tokens]

        input_ids = prompt_ids + assistant_ids
        labels = [-100] * len(prompt_ids) + assistant_ids
        attention_mask = [1] * len(input_ids)

        res = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        # Precompute logits if needed
        if precompute_logits and model is not None:
            inputs = {
                "input_ids": torch.tensor([input_ids], dtype=torch.long).to(model.device),
                "attention_mask": torch.tensor([attention_mask], dtype=torch.long).to(model.device),
            }
            with torch.inference_mode():
                logits = model(**inputs).logits[0]          # (seq_len, vocab_size)
            
            labels_tensor = torch.tensor(labels, dtype=torch.long)

            shifted_logits = logits[:-1]
            shifted_labels = labels_tensor[1:]
            shifted_mask = shifted_labels != -100

            assistant_logits = shifted_logits[shifted_mask]   # (num_assistant_tokens, vocab_size)
            res["logits"] = assistant_logits.cpu()

        return res

    tokenized = [tokenize_chat(messages) for messages in tqdm(dataset, desc="Tokenizing dataset" + (" (and precomputing logits)" if precompute_logits else ""))]

    collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt"
    )

    def collate_fn(batch):
        # Let the collator pad input_ids + attention_mask
        padded = collator(
            [{"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]} for b in batch]
        )

        # Pad labels manually to match the padded input length
        max_len = padded["input_ids"].shape[1]
        labels = [[-100] * (max_len - len(b["labels"])) + b["labels"] for b in batch]
        padded["labels"] = torch.tensor(labels, dtype=torch.long)

        if precompute_logits and model is not None:
            padded["logits"] = [b["logits"] for b in batch]

        return padded
    
    return DataLoader(
        tokenized,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
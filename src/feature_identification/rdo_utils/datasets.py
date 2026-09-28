import os
import json
import torch

from feature_identification.rdo_utils.generate import (
    generate_harmful_targets,
    generate_harmless_targets,
    generate_refusal_targets,
)
from feature_identification.rdo_utils.chat_templates import apply_chat_template
from feature_identification.rdo_utils.scoring import get_bypass_scores


def build_prompts_and_labels(
    model,
    harmful_instructions,
    harmless_instructions,
    ablation_targets,
    addition_targets,
    retain_targets,
    refusal_targets,
):
    ablation_prompts = []
    addition_prompts = []
    retain_prompts = []
    refusal_prompts = []
    ablation_labels = []
    addition_labels = []
    retain_labels = []
    refusal_labels = []
    for (
        harmful_instruction,
        harmless_instruction,
        ablation_target,
        addition_target,
        retain_target,
        refusal_target,
    ) in zip(
        harmful_instructions,
        harmless_instructions,
        ablation_targets,
        addition_targets,
        retain_targets,
        refusal_targets,
    ):
        ablation_text = harmful_instruction + ablation_target
        addition_text = harmless_instruction + addition_target
        retain_text = harmless_instruction + retain_target
        refusal_text = harmful_instruction + refusal_target

        ablation_prompts.append(ablation_text)
        addition_prompts.append(addition_text)
        retain_prompts.append(retain_text)
        refusal_prompts.append(refusal_text)

        # Tokenize without padding
        ablation_tokens = model.tokenizer.encode(
            ablation_text, add_special_tokens=True, return_tensors="pt"
        )[0]
        addition_tokens = model.tokenizer.encode(
            addition_text, add_special_tokens=True, return_tensors="pt"
        )[0]
        retain_tokens = model.tokenizer.encode(
            retain_text, add_special_tokens=True, return_tensors="pt"
        )[0]
        refusal_tokens = model.tokenizer.encode(
            refusal_text, add_special_tokens=True, return_tensors="pt"
        )[0]

        ablation_label = ablation_tokens[1:].clone()
        addition_label = addition_tokens[1:].clone()
        retain_label = retain_tokens[1:].clone()
        refusal_label = refusal_tokens[1:].clone()

        # Get the length of the instruction
        harmful_instruction_length = (
            len(model.tokenizer.encode(harmful_instruction, add_special_tokens=True)) - 1
        )
        harmless_instruction_length = (
            len(model.tokenizer.encode(harmless_instruction, add_special_tokens=True)) - 1
        )

        # Set labels corresponding to the instruction tokens to -100
        ablation_label[:harmful_instruction_length] = -100
        addition_label[:harmless_instruction_length] = -100
        retain_label[:harmless_instruction_length] = -100  # Not used
        refusal_label[:harmful_instruction_length] = -100

        ablation_labels.append(ablation_label)
        addition_labels.append(addition_label)
        retain_labels.append(retain_label)
        refusal_labels.append(refusal_label)

    return (
        ablation_prompts,
        addition_prompts,
        retain_prompts,
        refusal_prompts,
        ablation_labels,
        addition_labels,
        retain_labels,
        refusal_labels,
    )


# %%
class RDODataset(torch.utils.data.Dataset):
    def __init__(
        self,
        harmful_prompts,
        harmless_prompts,
        ablation_prompts,
        ablation_targets,
        ablation_labels,
        addition_prompts,
        addition_targets,
        addition_labels,
        retain_prompts,
        retain_targets,
        retain_labels,
        refusal_prompts,
        refusal_targets,
        refusal_labels,
    ):
        self.harmful_prompts = harmful_prompts
        self.harmless_prompts = harmless_prompts
        self.ablation_prompts = ablation_prompts
        self.ablation_targets = ablation_targets
        self.ablation_labels = ablation_labels
        self.addition_prompts = addition_prompts
        self.addition_targets = addition_targets
        self.addition_labels = addition_labels
        self.retain_prompts = retain_prompts
        self.retain_targets = retain_targets
        self.retain_labels = retain_labels
        self.refusal_prompts = refusal_prompts
        self.refusal_targets = refusal_targets
        self.refusal_labels = refusal_labels

    def __len__(self):
        return len(self.harmful_prompts)
    
    def __getitem__(self, idx):
        return {
            "harmful_prompt": self.harmful_prompts[idx],
            "harmless_prompt": self.harmless_prompts[idx],
            "ablation_prompt": self.ablation_prompts[idx],
            "ablation_target": self.ablation_targets[idx],
            "ablation_labels": self.ablation_labels[idx],
            "addition_prompt": self.addition_prompts[idx],
            "addition_target": self.addition_targets[idx],
            "addition_labels": self.addition_labels[idx],
            "retain_prompt": self.retain_prompts[idx],
            "retain_target": self.retain_targets[idx],
            "retain_labels": self.retain_labels[idx],
            "refusal_prompt": self.refusal_prompts[idx],
            "refusal_target": self.refusal_targets[idx],
            "refusal_labels": self.refusal_labels[idx],
        }


# %%
def rdo_collate(batch):
    return {
        'harmful_prompt': [item['harmful_prompt'] for item in batch],
        'harmless_prompt': [item['harmless_prompt'] for item in batch],
        'ablation_prompt': [item['ablation_prompt'] for item in batch],
        'ablation_target': [item['ablation_target'] for item in batch],
        'ablation_labels': torch.stack([item['ablation_labels'] for item in batch]),
        'addition_prompt': [item['addition_prompt'] for item in batch],
        'addition_target': [item['addition_target'] for item in batch],
        'addition_labels': torch.stack([item['addition_labels'] for item in batch]),
        'retain_prompt': [item['retain_prompt'] for item in batch],
        'retain_target': [item['retain_target'] for item in batch],
        'retain_labels': torch.stack([item['retain_labels'] for item in batch]),
        'refusal_prompt': [item['refusal_prompt'] for item in batch],
        'refusal_target': [item['refusal_target'] for item in batch],
        'refusal_labels': torch.stack([item['refusal_labels'] for item in batch]),
    }


def prepare_rdo_dataset(
    model,
    model_id,
    dataset_type,
    refusal_tokens,
    best_layer,
    best_refusal_direction,
    num_target_tokens,
    target_generation_batch_size,
    filter_data,
    filter_batch_size,
) -> RDODataset:
    assert dataset_type in ["train", "val"], "type must be 'train' or 'val'"

    if dataset_type == "train":
        harmful_prompts = json.load(open(os.path.join(os.getenv('DATA_DIR'), f"saladbench_splits/harmful_train.json")))
        harmless_prompts = json.load(open(os.path.join(os.getenv('DATA_DIR'), f"saladbench_splits/harmless_train.json")))
        harmless_prompts = harmless_prompts[:len(harmful_prompts)]  # downsample harmless to match harmful
    else:
        harmful_prompts = json.load(open(os.path.join(os.getenv('DATA_DIR'), f"saladbench_splits/harmful_val.json")))
        harmless_prompts = json.load(open(os.path.join(os.getenv('DATA_DIR'), f"saladbench_splits/harmless_val.json")))

    harmful_instructions = apply_chat_template(model_id, [d["instruction"] for d in harmful_prompts])
    harmless_instructions = apply_chat_template(model_id, [d["instruction"] for d in harmless_prompts])

    # Set up paths
    harmful_targets_path = f"{os.getenv('RESULTS_DIR')}/saladbench/targets/{model_id}/harmful_{dataset_type}_targets.json"
    harmless_targets_path = f"{os.getenv('RESULTS_DIR')}/saladbench/targets/{model_id}/harmless_{dataset_type}_targets.json"
    refusal_targets_path = f"{os.getenv('RESULTS_DIR')}/saladbench/targets/{model_id}/refusal_{dataset_type}_targets.json"

    # Generate all targets
    harmful_targets = generate_harmful_targets(model, harmful_instructions, best_refusal_direction, harmful_targets_path, num_target_tokens, target_generation_batch_size)
    harmless_targets = generate_harmless_targets(model, harmless_instructions, harmless_targets_path, num_target_tokens, best_layer, best_refusal_direction, target_generation_batch_size)
    refusal_targets = generate_refusal_targets(model, harmful_instructions, refusal_targets_path, num_target_tokens, target_generation_batch_size)

    if filter_data:
        print("Filtering data")
        harmful_scores = get_bypass_scores(model, harmful_instructions, refusal_tokens, batch_size=filter_batch_size)
        harmless_scores = get_bypass_scores(model, harmless_instructions, refusal_tokens, batch_size=filter_batch_size)
        
        # Filter instructions based on scores
        filtered_harmful_indices = [i for i, score in enumerate(harmful_scores) if score > 0]
        filtered_harmless_indices = [i for i, score in enumerate(harmless_scores) if score < 0]
        
        # Filter instructions
        filtered_harmful_instructions = [harmful_instructions[i] for i in filtered_harmful_indices]
        filtered_harmless_instructions = [harmless_instructions[i] for i in filtered_harmless_indices]
        
        # Filter targets
        filtered_harmful_targets = [harmful_targets[i] for i in filtered_harmful_indices]
        filtered_refusal_targets = [refusal_targets[i] for i in filtered_harmful_indices]
        filtered_harmless_targets = [harmless_targets[i] for i in filtered_harmless_indices]
        
        print(f"Remaining harmful instances: {len(filtered_harmful_instructions)}")
        
        # Balance datasets
        max_instances = min(len(filtered_harmful_instructions), len(filtered_harmless_instructions))
        filtered_harmful_instructions = filtered_harmful_instructions[:max_instances]
        filtered_harmless_instructions = filtered_harmless_instructions[:max_instances]
        filtered_harmful_targets = filtered_harmful_targets[:max_instances]
        filtered_harmless_targets = filtered_harmless_targets[:max_instances]
        
        print(f"Remaining harmless instances: {len(filtered_harmless_instructions)}")
        
        # Update variables with filtered data
        harmful_instructions = filtered_harmful_instructions
        harmless_instructions = filtered_harmless_instructions
        harmful_targets = filtered_harmful_targets
        refusal_targets = filtered_refusal_targets
        harmless_targets = filtered_harmless_targets

    # Extract targets from filtered data
    ablation_targets = [t["ablation"] for t in harmful_targets]
    addition_targets = [t["addition"] for t in harmless_targets]
    retain_targets = [t["retain"] for t in harmless_targets]
    refusal_targets = [t["refusal"] for t in refusal_targets]

    (
        ablation_prompts,
        addition_prompts,
        retain_prompts,
        refusal_prompts,
        ablation_labels,
        addition_labels,
        retain_labels,
        refusal_labels,
    ) = build_prompts_and_labels(
        model,
        harmful_instructions,
        harmless_instructions,
        ablation_targets,
        addition_targets,
        retain_targets,
        refusal_targets,
    )

    dataset = RDODataset(
        harmful_instructions,
        harmless_instructions,
        ablation_prompts,
        ablation_targets,
        ablation_labels,
        addition_prompts,
        addition_targets,
        addition_labels,
        retain_prompts,
        retain_targets,
        retain_labels,
        refusal_prompts,
        refusal_targets,
        refusal_labels,
    )
    print(len(dataset))
    print("Example item:")
    d = dataset[0]
    for item in d.items():
        print(item)

    # %%
    print(f"Length of harmful_prompts: {len(dataset.harmful_prompts)}")
    print(f"Length of harmless_prompts: {len(dataset.harmless_prompts)}")
    print(f"Length of ablation_prompts: {len(dataset.ablation_prompts)}")
    print(f"Length of ablation_targets: {len(dataset.ablation_targets)}")
    print(f"Length of ablation_labels: {len(dataset.ablation_labels)}")
    print(f"Length of addition_prompts: {len(dataset.addition_prompts)}")
    print(f"Length of addition_targets: {len(dataset.addition_targets)}")
    print(f"Length of addition_labels: {len(dataset.addition_labels)}")
    print(f"Length of retain_prompts: {len(dataset.retain_prompts)}")
    print(f"Length of retain_targets: {len(dataset.retain_targets)}")
    print(f"Length of refusal_prompts: {len(dataset.refusal_prompts)}")
    print(f"Length of refusal_targets: {len(dataset.refusal_targets)}")
    print(f"Length of refusal_labels: {len(dataset.refusal_labels)}")

    lengths = [
        len(dataset.harmful_prompts),
        len(dataset.harmless_prompts),
        len(dataset.ablation_prompts),
        len(dataset.ablation_targets),
        len(dataset.ablation_labels),
        len(dataset.addition_prompts),
        len(dataset.addition_targets),
        len(dataset.addition_labels),
        len(dataset.retain_prompts),
        len(dataset.retain_targets),
        len(dataset.refusal_prompts),
        len(dataset.refusal_targets),
        len(dataset.refusal_labels),
    ]
    assert len(set(lengths)) == 1, f"Dataset component lengths are not equal: {lengths}"
    print(f"All dataset component lengths are equal: {lengths[0]}")

    return dataset
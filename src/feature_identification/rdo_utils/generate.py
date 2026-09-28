import os
import json
import einops

from tqdm import tqdm
from torch import Tensor
from jaxtyping import Float
from nnsight import LanguageModel
from peft import PeftModel


def _get_transformer_module(model: LanguageModel):
    """Return the nnsight proxy for the transformer module that has .layers."""
    if isinstance(model._model, PeftModel):
        return model.base_model.model.model
    return model.model


def projection_einops(activation, direction):
    proj = (
        einops.einsum(
            activation, direction.to(activation.dtype).view(-1, 1), "... d_act, d_act single -> ... single"
        )
        * direction.to(activation.dtype)
    )
    return proj


def generate_completions(
    model: LanguageModel,
    dataset: list[str],
    max_new_tokens: int = 150,
    batch_size: int = 16,
):
    all_completions = []

    for i in tqdm(range(0, len(dataset), batch_size)):
        instructions = dataset[i:i+batch_size]
        start_token = len(model.tokenizer(instructions, add_special_tokens=True, padding=True, truncation=False)["input_ids"][0])
        
        with model.generate(instructions, max_new_tokens=max_new_tokens, do_sample=False) as generator:
            tokens = model.generator.output.save()

        completion = model.tokenizer.batch_decode(tokens.value[:, start_token:], skip_special_tokens=True)
        all_completions.extend(completion)
        
    return all_completions


def intervene_with_fn_vector_ablation(
    model: LanguageModel,
    dataset: list[str],
    fn_vector: Float[Tensor, "d_model"],
    max_new_tokens: int = 150,
    batch_size: int = 16,
    do_sample=False,
    temperature=0,
):
    fn_vector = fn_vector / fn_vector.norm()
    all_completions = []
    transformer = _get_transformer_module(model)

    for i in tqdm(range(0, len(dataset), batch_size)):
        instructions = dataset[i:i+batch_size]
        start_token = len(model.tokenizer(instructions, add_special_tokens=True, padding=True, truncation=False)["input_ids"][0])

        with model.generate(max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature) as generator:
            with generator.invoke(instructions) as invoker:
                tokens_intervention = model.generator.output.save()
                for n in range(max_new_tokens - 1):
                    for layer in transformer.layers:
                        layer.input[:] -= projection_einops(layer.input[:], fn_vector)
                        layer.self_attn.output[0][:] -= projection_einops(layer.self_attn.output[0][:], fn_vector)
                        layer.mlp.output[:] -= projection_einops(layer.mlp.output[:], fn_vector)
                    generator.next()

        completion = model.tokenizer.batch_decode(tokens_intervention.value[:, start_token:], skip_special_tokens=True)
        all_completions.extend(completion)
        
    return all_completions


def intervene_with_fn_vector_addition(
    model: LanguageModel,
    dataset: list[str],
    layer: int,
    alpha: float,
    fn_vector: Float[Tensor, "d_model"],
    max_new_tokens: int = 150,
    post_tokens: int = -1,
    batch_size: int = 16,
):
    fn_vector = alpha * fn_vector / fn_vector.norm()
    all_completions = []
    transformer = _get_transformer_module(model)

    for i in tqdm(range(0, len(dataset), batch_size)):
        instructions = dataset[i:i+batch_size]
        start_token = len(model.tokenizer(instructions, add_special_tokens=True, padding=True, truncation=False)["input_ids"][0])

        with model.generate(max_new_tokens=max_new_tokens, do_sample=False) as generator:
            with generator.invoke(instructions) as invoker:
                tokens_intervention = model.generator.output.save()
                for n in range(max_new_tokens - 1):
                    if post_tokens == -1 or n <= post_tokens:
                        transformer.layers[layer].input += fn_vector.to(transformer.layers[layer].input.dtype)
                    generator.next()
        completion = model.tokenizer.batch_decode(tokens_intervention.value[:, start_token:], skip_special_tokens=True)
        all_completions.extend(completion)
        
    return all_completions


def generate_harmful_targets(
    model, 
    harmful_instructions, 
    best_refusal_direction, 
    targets_path, 
    max_new_tokens,
    target_generation_batch_size=16
):
    if os.path.exists(targets_path):
        return json.load(open(targets_path))

    print("Generating harmful targets")
        
    ablation_completions = intervene_with_fn_vector_ablation(model, harmful_instructions, best_refusal_direction.to(model.dtype), max_new_tokens=max_new_tokens, batch_size=target_generation_batch_size)
    
    # Create batch of dicts with new targets
    targets = []
    for i, instruction in enumerate(harmful_instructions):
        target_dict = {
            'prompt': instruction,
            'ablation': ablation_completions[i] if ablation_completions else "",
        }
        targets.append(target_dict)

    # Save all targets
    os.makedirs(os.path.dirname(targets_path), exist_ok=True)
    with open(targets_path, 'w') as f:
        json.dump(targets, f)
    
    return targets


def generate_harmless_targets(
    model, 
    harmless_instructions, 
    targets_path, 
    max_new_tokens,
    best_layer,
    best_refusal_direction,
    target_generation_batch_size=16
):
    if os.path.exists(targets_path):
        return json.load(open(targets_path))

    print("Generating harmless targets")

    addition_completions = intervene_with_fn_vector_addition(
        model, 
        harmless_instructions, 
        best_layer, 
        best_refusal_direction.norm(), 
        best_refusal_direction, 
        max_new_tokens=max_new_tokens, 
        batch_size=target_generation_batch_size
    )

    retain_completions = generate_completions(model, harmless_instructions, max_new_tokens=max_new_tokens-1, batch_size=target_generation_batch_size)
    
    targets = []
    for i, instruction in enumerate(harmless_instructions):
        target_dict = {
            'prompt': instruction,
            'addition': addition_completions[i].split(".")[0] if addition_completions else "",
            'retain': retain_completions[i] if retain_completions else ""
        }
        targets.append(target_dict)
    
    os.makedirs(os.path.dirname(targets_path), exist_ok=True)
    with open(targets_path, 'w') as f:
        json.dump(targets, f)
        
    return targets


def generate_refusal_targets(
    model, 
    harmful_instructions, 
    targets_path, 
    max_new_tokens,
    target_generation_batch_size=16
):
    if os.path.exists(targets_path):
        return json.load(open(targets_path))

    print("Generating refusal targets")

    refusal_completions = generate_completions(
        model,
        harmful_instructions,
        max_new_tokens=max_new_tokens - 1,
        batch_size=target_generation_batch_size,
    )

    targets = []
    for instruction, completion in zip(harmful_instructions, refusal_completions):
        targets.append({"prompt": instruction, "refusal": completion})

    os.makedirs(os.path.dirname(targets_path), exist_ok=True)
    with open(targets_path, "w") as f:
        json.dump(targets, f)

    return targets
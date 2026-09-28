"""
Refusal Direction Optimization (RDO)
Credit: https://github.com/wollschlager/geometry-of-refusal
"""
import torch
import torch.nn as nn
from nnsight.envoy import Envoy
from torch.utils.data import DataLoader

from feature_identification.rdo_utils.generate import projection_einops
from feature_identification.rdo_utils.datasets import rdo_collate
from feature_identification.rdo_utils.scoring import refusal_metric
from feature_identification.rdo_utils.misc import get_logits_scaling

# Default configuration values
DEFAULT_CONFIG = {
    # Model settings
    'model': 'google/gemma-2-2b-it',  # Model identifier from HuggingFace
    'dtype': 'bfloat16',              # Floating point precision (bfloat16, float16, float32)
    
    # Training objectives
    'train_direction': False,         # Whether to train a single refusal direction
    'train_orthogonal_direction': False,  # Whether to train a direction orthogonal to the DIM direction
    'train_cone': False,              # Whether to train a refusal cone
    'train_independent_direction': False, # Whether to train a direction that is independent of the DIM direction
    
    # Optimization parameters
    'epochs': 1,                      # Number of training epochs
    'lr': 1e-2,                       # Learning rate for optimization
    'batch_size': 1,                  # Batch size for training
    'effective_batch_size': 16,       # Effective batch size (uses gradient accumulation)
    'patience': 5,                    # Patience for early stopping
    'n_lr_reduce': 2,                 # Number of learning rate reductions before stopping
    
    # Cone parameters
    'min_cone_dim': 2,                # Minimum dimension of the refusal cone (number of basis vectors)
    'max_cone_dim': 3,               # Maximum dimension of the refusal cone (number of basis vectors)
    'n_sample': 8,                    # Number of random samples to use during training
    'fixed_samples': 8,               # Number of fixed samples for evaluation
    'sampling_method': "hypersphere", # Method for sampling vectors ('hypersphere' or 'interpolation')
    'optimize_basis': True,           # Whether to optimize the basis vectors directly
    
    # Loss weights
    'ablation_lambda': 1,             # Weight for the ablation loss
    'addition_lambda': 0.2,           # Weight for the addition loss
    'retain_lambda': 1,               # Weight for the retain loss
    
    # Miscellaneous
    'target_generation_batch_size': 512,  # Batch size for generating targets
    'filter_data': True,              # Whether to filter data
    'filter_batch_size': 32,          # Batch size for filtering data
    'splits': "saladbench",           # Dataset split to use
}


def sample_hypersphere_gaussian(batch_size, dim, dtype, device):
    # Sample from standard normal distribution
    samples = torch.randn(batch_size, dim, dtype=dtype, device=device).abs()
    # Normalize to unit length
    samples = samples / torch.norm(samples, dim=1, keepdim=True)
    return samples

def sample_prob_vectors(batch_size, dim, dtype, device):
    samples = torch.exp(torch.randn(batch_size, dim, dtype=dtype, device=device))
    samples = samples / samples.sum(dim=1, keepdim=True)
    return samples

def compute_ce_loss(logits, labels):
    logits = logits.view(-1, logits.size(-1))
    labels = labels.view(-1)
    return torch.nn.functional.cross_entropy(logits, labels, ignore_index=-100)

def kl_div_fn(logits_a, logits_b, reduction='batchmean'):
    # Compute log-probabilities for the first distribution
    logits_a = logits_a.to(torch.float64)
    logits_b = logits_b.to(torch.float64)
    
    return torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(logits_a, dim=-1), 
        torch.nn.functional.softmax(logits_b, dim=-1),
        reduction=reduction
    )


def clip_grad_norm(grad, max_norm):
    total_norm = grad.norm()
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    return grad * clip_coef_clamped


class RefusalCone(nn.Module):
    def __init__(
        self, 
        module: Envoy, 
        dim: int, 
        n_vectors: int, 
        dtype: torch.dtype = torch.bfloat16,
        init_vectors: torch.Tensor | None = None, 
        orthogonal_vectors: torch.Tensor | None = None
    ) -> None:
        super(RefusalCone, self).__init__()
        self.module = module
        self.dtype = dtype
        self.n_vectors = n_vectors
        self.fn_vectors = [torch.nn.Parameter(torch.randn(dim, dtype=self.dtype).cuda(), requires_grad=True) for _ in range(n_vectors)]
        if init_vectors is not None:
            for i, init_vector in enumerate(init_vectors):
                init_vector = init_vector / init_vector.norm()
                self.fn_vectors[i].data = init_vector.detach().clone().cuda().to(self.dtype)
        if orthogonal_vectors is None:
            self.orthogonal_vectors = None
        else:
            self.orthogonal_vectors = [(o / o.norm()).to(self.dtype).cpu() for o in orthogonal_vectors]
        self.orthogonalize()
    
    def __call__(self, direction):
        normalized_direction = direction / direction.norm()
        normalized_direction = normalized_direction.to(self.dtype)
        for layer in self.module.layers:
            self.ablate_input(layer, normalized_direction)
            self.ablate_output(layer.self_attn, normalized_direction, 3)
            self.ablate_output(layer.mlp, normalized_direction, 1)
    
    def ablate_output(self, layer, direction, tuple_length=1):
        if tuple_length > 1:
            activation = layer.output[0][:]
        else:
            activation = layer.output
        projection = projection_einops(activation, direction)
        new_activation = activation - projection
        if tuple_length == 2:
            layer.output = (new_activation, layer.output[1])
        elif tuple_length == 3:
            layer.output = (new_activation, layer.output[1], layer.output[2])
        elif tuple_length == 1:
            layer.output = new_activation
    
    def ablate_input(self, layer, direction):
        projection = projection_einops(layer.input, direction)
        new_activation = layer.input - projection
        layer.input = new_activation

    def add(self, direction, alpha, layer_idx):
        direction = direction / direction.norm()
        direction = direction.to(self.dtype)
        self.module.layers[layer_idx].input += alpha * direction
    
    def transform(self, sample):
        fn_vectors = torch.stack(self.fn_vectors, dim=0)
        transformed_sample = torch.matmul(sample, fn_vectors).to(self.dtype)
        transformed_sample = transformed_sample / torch.norm(transformed_sample)
        return transformed_sample

    def parameters(self):
        return self.fn_vectors
    
    def orthogonalize(self):
        with torch.no_grad():
            for i in range(len(self.fn_vectors)):
                for j in range(i):
                    self.fn_vectors[i].data.sub_(projection_einops(self.fn_vectors[i].data, self.fn_vectors[j].data))
                self.fn_vectors[i].data.div_(self.fn_vectors[i].data.norm())
            
            if self.orthogonal_vectors:
                v = self.fn_vectors[0].data.clone().cpu().to(torch.float64)
                
                # Stack your vectors as rows in a matrix A
                A = torch.stack([vec.flatten().to(torch.float64) for vec in self.orthogonal_vectors])
                # Compute projection matrix P = A^T(AA^T)^-1A
                # The nullspace projector is then I - P
                AAT = A @ A.t()
                AAT_inv = torch.inverse(AAT)
                P = A.t() @ AAT_inv @ A
                I = torch.eye(P.shape[0], device=P.device)
                
                # Project onto nullspace (orthogonal complement)
                v_flat = v.flatten()
                v_ortho = (I - P) @ v_flat
                
                # Reshape back to original shape and normalize
                v_ortho = v_ortho.reshape(v.shape)
                v_ortho = v_ortho / torch.norm(v_ortho)

                self.fn_vectors[0].data = v_ortho.to(self.fn_vectors[0].dtype).to(self.fn_vectors[0].device)
    
    def normalize(self):
        for i in range(len(self.fn_vectors)):
            self.fn_vectors[i].data.div_(self.fn_vectors[i].data.norm())
                    

def refusal_direction_optimization(
    model, train_dataset, alpha, best_layer, dtype, num_target_tokens, refusal_tokens,
    batch_size=DEFAULT_CONFIG['batch_size'], 
    effective_batch_size=DEFAULT_CONFIG['effective_batch_size'], 
    epochs=DEFAULT_CONFIG['epochs'], 
    lr=DEFAULT_CONFIG['lr'], 
    cone_dim=1,
    n_sample=DEFAULT_CONFIG['n_sample'], 
    fixed_samples=DEFAULT_CONFIG['fixed_samples'], 
    sampling_method=DEFAULT_CONFIG['sampling_method'], 
    optimize_basis=DEFAULT_CONFIG['optimize_basis'], 
    fixed_basis_vectors=[], 
    ablation_lambda=DEFAULT_CONFIG['ablation_lambda'], 
    addition_lambda=DEFAULT_CONFIG['addition_lambda'], 
    retain_lambda=DEFAULT_CONFIG['retain_lambda'], 
    patience=DEFAULT_CONFIG['patience'], 
    init_vectors=[], 
    n_lr_reduce=DEFAULT_CONFIG['n_lr_reduce'], 
    orthogonal_vectors=[],
    independence_threshold=1e-5,
    previous_directions=None
):

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=rdo_collate)

    if hasattr(model, "base_model") and hasattr(model.base_model, "model") and hasattr(model.base_model.model, "model"):
        module = model.base_model.model.model
    else:
        module = model.model

    operation = RefusalCone(
        module,
        model.config.hidden_size,
        cone_dim, dtype=dtype,
        init_vectors=init_vectors,
        orthogonal_vectors=orthogonal_vectors
    )

    logits_scaling = get_logits_scaling(model)

    optimizer = torch.optim.AdamW(operation.parameters(), lr=lr, betas=(.9,.98), weight_decay=0.0, amsgrad=True)

    print("Cone dim", cone_dim)
    if cone_dim == 1:
        n_sample = 0

    accumulation_steps = effective_batch_size // batch_size
    print("Accumulation steps", accumulation_steps)
    vectors = []
    train_losses = []
    stopped = False
    lowest_training_loss = float('inf')
    bypass_scores = []
    patience_counter = 0
    lr_reduce_counter = 0

    print("Starting training")

    step_counter = 0
    batch_sample_ablation_loss = 0.0
    batch_sample_addition_loss = 0.0
    batch_sample_retain_loss = 0.0
    batch_basis_ablation_loss = 0.0
    batch_basis_addition_loss = 0.0
    batch_basis_retain_loss = 0.0

    batch_sample_bypass_scores = []
    batch_sample_induce_scores = []
    batch_basis_bypass_scores = []
    batch_basis_induce_scores = []

    add_layer = best_layer

    if n_sample > 0:
        if sampling_method == "hypersphere":
            fixed_sample_vectors = sample_hypersphere_gaussian(fixed_samples, cone_dim, dtype, model.device)
        elif sampling_method == "interpolation":
            fixed_sample_vectors = sample_prob_vectors(fixed_samples, cone_dim, dtype, model.device)
        fixed_sample_vectors = [fixed_sample_vectors[i] for i in range(fixed_samples)]

    for epoch in range(epochs):
        print('Epoch', epoch)
        for _, batch in enumerate(train_dataloader):
            ablation_prompt = batch['ablation_prompt']
            ablation_labels = batch['ablation_labels']
            addition_prompt = batch['addition_prompt']
            addition_labels = batch['addition_labels']
            retain_prompt = batch['retain_prompt']
            harmful_prompt = batch['harmful_prompt']
            harmless_prompt = batch['harmless_prompt']

            if n_sample > 0:
                if sampling_method == "hypersphere":
                    sample_vectors = sample_hypersphere_gaussian(n_sample, cone_dim, dtype, model.device)
                elif sampling_method == "interpolation":
                    sample_vectors = sample_prob_vectors(n_sample, cone_dim, dtype, model.device)

                for sample_vector in sample_vectors:
                    if ablation_lambda > 0:
                        with model.trace() as tracer:
                            with tracer.invoke(ablation_prompt):
                                direction = operation.transform(sample_vector)
                                operation(direction)
                                logits = model.lm_head.output[:, :-1] / logits_scaling
                                sample_ablation_loss = compute_ce_loss(logits, ablation_labels) / n_sample
                                log = sample_ablation_loss.detach().item().save()
                            (ablation_lambda * sample_ablation_loss).backward()
                    batch_sample_ablation_loss += log
                    if addition_lambda > 0:
                        with model.trace() as tracer:
                            with tracer.invoke(addition_prompt):
                                direction = operation.transform(sample_vector)
                                operation.add(direction, alpha, add_layer)
                                logits = model.lm_head.output[:, :-1] / logits_scaling
                                sample_addition_loss = compute_ce_loss(logits, addition_labels) / n_sample
                                log = sample_addition_loss.detach().item().save()
                            (addition_lambda * sample_addition_loss).backward()
                        batch_sample_addition_loss += log
                    if retain_lambda > 0:
                        with model.trace() as tracer:
                            with tracer.invoke(retain_prompt):
                                baseline_retain_logits = model.lm_head.output[:, -num_target_tokens:] / logits_scaling
                            with tracer.invoke(retain_prompt):
                                direction = operation.transform(sample_vector)
                                operation(direction)
                                sample_retain_logits = model.lm_head.output[:, -num_target_tokens:] / logits_scaling
                                sample_retain_loss = kl_div_fn(baseline_retain_logits, sample_retain_logits).mean() / n_sample
                                log = sample_retain_loss.detach().item().save()
                            (retain_lambda * sample_retain_loss).backward()
                        batch_sample_retain_loss += log
                
            if optimize_basis:
                for fn_vector in operation.fn_vectors:
                    if ablation_lambda > 0:
                        with model.trace() as tracer:
                            with tracer.invoke(ablation_prompt):
                                operation(fn_vector)
                                logits = model.lm_head.output[:, :-1] / logits_scaling
                                basis_ablation_loss = compute_ce_loss(logits, ablation_labels) / cone_dim
                                log = basis_ablation_loss.detach().item().save()
                            (ablation_lambda * basis_ablation_loss).backward()
                        batch_basis_ablation_loss += log

                    if addition_lambda > 0:
                        with model.trace() as tracer:
                            with tracer.invoke(addition_prompt):
                                operation.add(fn_vector, alpha, add_layer)
                                logits = model.lm_head.output[:, :-1] / logits_scaling
                                basis_addition_loss = compute_ce_loss(logits, addition_labels) / cone_dim
                                log = basis_addition_loss.detach().item().save()
                            (addition_lambda * basis_addition_loss).backward()
                        batch_basis_addition_loss += log

                    if retain_lambda > 0:
                        with model.trace() as tracer:
                            with tracer.invoke(retain_prompt):
                                baseline_retain_logits = model.lm_head.output[:, -num_target_tokens:] / logits_scaling
                            with tracer.invoke(retain_prompt):
                                operation(fn_vector)
                                retain_logits = model.lm_head.output[:, -num_target_tokens:] / logits_scaling
                                basis_retain_loss = kl_div_fn(baseline_retain_logits, retain_logits).mean() / cone_dim
                                log = basis_retain_loss.detach().item().save()
                            (retain_lambda * basis_retain_loss).backward()
                        batch_basis_retain_loss += log
                
            with torch.no_grad():
                with model.trace() as tracer:
                    for fn_vector in operation.fn_vectors:
                        with tracer.invoke(harmful_prompt):
                            operation(fn_vector)
                            last_token_logits = model.lm_head.output[:, -1] / logits_scaling
                            bypass_score = refusal_metric(last_token_logits, refusal_tokens).detach().item().save()
                        batch_basis_bypass_scores.append(bypass_score)

                with model.trace() as tracer:
                    for fn_vector in operation.fn_vectors:
                        with tracer.invoke(harmless_prompt):
                            operation.add(fn_vector, alpha, add_layer)
                            last_token_logits = model.lm_head.output[:, -1] / logits_scaling
                            induce_score = refusal_metric(last_token_logits, refusal_tokens).detach().item().save()
                        batch_basis_induce_scores.append(induce_score)
                if n_sample > 0:
                    with model.trace() as tracer:
                        for fixed_sample_vector in fixed_sample_vectors:
                            with tracer.invoke(harmful_prompt):
                                direction = operation.transform(fixed_sample_vector)
                                operation(direction)
                                sample_last_token_logits = model.lm_head.output[:, -1] / logits_scaling
                                sample_bypass_score = refusal_metric(sample_last_token_logits, refusal_tokens).detach().item().save()
                                batch_sample_bypass_scores.append(sample_bypass_score)
                    with model.trace() as tracer:
                        for fixed_sample_vector in fixed_sample_vectors:
                            with tracer.invoke(harmless_prompt):
                                direction = operation.transform(fixed_sample_vector)
                                operation.add(direction, alpha, add_layer)
                                sample_last_token_logits = model.lm_head.output[:, -1] / logits_scaling
                                sample_induce_score = refusal_metric(sample_last_token_logits, refusal_tokens).detach().item().save()
                                batch_sample_induce_scores.append(sample_induce_score)

                step_counter += 1
                if step_counter % accumulation_steps == 0:
                    for fn_vector in operation.fn_vectors:
                        fn_vector.grad.sub_(projection_einops(fn_vector.grad, fn_vector.data))
                    for fn_vector in operation.fn_vectors:
                        fn_vector.grad.div_(accumulation_steps)
                    torch.nn.utils.clip_grad_norm_(operation.parameters(), 10.0)
                    grad_norm = operation.fn_vectors[-1].grad.norm().item()
                    optimizer.step()
                    optimizer.zero_grad()
                    if len(fixed_basis_vectors) > 0:
                        for i, fixed_basis_vector in enumerate(fixed_basis_vectors):
                            fixed_basis_vector = fixed_basis_vector / fixed_basis_vector.norm()
                            operation.fn_vectors[i].data.copy_(fixed_basis_vector.data)
                    operation.orthogonalize()

                    batch_sample_ablation_loss /= accumulation_steps
                    batch_sample_addition_loss /= accumulation_steps
                    batch_sample_retain_loss /= accumulation_steps
                    batch_basis_ablation_loss /= accumulation_steps
                    batch_basis_addition_loss /= accumulation_steps
                    batch_basis_retain_loss /= accumulation_steps

                    train_loss = batch_sample_ablation_loss + batch_sample_addition_loss + batch_sample_retain_loss + batch_basis_ablation_loss + batch_basis_addition_loss + batch_basis_retain_loss
                    train_losses.append(train_loss)

                    batch_basis_bypass_scores = [s.value for s in batch_basis_bypass_scores]
                    batch_basis_induce_scores = [s.value for s in batch_basis_induce_scores]
                    basis_bypass_scores = [torch.mean(torch.tensor(batch_basis_bypass_scores[i::cone_dim])).item() for i in range(cone_dim)]
                    basis_induce_scores = [torch.mean(torch.tensor(batch_basis_induce_scores[i::cone_dim])).item() for i in range(cone_dim)]

                    vectors.append(torch.stack(operation.fn_vectors, dim=0).detach().cpu().data.clone())
                    bypass_scores.append(basis_bypass_scores)

                    if n_sample > 0:
                        batch_sample_bypass_scores = [s.value for s in batch_sample_bypass_scores]
                        batch_sample_induce_scores = [s.value for s in batch_sample_induce_scores]

                        sample_vector_bypass_scores = [torch.mean(torch.tensor(batch_sample_bypass_scores[i::fixed_samples])).item() for i in range(fixed_samples)]
                        min_sample_bypass_score = min(sample_vector_bypass_scores)
                        max_sample_bypass_score = max(sample_vector_bypass_scores)
                        mean_sample_bypass_score = torch.mean(torch.tensor(sample_vector_bypass_scores)).item()
                        std_sample_bypass_score = torch.std(torch.tensor(sample_vector_bypass_scores)).item()

                        sample_vector_induce_scores = [torch.mean(torch.tensor(batch_sample_induce_scores[i::fixed_samples])).item() for i in range(fixed_samples)]
                        min_sample_induce_score = min(sample_vector_induce_scores)
                        max_sample_induce_score = max(sample_vector_induce_scores)
                        mean_sample_induce_score = torch.mean(torch.tensor(sample_vector_induce_scores)).item()
                        std_sample_induce_score = torch.std(torch.tensor(sample_vector_induce_scores)).item()

                    print("Step", step_counter, "Loss", round(train_loss, 3), "train/basis_vector_bypass_score", [round(s, 2) for s in basis_bypass_scores], "train/basis_vector_induce_score", [round(s, 2) for s in basis_induce_scores], "Grad norm", round(grad_norm, 2))

                    if n_sample > 0:
                        print("train/mean_sample_bypass_scores", round(mean_sample_bypass_score, 2), "train/mean_sample_induce_scores", round(mean_sample_induce_score, 2))
                    if train_loss >= lowest_training_loss:
                        patience_counter += 1
                    else:
                        lowest_training_loss = train_loss
                        patience_counter = 0
                    if patience_counter >= patience:
                        if lr_reduce_counter >= n_lr_reduce:
                            print(f'Stopping')
                            stopped = True
                            break
                        lr_reduce_counter += 1
                        print("Reducing lr to", optimizer.param_groups[0]['lr'] / 10)
                        optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / 10
                        patience_counter = 0

                    batch_sample_ablation_loss = 0.
                    batch_sample_addition_loss = 0.
                    batch_sample_retain_loss = 0.
                    batch_basis_ablation_loss = 0.
                    batch_basis_addition_loss = 0.
                    batch_basis_retain_loss = 0.

                    batch_sample_bypass_scores = []
                    batch_sample_induce_scores = []
                    batch_basis_bypass_scores = []
                    batch_basis_induce_scores = []

                    torch.cuda.empty_cache()

        if stopped:
            break

    save_vectors = vectors
    sorted_indices = torch.argsort(torch.tensor(train_losses))
    
    best_independent_vector = None
    best_independent_loss = None
    best_residual_norm = None
    
    for idx in sorted_indices:
        candidate = save_vectors[idx][0].clone()  # Shape: (hidden_dim,)
        candidate = candidate / candidate.norm()
        
        # Check linear independence if we have previous directions
        if previous_directions is not None and len(previous_directions) > 0:
            prev_dirs = torch.stack([d.to(candidate.device).to(candidate.dtype) for d in previous_directions])
            
            residual = candidate.clone()
            for prev_dir in prev_dirs:
                prev_dir = prev_dir / prev_dir.norm()
                proj = (residual @ prev_dir) * prev_dir
                residual = residual - proj
            
            residual_norm = residual.norm().item()
            
            if residual_norm < independence_threshold:
                continue  # Skip this one, try next best
            
            best_residual_norm = residual_norm
        
        # Found a valid direction
        best_independent_vector = save_vectors[idx]
        best_independent_loss = train_losses[idx]
        break
    
    if best_independent_vector is None:
        print(f"No linearly independent direction found (checked {len(save_vectors)} candidates)")
    else:
        print(f"Found independent direction at rank {sorted_indices.tolist().index(idx)} with loss {best_independent_loss:.4f}, residual norm {best_residual_norm}")
    
    return {"vectors": vectors, "lowest_loss": lowest_training_loss, "refusal_scores": bypass_scores, "train_losses": train_losses, "lowest_loss_vector": best_independent_vector}
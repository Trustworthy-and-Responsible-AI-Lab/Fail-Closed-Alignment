import torch
import torch.nn.functional as F


def compute_ce_loss(logits, labels) -> torch.Tensor:
    shifted_logits = logits[..., :-1, :].contiguous()
    shifted_labels = labels[..., 1:].contiguous()

    return F.cross_entropy(
        shifted_logits.view(-1, shifted_logits.size(-1)),
        shifted_labels.view(-1), 
        ignore_index=-100
    )


def compute_kl_div_loss(
    logits: torch.Tensor,
    ref_logits: torch.Tensor,   # pre-processed logits from base model
    labels: torch.Tensor, 
    reduction: str ='batchmean'
):
    # Compute log-probabilities for the first distribution
    shifted_logits = logits[..., :-1, :].contiguous()   # (batch_size, seq_len - 1, vocab_size)
    shifted_labels = labels[..., 1:].contiguous()       # (batch_size, seq_len - 1)
    shifted_mask = shifted_labels != -100

    # Filter out the logits corresponding to padding/user tokens
    processed_logits = []
    for i in range(len(ref_logits)):
        processed_logits.append(shifted_logits[i][shifted_mask[i]])

    res = []
    for i, curr_ref_logits in enumerate(ref_logits):
        curr_logits = processed_logits[i]
        res.append(torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(curr_logits, dim=-1), 
            torch.nn.functional.softmax(curr_ref_logits.to(curr_logits.device), dim=-1),
            reduction=reduction
        ))
    
    return torch.stack(res).mean()
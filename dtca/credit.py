"""Target filtering and within-state POS credit normalization."""

import math

import torch


def targets(logps, masked, committed, valid, threshold=0.2):
    keep = masked.bool() & valid.bool()
    if threshold > 0:
        keep &= logps.detach().ge(math.log(threshold))
    return keep | (committed.bool() & valid.bool())


def credit(weights, advantages, mask):
    """Preserve each nonempty state's mean response advantage."""
    mask = mask.bool()
    total = (weights * mask).sum(-1, keepdim=True)
    mean = total / mask.sum(-1, keepdim=True).clamp_min(1)
    return torch.where(mask, weights / mean.clamp_min(torch.finfo(weights.dtype).tiny) * advantages[:, None], 0)

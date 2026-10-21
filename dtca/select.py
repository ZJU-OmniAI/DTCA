"""Contribution-weighted timestep selection."""

import torch
import torch.distributed as dist


def select(steps, mask, probabilities, weights, mass=0.7, minimum=2, maximum=32):
    """Return chronological steps selected by sum(w_pos * (1-p_commit))."""
    if not (0 < mass <= 1 and 1 <= minimum <= maximum):
        raise ValueError("Invalid selection mass or step limits")
    if not (steps.shape == mask.shape == probabilities.shape == weights.shape):
        raise ValueError("Selection tensors must align")
    chosen = []
    for row in range(steps.shape[0]):
        valid = mask[row].bool()
        ids = steps[row, valid].unique(sorted=True)
        scores = torch.stack([(weights[row] * (1 - probabilities[row]) * (valid & steps[row].eq(s))).sum() for s in ids]) if ids.numel() else torch.empty(0)
        order = scores.argsort(descending=True, stable=True)
        if not ids.numel():
            chosen.append([])
            continue
        count = int(torch.searchsorted(scores[order].cumsum(0), scores.sum() * mass).item()) + 1
        count = min(len(ids), maximum, max(minimum, count))
        chosen.append(sorted(ids[order[:count]].tolist()))
    return chosen


def schedule(steps, mask, probabilities, weights, **kwargs):
    from trainer.masking_strategy import MaskingSchedule

    chosen = select(steps, mask, probabilities, weights, **kwargs)
    maximum = max([len(row) for row in chosen] + [1])
    if dist.is_initialized():
        device = torch.device("cuda", torch.cuda.current_device()) if dist.get_backend() == "nccl" else torch.device("cpu")
        count = torch.tensor(maximum, device=device)
        dist.all_reduce(count, op=dist.ReduceOp.MAX)
        maximum = int(count)
    rows = []
    for row, selected in enumerate(chosen):
        valid = mask[row].bool()
        entries = [(step, steps[row].ge(step), valid & steps[row].eq(step)) for step in selected]
        entries += [(-1, torch.zeros_like(valid), torch.zeros_like(valid)) for _ in range(maximum - len(entries))]
        rows.append(entries)
    return MaskingSchedule(rows, "dtca")

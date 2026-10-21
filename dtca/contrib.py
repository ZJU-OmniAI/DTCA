"""Isolated token interventions using total maximum-probability change."""

from dataclasses import dataclass

import torch


def confidence(logits):
    """Return each position's maximum vocabulary probability in float32."""
    return logits.float().softmax(-1).amax(-1)


def change(before, after, remaining):
    """Return confidence sums over a shared mask and their positive total change."""
    if before.shape != after.shape or before.shape != remaining.shape:
        raise ValueError("Confidence tensors and remaining mask must align")
    before = before.double().masked_fill(~remaining.bool(), 0).sum(-1)
    after = after.double().masked_fill(~remaining.bool(), 0).sum(-1)
    return before, after, (after - before).clamp_min(0)


@dataclass
class Observation:
    tokens: torch.Tensor
    before: torch.Tensor
    after: torch.Tensor
    values: torch.Tensor
    counts: torch.Tensor
    steps: torch.Tensor


@torch.no_grad()
def reveal(model, prompt, *, length=256, temperature=1.0, mask_id=126336):
    """Commit two tokens per step and probe each separately without KV cache.

    Each probe starts at the true precommit state. The other planned token is
    still masked and belongs to the evaluation set, including the last step.
    Input contains a single unpadded prompt.
    """
    if prompt.ndim != 2 or prompt.shape[0] != 1:
        raise ValueError("Contribution collection accepts one prompt at a time")
    if length < 2 or length % 2 or temperature < 0:
        raise ValueError("Use an even generation length >= 2 and temperature >= 0")
    start = prompt.shape[1]
    state = torch.full((1, start + length), mask_id, device=prompt.device, dtype=torch.long)
    state[:, :start] = prompt
    before = torch.zeros_like(state, dtype=torch.float64)
    after, values = before.clone(), before.clone()
    counts, steps = torch.zeros_like(state), torch.full_like(state, -1)
    was_training = model.training
    model.eval()
    try:
        for step in range(length // 2):
            masked = state.eq(mask_id)
            masked[:, :start] = False
            logits = model(state, use_cache=False).logits
            maxima = confidence(logits)
            # Same Gumbel sampling rule as Drift; mask is never a commitment.
            scores = logits.double().clone()
            scores[..., mask_id] = -torch.inf
            if temperature:
                noise = -torch.log(torch.rand_like(scores).clamp_min(torch.finfo(scores.dtype).tiny))
                scores = scores - temperature * noise.log()
            proposed = scores.argmax(-1)
            token_prob = logits.float().softmax(-1).gather(-1, proposed[..., None]).squeeze(-1)
            positions = token_prob.masked_fill(~masked, -torch.inf).topk(2, dim=1).indices[0]
            del logits, scores, token_prob
            for position in positions.tolist():
                remaining = masked.clone()
                remaining[0, position] = False
                probe = state.clone()
                probe[0, position] = proposed[0, position]
                probe_logits = model(probe, use_cache=False).logits
                b, a, value = change(maxima, confidence(probe_logits), remaining)
                before[0, position], after[0, position], values[0, position] = b[0], a[0], value[0]
                counts[0, position] = remaining.sum()
                steps[0, position] = step
                del probe_logits
            state[0, positions] = proposed[0, positions]
        return Observation(*(tensor[:, start:].cpu() for tensor in (state, before, after, values, counts, steps)))
    finally:
        model.train(was_training)

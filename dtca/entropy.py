"""Response-level entropy-flow calibration."""

import torch


@torch.no_grad()
def calibrate(logps, entropy, credits, responses, learning_rate=1e-6):
    if not (logps.shape == entropy.shape == credits.shape == responses.shape) or logps.ndim != 1:
        raise ValueError("Flattened entropy statistics must align")
    delta = -learning_rate * credits * (1 - logps.exp()).square() * (logps + entropy)
    result = credits.clone()
    for response in responses.unique():
        index = responses.eq(response)
        flow = delta[index]
        positive, negative = flow.clamp_min(0).sum(), (-flow).clamp_min(0).sum()
        magnitude = positive + negative
        balancing = ((negative - positive) / magnitude if magnitude > 0 else magnitude).clamp(-0.95, 0.95)
        scaled = credits[index] * torch.where(flow > 0, 1 + balancing, 1 - balancing)
        if scaled.sum() != 0:
            result[index] = scaled * (credits[index].sum() / scaled.sum())
    return result

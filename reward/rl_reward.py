# Modified for DTCA: retain Drift's math scoring and reward dispatcher.
from typing import Callable, Dict, List, Union

import torch

from reward.math_utils import is_math_equal as _is_math_equal_sync


def extract_final_boxed_answer(s: str) -> str:
    r"""Extract the content of the last \boxed{...} in model output."""
    tag = r'\boxed{'
    start = s.rfind(tag)
    if start == -1:
        return "Can not extract the answer!"

    i = start + len(tag)
    depth = 1
    buf = []

    while i < len(s) and depth:
        ch = s[i]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                break
        buf.append(ch)
        i += 1

    return ''.join(buf) if depth == 0 else "Can not extract the answer!"


def is_math_equal(prediction: str, ground_truth: str) -> bool:
    """Synchronous math equality check using math_utils.is_math_equal."""
    return _is_math_equal_sync(prediction, ground_truth)


def math_reward(
    completions: List[str],
    ground_truths: List[str],
    **kwargs,
) -> List[float]:
    r"""
    Math correctness reward: extract \boxed{} answer and compare with
    ground truth. Correct → +1.0, incorrect → 0.0.
    """
    rewards = []
    for completion, gt in zip(completions, ground_truths):
        prediction = extract_final_boxed_answer(completion)
        if prediction == "Can not extract the answer!":
            rewards.append(0.0)
        elif is_math_equal(prediction, gt):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


REWARD_FUNCS_REGISTRY: Dict[str, Callable] = {"math": math_reward}


def compute_reward_per_func(
    completions: List[str],
    ground_truths: Union[List[str], List[dict]],
    reward_funcs: List[str],
    n_workers: int = 64,
    num_chunks: int = 64,
) -> torch.Tensor:
    """Compute math rewards, returning an (N, num_funcs) tensor."""
    N = len(completions)
    num_funcs = len(reward_funcs)
    rewards_per_func = torch.zeros(N, num_funcs, dtype=torch.float32)

    shared_kwargs = dict(
        n_workers=n_workers,
        num_chunks=num_chunks,
    )

    for func_idx, func_name in enumerate(reward_funcs):
        if func_name not in REWARD_FUNCS_REGISTRY:
            raise ValueError(
                f"Unknown reward function '{func_name}'. "
                f"Available: {list(REWARD_FUNCS_REGISTRY.keys())}"
            )
        reward_fn = REWARD_FUNCS_REGISTRY[func_name]

        func_rewards = reward_fn(
            completions=completions,
            ground_truths=ground_truths,
            **shared_kwargs,
        )

        rewards_per_func[:, func_idx] = torch.tensor(
            func_rewards, dtype=torch.float32
        )

    return rewards_per_func

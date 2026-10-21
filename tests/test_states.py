"""Denoising-state reconstruction and sparse-trajectory boundary cases."""

import torch
from accelerate import Accelerator

from dtca.select import schedule
from rollout.llada_rollout import remap_step_map_continuous
from trainer.masking_strategy import build_masking_schedule


def test_step_remapping_preserves_special_token_commitment_time():
    steps = torch.tensor([[2, 8, 8, 12]])
    valid = torch.tensor([[1, 0, 1, 1]])
    assert remap_step_map_continuous(steps, valid).tolist() == [[0, 1, 1, 2]]


def test_dtca_masks_future_special_tokens_without_optimizing_them():
    steps = torch.tensor([[0, 1, 2]])
    valid = torch.tensor([[1, 0, 1]])
    result = schedule(steps, valid, torch.full((1, 3), .5), torch.ones(1, 3))
    first, last = result.masking_inputs[0]
    assert first[0] == 0 and first[1].tolist() == [True, True, True]
    assert first[2].tolist() == [True, False, False]
    assert last[0] == 2 and last[1].tolist() == [False, False, True]


def test_rldf_one_observed_state_is_a_real_training_state():
    Accelerator(cpu=True)
    result = build_masking_schedule(torch.zeros(1, 3, dtype=torch.long), torch.ones(1, 3),
                                     "sequential_masking", k=16, token_probs=torch.full((1, 3), .8))
    assert len(result.masking_inputs[0]) == 16
    assert all(step == 0 and masked.all() and committed.all()
               for step, masked, committed in result.masking_inputs[0])

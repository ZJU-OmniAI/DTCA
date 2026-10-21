import torch

from dtca.select import select


def test_mass_uses_committed_probabilities_and_pos_weights():
    steps = torch.tensor([[0, 0, 1, 2, 3]])
    probabilities = torch.tensor([[0.9, 0.9, 0.5, 0.6, 0.75]])
    weights = torch.tensor([[1.0, 1.0, 1.2, 0.8, 1.0]])
    chosen = select(steps, torch.ones_like(steps), probabilities, weights, mass=.7, minimum=1, maximum=4)
    assert chosen == [[1, 2, 3]]  # Scores .2, .6, .32, .25; target .959.


def test_cap_small_rollout_zero_mass_and_empty():
    steps = torch.arange(40)[None]
    result = select(steps, torch.ones_like(steps), torch.zeros_like(steps), torch.ones_like(steps), mass=1, maximum=32)
    assert len(result[0]) == 32
    assert select(steps[:, :1], torch.ones(1, 1), torch.ones(1, 1), torch.ones(1, 1)) == [[0]]
    assert select(steps, torch.zeros_like(steps), torch.ones_like(steps), torch.ones_like(steps)) == [[]]

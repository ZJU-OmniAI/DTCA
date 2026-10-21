import pytest
import torch

from dtca.credit import credit, targets
from dtca.entropy import calibrate


def test_state_mean_credit_and_current_commitment_retention():
    keep = targets(torch.tensor([[.1, .3, .01]]).log(), torch.ones(1, 3),
                   torch.tensor([[True, False, False]]), torch.ones(1, 3), threshold=.2)
    assert keep.tolist() == [[True, True, False]]
    weights, advantage = torch.tensor([[.8, 1.2, 1.0]]), torch.tensor([-2.0])
    values = credit(weights, advantage, keep)
    assert values[keep].mean() == -2
    assert values[~keep].eq(0).all()
    assert credit(torch.ones_like(weights), advantage, keep).tolist() == [[-2, -2, 0]]


def test_entropy_response_conservation_and_common_scale_invariance():
    logps = torch.tensor([.2, .7, .3, .8, .5, .4]).log()
    entropy = torch.tensor([1.5, .1, 1.4, .1, 1., .5])
    signals = torch.tensor([1., 2., 1., -1., -2., -3.])
    rows = torch.tensor([0, 0, 0, 1, 1, 1])
    result = calibrate(logps, entropy, signals, rows)
    for row in (0, 1):
        assert result[rows.eq(row)].sum() == pytest.approx(float(signals[rows.eq(row)].sum()))
    assert torch.allclose(calibrate(logps, entropy, signals*.01, rows), result*.01, atol=1e-7)
    assert torch.equal(calibrate(logps, entropy, torch.zeros(6), rows), torch.zeros(6))

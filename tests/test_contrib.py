from types import SimpleNamespace

import pytest
import torch

from dtca.contrib import change, confidence, reveal


def test_full_vocabulary_maximum_can_change_its_argmax():
    before = torch.tensor([[[0.6, 0.1, 0.3], [0.1, 0.7, 0.2]]]).log()
    after = torch.tensor([[[0.1, 0.8, 0.1], [0.6, 0.1, 0.3]]]).log()
    b, a, delta = change(confidence(before), confidence(after), torch.ones(1, 2, dtype=torch.bool))
    assert float(b) == pytest.approx(1.3)
    assert float(a) == pytest.approx(1.4)
    # The final token at position 2 could be class 1, whose probability fell.
    # The contribution still uses the NEW class maximum at each position.
    assert float(delta) == pytest.approx(0.1)


def test_total_difference_not_mean_or_sum_of_positive_position_differences():
    b, a, value = change(torch.tensor([[0.3, 0.5, 0.2, 0.99]]),
                         torch.tensor([[0.4, 0.45, 0.35, 0.01]]),
                         torch.tensor([[True, True, True, False]]))
    assert float(b) == pytest.approx(1.0)
    assert float(a) == pytest.approx(1.2)
    assert float(value) == pytest.approx(0.2)  # Not .2/3 or .1+.15.
    assert float(change(torch.ones(1, 2), torch.zeros(1, 2), torch.ones(1, 2, dtype=torch.bool))[2]) == 0


class Probe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, ids, use_cache=False):
        assert not use_cache
        self.calls.append(ids.clone())
        # Non-mask class confidence grows with each commitment.
        n = ids[:, 1:].ne(3).sum(1).float()
        logits = torch.zeros((*ids.shape, 4))
        logits[..., 1] = (1 + n)[:, None]
        logits[..., 3] = -4
        return SimpleNamespace(logits=logits)


def test_individual_reveals_restart_from_true_state_and_include_last_other_mask():
    model = Probe().train()
    result = reveal(model, torch.tensor([[0]]), length=4, temperature=0, mask_id=3)
    assert model.training
    assert len(model.calls) == 6  # 2 true states + 4 probes.
    assert model.calls[0][0, 1:].eq(3).sum() == 4
    assert model.calls[1][0, 1:].eq(3).sum() == 3
    assert model.calls[2][0, 1:].eq(3).sum() == 3
    assert model.calls[3][0, 1:].eq(3).sum() == 2
    assert model.calls[4][0, 1:].eq(3).sum() == 1
    assert model.calls[5][0, 1:].eq(3).sum() == 1
    assert sorted(result.counts[0].tolist()) == [1, 1, 3, 3]
    assert result.values.gt(0).all()
    assert torch.allclose(result.values, (result.after-result.before).clamp_min(0))


def test_probe_restores_model_mode_on_failure():
    class Broken(Probe):
        def forward(self, *args, **kwargs):
            raise RuntimeError("failure")

    model = Broken().train()
    with pytest.raises(RuntimeError):
        reveal(model, torch.tensor([[0]]), length=2, mask_id=3)
    assert model.training

"""Independent scalar checks for the equations in the released manuscript."""

import math
import random
from types import SimpleNamespace

import pytest
import torch

from dtca.contrib import reveal
from dtca.credit import targets
from dtca.entropy import calibrate
from dtca.pos import TAGS, aggregate
from dtca.select import select
from dtca.table import Table


class Context(torch.nn.Module):
    def __init__(self, seed):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.embedding = torch.nn.Parameter(torch.randn(6, 4, generator=generator) * .5)
        self.projection = torch.nn.Parameter(torch.randn(4, 6, generator=generator) * .5)

    def forward(self, ids, attention_mask=None, use_cache=False):
        assert not use_cache
        embedded = self.embedding[ids]
        mask = torch.ones_like(ids) if attention_mask is None else attention_mask
        context = (embedded * mask[..., None]).sum(1, keepdim=True) / mask.sum(1)[:, None, None]
        return SimpleNamespace(logits=(embedded + context).tanh() @ self.projection)


@pytest.mark.parametrize("seed", [3, 19, 47])
@pytest.mark.parametrize("temperature", [0., .8])
def test_individual_contributions_match_reconstructed_scalar_interventions(seed, temperature):
    torch.manual_seed(seed)
    model = Context(seed).train()
    original = {name: p.detach().clone() for name, p in model.named_parameters()}
    prompt = torch.tensor([[0, 2]])
    observation = reveal(model, prompt, length=8, temperature=temperature, mask_id=5)
    assert model.training and observation.tokens.ne(5).all()
    for step in range(4):
        committed = observation.steps[0].eq(step).nonzero().flatten().tolist()
        assert len(committed) == 2
        state = torch.cat((prompt, torch.where(observation.steps < step, observation.tokens, 5)), dim=1)
        with torch.no_grad():
            before = model(state).logits.softmax(-1)[0]
            for position in committed:
                probe = state.clone()
                probe[0, prompt.shape[1] + position] = observation.tokens[0, position]
                after = model(probe).logits.softmax(-1)[0]
                remaining = [j for j in range(8) if observation.steps[0, j] >= step and j != position]
                b = sum(float(before[2+j].max()) for j in remaining)
                a = sum(float(after[2+j].max()) for j in remaining)
                assert float(observation.before[0, position]) == pytest.approx(b, abs=1e-6)
                assert float(observation.after[0, position]) == pytest.approx(a, abs=1e-6)
                assert float(observation.values[0, position]) == pytest.approx(max(0., a-b), abs=1e-6)
                assert int(observation.counts[0, position]) == len(remaining)
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, original[name]) and parameter.grad is None


@pytest.mark.parametrize("seed", [1, 7, 31, 53])
def test_pos_aggregation_and_minmax_match_scalar_definition(seed):
    rng = random.Random(seed)
    categories = [rng.randrange(18) for _ in range(80)]
    contributions = [0. if j % 5 == 0 else rng.random()*8 for j in range(80)]
    valid = [j % 7 != 0 for j in range(80)]
    sums, counts = aggregate(torch.tensor([contributions], dtype=torch.float64),
                             torch.tensor([categories]), torch.tensor([valid]))
    table = Table(sums, counts)
    means = []
    for category in range(len(TAGS)):
        observations = [v for c, v, keep in zip(categories, contributions, valid) if c == category and keep]
        means.append(sum(observations)/len(observations) if observations else 0.)
        assert int(counts[category]) == len(observations)
    expected = [.8 + .4*(value-min(means))/(max(means)-min(means)) for value in means]
    assert table.means.tolist() == pytest.approx(means)
    assert table.weights.tolist() == pytest.approx(expected)


@pytest.mark.parametrize("seed", [2, 13, 29, 73])
def test_selection_matches_minimum_ranked_set_with_bounds(seed):
    rng = random.Random(seed)
    step_ids = [rng.randrange(45) for _ in range(200)]
    probabilities = [rng.uniform(.01, .99) for _ in step_ids]
    weights = [rng.uniform(.8, 1.2) for _ in step_ids]
    valid = [rng.random() > .2 for _ in step_ids]
    scores = {step: sum(w*(1-p) for s, p, w, keep in zip(step_ids, probabilities, weights, valid)
                        if keep and s == step) for step in set(s for s, keep in zip(step_ids, valid) if keep)}
    ranked = sorted(scores, key=lambda step: (-scores[step], step))
    total, count = 0., 0
    while total < .7*sum(scores.values()):
        total += scores[ranked[count]]
        count += 1
    count = min(len(ranked), 32, max(2, count))
    expected = sorted(ranked[:count])
    actual = select(torch.tensor([step_ids]), torch.tensor([valid]),
                    torch.tensor([probabilities], dtype=torch.float64),
                    torch.tensor([weights], dtype=torch.float64))
    assert actual == [expected]


@pytest.mark.parametrize("seed", [3, 17, 37, 61])
def test_entropy_calibration_matches_scalar_response_wide_equations(seed):
    rng = random.Random(seed)
    probabilities = [rng.uniform(.01, .99) for _ in range(18)]
    entropies = [rng.uniform(.02, 2.4) for _ in probabilities]
    rows = [j % 3 for j in range(18)]  # Response occurrences need not be contiguous.
    credits = [(-1. if row == 1 else 1. if row == 0 else 0.) * rng.uniform(.8, 1.2) for row in rows]
    flows = [-1e-6*a*(1-p)**2*(math.log(p)+h) for a, p, h in zip(credits, probabilities, entropies)]
    expected = credits.copy()
    for row in range(3):
        positions = [j for j, response in enumerate(rows) if response == row]
        positive = sum(max(0., flows[j]) for j in positions)
        negative = sum(max(0., -flows[j]) for j in positions)
        coefficient = max(-.95, min(.95, (negative-positive)/(negative+positive))) if positive+negative else 0.
        scaled = [credits[j]*(1+coefficient if flows[j] > 0 else 1-coefficient) for j in positions]
        if sum(scaled):
            for j, value in zip(positions, scaled):
                expected[j] = value * sum(credits[k] for k in positions)/sum(scaled)
    logps = torch.tensor(probabilities, dtype=torch.float64).log().requires_grad_()
    entropy = torch.tensor(entropies, dtype=torch.float64, requires_grad=True)
    signals = torch.tensor(credits, dtype=torch.float64, requires_grad=True)
    actual = calibrate(logps, entropy, signals, torch.tensor(rows))
    assert actual.tolist() == pytest.approx(expected, abs=1e-12)
    assert not actual.requires_grad
    for row in range(3):
        index = torch.tensor(rows).eq(row)
        assert actual[index].sum().item() == pytest.approx(signals[index].sum().item())


def test_training_filter_uses_target_probability_not_vocabulary_maximum():
    distribution = torch.tensor([[[.8, .1, .1], [.1, .2, .7], [.85, .1, .05], [.8, .1, .1]]])
    target_ids = torch.tensor([[1, 1, 2, 1]])
    logps = distribution.gather(-1, target_ids[..., None]).squeeze(-1).log()
    result = targets(logps, torch.tensor([[True, True, True, False]]),
                      torch.tensor([[False, False, True, False]]), torch.ones(1, 4), .2)
    assert result.tolist() == [[False, True, True, False]]

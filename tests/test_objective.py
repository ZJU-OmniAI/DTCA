"""Compare actual optimizer updates against a separately written paper loss."""

from copy import deepcopy
import math
from pathlib import Path

from omegaconf import OmegaConf
import pytest
import torch

from trainer.dtca import Trainer
from trainer.masking_strategy import MaskingSchedule

ROOT = Path(__file__).resolve().parents[1]


class Model(torch.nn.Module):
    def __init__(self, seed=11):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.embedding = torch.nn.Parameter(torch.randn(6, 4, generator=generator) * .7)
        self.projection = torch.nn.Parameter(torch.randn(4, 6, generator=generator) * .7)

    def forward(self, ids, attention_mask=None, use_cache=False):
        from types import SimpleNamespace

        embedded = self.embedding[ids]
        mask = torch.ones_like(ids) if attention_mask is None else attention_mask
        pooled = (embedded * mask[..., None]).sum(1, keepdim=True) / mask.sum(1)[:, None, None]
        return SimpleNamespace(logits=(embedded + pooled).tanh() @ self.projection)


class Engine:
    def __init__(self, model):
        self.model = model
        self.optimizer = torch.optim.SGD(model.parameters(), lr=.03)
        self.boundaries = []
        self.steps = 0

    def set_gradient_accumulation_boundary(self, boundary):
        self.boundaries.append(boundary)

    def backward(self, loss):
        loss.backward()

    def step(self):
        self.optimizer.step()
        self.steps += 1


def example(seed=11, enabled=True, size=1, beta=.01):
    config = OmegaConf.load(ROOT / "configs/dtca.yaml")
    config.dtca.enabled = False
    config.training.mask_strategy = "sequential_masking"
    config.training.micro_batch_size = size
    config.training.beta = beta
    trainer = Trainer(config, 5)
    trainer.enabled = enabled
    trainer.weights = torch.tensor([[.8, 1.2, 1., .9, 1.1], [1.2, 1., .8, 1.1, .9], [1.] * 5])
    completion = torch.tensor([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0], [2, 2, 1, 0, 3]])
    valid = torch.tensor([[1, 1, 0, 1, 1], [1, 1, 1, 1, 0], [0, 0, 0, 0, 0]])
    full = torch.cat((torch.tensor([[0, 1], [1, 2], [2, 3]]), completion), dim=1)
    assignments = torch.tensor([[0, 0, 1, 2, 2], [0, 1, 1, 1, 2], [0, 0, 0, 0, 0]])
    rows = []
    for row, selected in enumerate([[0, 2], [1, 2], []]):
        entries = [(step, assignments[row].ge(step), assignments[row].eq(step)) for step in selected]
        entries += [(-1, torch.zeros(5, dtype=torch.bool), torch.zeros(5, dtype=torch.bool))] * (2-len(entries))
        rows.append(entries)
    model = Model(seed)
    reference = deepcopy(model).eval()
    with torch.no_grad():
        reference.projection.add_(torch.linspace(-.3, .2, 24).reshape(4, 6))
    return trainer, model, reference, MaskingSchedule(rows, "dtca"), full, completion, valid, torch.tensor([1.1, -.8, 0.])


def paper_loss(model, reference, schedule, full, completion, valid, advantages, weights, enabled, beta):
    """Scalar POS/entropy coefficients; autograd only through log probabilities and KL."""
    observations = []
    for row, entries in enumerate(schedule.masking_inputs):
        for _, masked, committed in entries:
            state = full[row:row+1].clone()
            state[0, 2:][masked] = 5
            log_distribution = model(state).logits[0, 2:].float().log_softmax(-1)
            with torch.no_grad():
                ref_distribution = reference(state).logits[0, 2:].float().log_softmax(-1)
            positions = [i for i in range(5) if valid[row, i] and
                         (committed[i] or (masked[i] and float(log_distribution[i, completion[row, i]].detach()) >= math.log(.2)))]
            mean = sum(float(weights[row, i]) for i in positions)/len(positions) if positions else 1.
            for i in positions:
                logp = log_distribution[i, completion[row, i]]
                entropy = float(-(log_distribution[i].detach().exp()*log_distribution[i].detach()).sum())
                credit = float(advantages[row]) * (float(weights[row, i])/mean if enabled else 1.)
                probability = float(logp.detach().exp())
                flow = -1e-6*credit*(1-probability)**2*(float(logp.detach())+entropy)
                observations.append(dict(row=row, logp=logp, ref=ref_distribution[i, completion[row, i]],
                                         credit=credit, flow=flow))
    corrected = [entry["credit"] for entry in observations]
    if enabled:
        for row in range(len(completion)):
            indices = [i for i, entry in enumerate(observations) if entry["row"] == row]
            up = sum(max(0., observations[i]["flow"]) for i in indices)
            down = sum(max(0., -observations[i]["flow"]) for i in indices)
            balance = max(-.95, min(.95, (down-up)/(down+up))) if up+down else 0.
            scaled = [observations[i]["credit"]*(1+balance if observations[i]["flow"] > 0 else 1-balance) for i in indices]
            if sum(scaled):
                original = sum(observations[i]["credit"] for i in indices)
                for index, value in zip(indices, scaled):
                    corrected[index] = value*original/sum(scaled)
    loss = sum((-credit*entry["logp"] + beta*((entry["ref"]-entry["logp"]).exp()-1-entry["ref"]+entry["logp"]))
               for entry, credit in zip(observations, corrected))/len(observations)
    return loss, len(observations)


@pytest.mark.parametrize("seed", [11, 23, 41])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("size", [1, 2, 3])
@pytest.mark.parametrize("beta", [0., .01])
def test_optimizer_update_matches_paper_objective(seed, enabled, size, beta):
    trainer, model, reference, schedule, full, completion, valid, advantages = example(seed, enabled, size, beta)
    oracle = deepcopy(model)
    loss, count = paper_loss(oracle, reference, schedule, full, completion, valid,
                             advantages, trainer.weights, enabled, beta)
    loss.backward()
    engine = Engine(model)
    metrics = trainer._compute_loss(model, reference, schedule, full, 2, completion, valid,
                                     None, advantages, engine, 3, torch.device("cpu"))
    assert metrics["total_mask_sum"] == count
    assert engine.steps == 1 and engine.boundaries.count(True) == 1
    assert engine.boundaries[-1]
    assert metrics["mean_loss"] == pytest.approx(float(loss.detach()), abs=2e-6)
    for actual, initial in zip(model.parameters(), oracle.parameters()):
        assert torch.allclose(actual.grad, initial.grad, atol=3e-7, rtol=3e-5)
        assert torch.allclose(actual, initial.detach()-.03*initial.grad, atol=3e-7, rtol=3e-5)
    assert all(parameter.grad is None for parameter in reference.parameters())


def test_no_targets_has_zero_loss_and_zero_gradient():
    trainer, model, reference, schedule, full, completion, valid, advantages = example()
    valid.zero_()
    before = [parameter.detach().clone() for parameter in model.parameters()]
    engine = Engine(model)
    result = trainer._compute_loss(model, reference, schedule, full, 2, completion, valid,
                                    None, advantages, engine, 3, torch.device("cpu"))
    assert result["total_mask_sum"] == 0 and result["mean_loss"] == 0
    for parameter, initial in zip(model.parameters(), before):
        assert torch.equal(parameter, initial) and parameter.grad.eq(0).all()


def test_pos_table_stays_fixed_across_policy_updates(tmp_path, monkeypatch):
    import dtca.contrib
    import dtca.pos
    from dtca.table import Table
    from rollout.llada_rollout import RolloutOutput

    trainer, model, reference, _, full, completion, valid, advantages = example()
    path = tmp_path / "pos.json"
    Table(torch.arange(18), torch.ones(18)).save(path)
    original = path.read_bytes()
    config = OmegaConf.load(ROOT / "configs/dtca.yaml")
    config.dtca.table = str(path)

    class Tagger:
        def project(self, tokenizer, ids, mask):
            return torch.arange(ids.numel()).reshape_as(ids) % 18

    def unexpected(*args, **kwargs):
        raise AssertionError("RL updates must not collect new contribution estimates")

    monkeypatch.setattr(dtca.pos, "Tagger", lambda *args: Tagger())
    monkeypatch.setattr(dtca.contrib, "reveal", unexpected)
    trainer = Trainer(config, 5)
    rollout = RolloutOutput(full[:, :2], torch.ones(3, 2), completion, valid, [], [],
                            torch.tensor([[0, 0, 1, 2, 2], [0, 1, 1, 1, 2], [0, 0, 0, 0, 0]]),
                            torch.full((3, 5), .5))
    expected = trainer.table.weights.clone()
    engine = Engine(model)
    for _ in range(2):
        engine.optimizer.zero_grad(set_to_none=True)
        schedule = trainer.build_schedule(rollout)
        trainer._compute_loss(model, reference, schedule, full, 2, completion, valid, None,
                               advantages, engine, 3, torch.device("cpu"))
        assert torch.equal(trainer.table.weights, expected)
        assert path.read_bytes() == original
    assert engine.steps == 2

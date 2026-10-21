from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from trainer.dtca import Trainer
from trainer.masking_strategy import MaskingSchedule

ROOT = Path(__file__).resolve().parents[1]


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor([.2, -.3, .1]))

    def forward(self, ids, attention_mask=None, use_cache=False):
        return SimpleNamespace(logits=self.logits.expand(*ids.shape, 3))


class Engine:
    def __init__(self, model):
        self.optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        self.boundaries, self.steps = [], 0

    def set_gradient_accumulation_boundary(self, flag):
        self.boundaries.append(flag)

    def backward(self, loss):
        loss.backward()

    def step(self):
        self.optimizer.step()
        self.steps += 1


def run(size, enabled=False, dummy=False):
    config = OmegaConf.load(ROOT / "configs/dtca.yaml")
    config.dtca.enabled = False
    config.training.mask_strategy = "sequential_masking"
    config.training.micro_batch_size = size
    config.training.beta = 0
    trainer = Trainer(config, mask_id=2)
    trainer.enabled = enabled
    trainer.weights = torch.tensor([[.8, 1.2], [1.2, .8]])
    model = Tiny()
    engine = Engine(model)
    mask = torch.ones(2, dtype=torch.bool)
    entries = [[(0, mask.clone(), mask.clone())] for _ in range(2)]
    if dummy:
        for row in entries:
            row.append((-1, ~mask, ~mask))
    schedule = MaskingSchedule(entries, "dtca" if enabled else "sequential_masking")
    result = trainer._compute_loss(model, None, schedule, torch.tensor([[0, 0, 0], [0, 1, 0]]), 1,
                                   torch.tensor([[0, 0], [1, 0]]), torch.ones(2, 2),
                                   None, torch.tensor([1., -1.]), engine, 2, torch.device("cpu"))
    return model.logits.detach(), engine, result


@pytest.mark.parametrize("enabled", [False, True])
def test_chunk_size_and_dummy_steps_leave_update_unchanged(enabled):
    large, _, _ = run(2, enabled)
    small, engine, result = run(1, enabled, dummy=True)
    assert torch.allclose(large, small, atol=1e-7)
    assert engine.steps == 1
    assert engine.boundaries == [False, False, False, True]
    assert result["total_mask_sum"] == 4


def test_uniform_rldf_gradient_matches_manual_objective():
    updated, _, _ = run(1)
    model = Tiny()
    logps = model.logits.log_softmax(-1)
    objective = -(logps[torch.tensor([[0, 0], [1, 0]])] * torch.tensor([[1.], [-1.]])).sum() / 4
    objective.backward()
    assert model.logits.grad.abs().sum() > 0
    assert torch.allclose(updated, model.logits.detach() - .1*model.logits.grad)


def test_only_dtca_config_and_uniform_credit_has_no_pos_dependency():
    assert {p.name for p in (ROOT / "configs").rglob("*.yaml")} == {"dtca.yaml"}
    config = OmegaConf.load(ROOT / "configs/dtca.yaml")
    assert config.training.num_iterations == 1
    assert list(config.evaluation.steps) == [128, 256, 512]
    assert list(config.evaluation.steps) == list(config.evaluation.max_gen_length)
    config.dtca.enabled = False
    config.training.mask_strategy = "sequential_masking"
    trainer = Trainer(config, mask_id=2)
    assert not hasattr(trainer, "table")
    assert not hasattr(trainer, "tagger")


def test_missing_pos_table_fails_before_spacy_or_model_loading(tmp_path):
    config = OmegaConf.load(ROOT / "configs/dtca.yaml")
    config.dtca.table = str(tmp_path / "missing.json")
    with pytest.raises(FileNotFoundError):
        Trainer(config, mask_id=2)


def test_complete_reward_path_uses_padding_mask_and_frozen_reference(monkeypatch):
    from accelerate import Accelerator
    from rollout.llada_rollout import RolloutOutput
    import trainer.main_rl as upstream

    Accelerator(cpu=True)
    config = OmegaConf.load(ROOT / "configs/dtca.yaml")
    config.dtca.enabled = False
    config.training.mask_strategy = "sequential_masking"
    config.rollout.num_generations = 2
    trainer = Trainer(config, mask_id=2)
    observed = []

    class Checked(Tiny):
        def forward(self, ids, attention_mask=None, use_cache=False):
            assert torch.equal(attention_mask, torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1]])[len(observed) % 2:len(observed) % 2 + 1])
            observed.append(attention_mask.clone())
            return super().forward(ids, attention_mask, use_cache)

    # Use one chunk for each response to exercise masking in both passes.
    config.training.micro_batch_size = 1
    model, reference = Checked(), Tiny()
    reference.logits.data.add_(torch.tensor([.4, -.2, .1]))
    engine = Engine(model)
    accelerator = SimpleNamespace(deepspeed_engine_wrapped=SimpleNamespace(engine=engine))
    monkeypatch.setattr(upstream, "compute_reward_per_func", lambda **kwargs: torch.tensor([[1.], [0.]]))
    rollout = RolloutOutput(prompt_ids=torch.tensor([[0, 0], [1, 0]]),
                            prompt_mask=torch.tensor([[0, 1], [1, 1]]),
                            completion_ids=torch.tensor([[0, 0], [1, 0]]),
                            completion_mask=torch.ones(2, 2), completions=["a", "b"],
                            ground_truths=["a", "a"], step_map=torch.zeros(2, 2, dtype=torch.long),
                            token_probs=torch.full((2, 2), .5))
    mask = torch.ones(2, dtype=torch.bool)
    schedule = MaskingSchedule([[(0, mask, mask)], [(0, mask, mask)]], "sequential_masking")
    initial = model.logits.detach().clone()
    result = trainer.compute_loss(model, reference, rollout, schedule, accelerator=accelerator)
    assert engine.steps == 1 and len(observed) == 4
    assert torch.equal(result["rewards"], torch.tensor([1., 0.]))
    assert result["advantages"][0] > 0 > result["advantages"][1]
    assert result["mean_kl"] > 0
    assert reference.logits.grad is None
    assert not torch.equal(model.logits, initial)
    assert trainer.attention is None


def test_config_matches_released_paper_training_and_evaluation_protocol():
    settings = OmegaConf.load(ROOT / "configs/dtca.yaml")
    assert settings.model.model_type == "llada"
    assert settings.model.pretrained_model.endswith("LLaDA-8B-Instruct")
    assert settings.training.mixed_precision == "bf16"
    assert settings.optimizer.name == "adamw"
    assert settings.optimizer.params.learning_rate == 1e-6
    assert settings.optimizer.params.weight_decay == 0
    assert settings.lr_scheduler.scheduler == "constant"
    assert settings.training.num_iterations == 1
    assert settings.training.entropy_coeff == 0
    assert settings.training.loss_norm == "token"
    assert settings.training.low_prob_threshold == .2
    assert settings.training.beta == .01 and settings.training.kl_type == "k3"
    assert settings.rollout.num_generations == 4
    assert settings.rollout.max_gen_length == settings.rollout.steps == 256
    for decoding in (settings.rollout, settings.evaluation):
        assert decoding.block_size == 32
        assert decoding.target == "confidence" and decoding.dynamic_threshold == .9
        assert decoding.remasking_strategy == "low_confidence_dynamic" and decoding.use_cache
    assert settings.evaluation.temperature == 0
    assert list(settings.evaluation.eval_dataset) == ["MATH500", "gsm8k_test"]
    assert list(settings.evaluation.steps) == list(settings.evaluation.max_gen_length) == [128, 256, 512]
    assert settings.evaluation.eval_every_steps == 5
    assert settings.evaluation.eval_at_start and settings.evaluation.save_after_eval
    assert settings.evaluation.top_k_checkpoints == 1
    assert settings.evaluation.top_k_metric == "eval_accuracy"
    assert settings.dtca.enabled and settings.dtca.mass == .7
    assert settings.dtca.min_steps == 2 and settings.dtca.max_steps == 32
    assert settings.dtca.pos_model == "en_core_web_sm"
    assert {row.name for row in settings.offline.datasets} == {"MATH", "GSM8K"}
    assert all(row.split == "train" for row in settings.offline.datasets)


def test_training_entry_preserves_evaluation_controls(monkeypatch):
    import train
    from common import config

    path = str(ROOT / "configs/dtca.yaml")
    for overrides, interval, initial, save in (
        ([], 5, True, True),
        (["evaluation.eval_every_steps=7", "evaluation.eval_at_start=false",
          "evaluation.save_after_eval=false"], 7, False, False),
    ):
        settings = config(SimpleNamespace(config=path, set=overrides))
        monkeypatch.setattr(train, "config", lambda args: settings)
        monkeypatch.setattr("sys.argv", ["train.py", "--config", path])
        received = []
        monkeypatch.setattr(train, "upstream_main", lambda module, value: received.append((module, value)))
        train.main()
        module, active = received[0]
        assert module == "trainer.main_rl"
        assert active.evaluation.eval_dataset == "MATH500"
        assert active.evaluation.steps == active.evaluation.max_gen_length == 256
        assert active.evaluation.eval_every_steps == interval
        assert active.evaluation.eval_at_start is initial
        assert active.evaluation.save_after_eval is save
        assert active.evaluation.top_k_checkpoints == 1
        unchanged = OmegaConf.load(path)
        assert list(unchanged.evaluation.eval_dataset) == ["MATH500", "gsm8k_test"]
        assert list(unchanged.evaluation.steps) == [128, 256, 512]

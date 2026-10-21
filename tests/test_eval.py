from pathlib import Path
from types import SimpleNamespace

from accelerate import Accelerator
from jinja2 import Template
from omegaconf import OmegaConf
import pytest
import torch

import trainer.eval as evaluation

ROOT = Path(__file__).resolve().parents[1]


def test_single_process_evaluation_counts_every_question_once(monkeypatch):
    Accelerator(cpu=True)
    settings = OmegaConf.load(ROOT / "configs/dtca.yaml")
    monkeypatch.setattr(evaluation, "load_eval_data", lambda *args: ([str(i) for i in range(5)], ["2"] * 5))
    monkeypatch.setattr(evaluation, "load_rollout_model", lambda *args: torch.nn.Linear(2, 2))
    visited = []

    def generate(**kwargs):
        visited.extend(kwargs["prompts"])
        assert kwargs["num_generations"] == 1
        config = kwargs["config"].rollout
        assert config.temperature == 0 and config.steps == config.max_gen_length == 128
        return SimpleNamespace(completions=[r"\boxed{2}" if int(p) % 2 == 0 else r"\boxed{0}" for p in kwargs["prompts"]],
                               ground_truths=kwargs["ground_truths"])

    def unexpected(*args, **kwargs):
        raise AssertionError("No collectives are needed for single-process evaluation")

    monkeypatch.setattr(evaluation, "run_rollout", generate)
    monkeypatch.setattr(evaluation.dist, "all_reduce", unexpected)
    monkeypatch.setattr(evaluation.dist, "barrier", unexpected)
    accelerator = SimpleNamespace(num_processes=1, process_index=0, device=torch.device("cpu"), is_local_main_process=True)
    result = evaluation.run_single_eval(accelerator, settings, None, "tiny", None, "llada", torch.float32,
                                         5, "MATH500", "math", 128, 128, 2, ["math"])
    assert visited == ["0", "1", "2", "3", "4"]
    assert result["num_samples"] == 5 and result["accuracy"] == pytest.approx(.6)


@pytest.mark.parametrize("contents", [None, "[]"])
def test_missing_or_empty_evaluation_split_is_rejected(tmp_path, monkeypatch, contents):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    if contents is not None:
        (tmp_path / "data/MATH500.json").write_text(contents)
    with pytest.raises(FileNotFoundError if contents is None else ValueError):
        evaluation.load_eval_data("MATH500", "math", Template("{{ problem }}"))


def test_training_monitor_scores_current_policy_and_logs_accuracy(tmp_path, monkeypatch):
    import json
    import logging
    import trainer.main_rl as training

    settings = OmegaConf.load(ROOT / "configs/dtca.yaml")
    settings.evaluation.eval_dataset = "MATH500"
    settings.evaluation.steps = settings.evaluation.max_gen_length = 256
    settings.evaluation.batch_size = 2
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data/MATH500.json").write_text(json.dumps([
        {"question": str(i), "ground_truth_answer": "2"} for i in range(5)
    ]))
    model = torch.nn.Linear(2, 2).eval()
    accelerator = SimpleNamespace(num_processes=1, process_index=0, device=torch.device("cpu"),
                                  is_local_main_process=False, is_main_process=True)
    monkeypatch.setattr(training, "logger", logging.getLogger("training_eval"))
    loaded, visited = [], []

    def load(current, active_accelerator, path, **kwargs):
        assert current is model and active_accelerator is accelerator
        loaded.append(current)
        return current

    def generate(**kwargs):
        assert kwargs["model"] is model and kwargs["num_generations"] == 1
        rollout = kwargs["config"].rollout
        assert rollout.steps == rollout.max_gen_length == 256 and rollout.temperature == 0
        questions = kwargs["prompts"]
        start = len(visited)
        visited.extend(questions)
        return SimpleNamespace(
            completions=[r"\boxed{2}" if (start + i) % 2 == 0 else r"\boxed{0}"
                         for i in range(len(questions))], ground_truths=kwargs["ground_truths"])

    monkeypatch.setattr(training, "load_rollout_model", load)
    monkeypatch.setattr(training, "run_rollout", generate)
    monkeypatch.setattr(training.dist, "all_reduce", lambda *args, **kwargs: None)
    monkeypatch.setattr(training.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(training.torch.cuda, "synchronize", lambda: None)
    path = tmp_path / "metrics.jsonl"
    result = training.run_evaluation(model, accelerator, settings, None, "tiny", 5, 1, path, 5,
                                     model_dtype=torch.float32)
    assert len(loaded) == 1 and len(visited) == len(set(visited)) == 5
    assert result["eval_num_samples"] == 5 and result["eval_accuracy"] == pytest.approx(.6)
    record = json.loads(path.read_text())
    assert record["global_step"] == 5 and record["datasets"] == ["MATH500"]
    assert record["eval_accuracy"] == pytest.approx(.6)

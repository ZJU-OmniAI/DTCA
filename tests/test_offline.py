import json
from pathlib import Path
import sys
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest
import torch

import build_table
from dtca.pos import IDS
from dtca.table import Table

ROOT = Path(__file__).resolve().parents[1]


def settings_for(tmp_path):
    settings = OmegaConf.load(ROOT / "configs/dtca.yaml")
    for spec in settings.offline.datasets:
        path = tmp_path / f"{spec.name}_train.json"
        path.write_text(json.dumps([{"question": "Compute 1+1"}]), encoding="utf-8")
        spec.path = str(path)
    settings.dtca.table = str(tmp_path / "pos.json")
    settings.offline.records = str(tmp_path / "records")
    settings.offline.length = 4
    settings.offline.responses = 1
    return settings


def test_train_splits_only(tmp_path):
    settings = settings_for(tmp_path)
    examples, sources = build_table.training_data(settings)
    assert len(examples) == 2 and {s["name"] for s in sources} == {"MATH", "GSM8K"}
    settings.offline.datasets[0].split = "test"
    with pytest.raises(ValueError, match="TRAIN"):
        build_table.training_data(settings)


def test_collection_table_and_resume_from_response_records(tmp_path, monkeypatch):
    import accelerate
    import dtca.contrib
    import dtca.pos
    import rollout.llada
    import trainer.model_utils

    settings = settings_for(tmp_path)
    monkeypatch.setattr(build_table, "config", lambda args: settings)
    monkeypatch.setattr(sys, "argv", ["build_table.py"])

    class Worker:
        is_main_process = True
        process_index = 0
        num_processes = 1
        device = torch.device("cpu")

        def __init__(self, **kwargs):
            pass

        def wait_for_everyone(self):
            pass

    class Model(torch.nn.Module):
        def forward(self, ids, use_cache=False):
            n = ids[:, 1:].ne(3).sum(1).float()
            logits = torch.zeros(*ids.shape, 4)
            logits[..., 1] = (1 + n)[:, None]
            logits[..., 3] = -4
            return SimpleNamespace(logits=logits)

    class Tokenizer:
        all_special_ids = [3]

        def __call__(self, text, **kwargs):
            return {"input_ids": torch.tensor([[0]])}

    class Tags:
        def project(self, tokenizer, tokens, valid):
            return torch.full_like(tokens, IDS["X"])

    monkeypatch.setattr(accelerate, "Accelerator", Worker)
    monkeypatch.setattr(trainer.model_utils, "load_tokenizer", lambda *args: (Tokenizer(), 3))
    monkeypatch.setattr(rollout.llada.LLaDAModelLM, "from_pretrained", lambda *args, **kwargs: Model())
    monkeypatch.setattr(dtca.pos, "Tagger", lambda *args: Tags())
    build_table.main()
    table = Table.load(settings.dtca.table)
    assert table.as_dict()["counts"][IDS["X"]] == 8
    assert table.as_dict()["sums"][IDS["X"]] > 0
    original = table.as_dict()
    Path(settings.dtca.table).unlink()

    def unexpected(*args, **kwargs):
        raise AssertionError("Completed response records must be reused")

    monkeypatch.setattr(dtca.contrib, "reveal", unexpected)
    build_table.main()
    assert Table.load(settings.dtca.table).as_dict() == original
    settings.offline.temperature = .5
    with pytest.raises(ValueError, match="different collection"):
        build_table.main()


def record():
    row = {"example": {"dataset": "MATH", "index": 0, "question": "Compute 1+1"}, "response": 0,
           "tokens": [1, 2, 1, 2], "pos": ["VERB", "NUM", "VERB", "NUM"],
           "before": [1.1, 1.1, .4, .4], "after": [1.3, 1.05, .45, .4],
           "contribution": [.2, 0., .05, 0.], "remaining": [3, 3, 1, 1],
           "sums": [0.] * 18, "counts": [0] * 18}
    row["sums"][IDS["VERB"]] = .25
    row["counts"][IDS["VERB"]] = row["counts"][IDS["NUM"]] = 2
    return row


def test_record_recomputes_zero_inclusive_pos_statistics():
    row = record()
    sums, counts = build_table.record_stats(row, row["example"], 0, 4, [3])
    assert sums[IDS["VERB"]] == pytest.approx(.25)
    assert counts[IDS["NUM"]] == 2 and sums[IDS["NUM"]] == 0


@pytest.mark.parametrize("field,index,value", [
    ("contribution", 0, .2/3), ("after", 0, float("nan")), ("before", 0, 4.),
    ("remaining", 0, 0), ("pos", 0, "INVALID"), ("tokens", 0, -1),
    ("sums", IDS["VERB"], .3), ("counts", IDS["NUM"], 0),
])
def test_invalid_records_cannot_silently_change_fixed_table(field, index, value):
    row = record()
    row[field][index] = value
    with pytest.raises(ValueError):
        build_table.record_stats(row, row["example"], 0, 4, [3])


@pytest.mark.parametrize("name", ["MATH500", "gsm8k_test"])
def test_holdout_filenames_are_rejected_even_if_split_is_mislabeled(tmp_path, name):
    settings = settings_for(tmp_path)
    settings.offline.datasets[0].path = str(tmp_path / f"{name}.json")
    with pytest.raises(ValueError, match="TRAIN"):
        build_table.training_data(settings)

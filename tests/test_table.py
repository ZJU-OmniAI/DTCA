import json

import pytest
import torch

from dtca.pos import IDS, TAGS, aggregate
from dtca.table import Table


def test_pos_average_counts_zero_effects_without_dividing_by_mask_count():
    sums, counts = aggregate(torch.tensor([[0.0, 6.0, 2.0]]),
                             torch.tensor([[IDS["NOUN"], IDS["NOUN"], IDS["VERB"]]]),
                             torch.ones(1, 3, dtype=torch.bool))
    table = Table(sums, counts)
    assert table.means[IDS["NOUN"]] == 3
    assert counts[IDS["NOUN"]] == 2
    assert table.weights.max() == pytest.approx(1.2)
    assert table.weights.min() == pytest.approx(0.8)
    assert len(table.weights) == 18


def test_equal_means_unit_weights_and_returned_weights_are_detached_copies():
    table = Table(torch.ones(18), torch.ones(18))
    assert table.weights.eq(1).all()
    table.weights[0] = 9
    assert table.weights.eq(1).all()


def test_save_load_rejects_mean_metric_and_tampered_weights(tmp_path):
    table = Table(torch.arange(18), torch.ones(18), {"model": "tiny"})
    path = tmp_path / "pos.json"
    table.save(path)
    assert Table.load(path).as_dict() == table.as_dict()
    payload = json.loads(path.read_text())
    payload["reduction"] = "mean"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="TOTAL"):
        Table.load(path)
    table.save(path)
    payload = json.loads(path.read_text())
    payload["weights"][0] = 42
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="weights"):
        Table.load(path)


@pytest.mark.parametrize("counts", [torch.full((18,), -1), torch.full((18,), 0.5)])
def test_invalid_counts_fail(counts):
    with pytest.raises(ValueError):
        Table(torch.zeros(18), counts)

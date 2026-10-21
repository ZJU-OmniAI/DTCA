"""Fixed POS weights constructed offline."""

import json
import os
from pathlib import Path

import torch

from dtca.pos import TAGS

METRIC = {"metric": "positive_total_confidence_change", "is_individual_reveal": True, "reduction": "sum", "probability": "max_over_vocabulary"}


class Table:
    def __init__(self, sums, counts, metadata=None):
        self._sums = torch.as_tensor(sums, dtype=torch.float64).cpu().clone()
        raw = torch.as_tensor(counts, dtype=torch.float64).cpu()
        if self._sums.shape != (len(TAGS),) or raw.shape != self._sums.shape:
            raise ValueError("A POS table must contain the complete 18-tag inventory")
        if not torch.isfinite(self._sums).all() or (self._sums < 0).any():
            raise ValueError("Invalid POS contribution sums")
        if not torch.isfinite(raw).all() or (raw < 0).any() or not raw.eq(raw.round()).all():
            raise ValueError("POS counts must be nonnegative integers")
        if self._sums[raw.eq(0)].ne(0).any():
            raise ValueError("Unobserved POS categories must have zero sums")
        self._counts = raw.long().clone()
        self.metadata = dict(metadata or {})

    @property
    def means(self):
        return self._sums / self._counts.clamp_min(1)

    @property
    def weights(self):
        means = self.means
        span = means.max() - means.min()
        return torch.ones_like(means) if span == 0 else 0.8 + 0.4 * (means - means.min()) / span

    def lookup(self, ids):
        return self.weights.to(device=ids.device, dtype=torch.float32)[ids.long()]

    def as_dict(self):
        return {**METRIC, "tags": list(TAGS), "sums": self._sums.tolist(), "counts": self._counts.tolist(), "weights": self.weights.tolist(), "metadata": self.metadata}

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(self.as_dict(), indent=2) + "\n", encoding="utf-8")
        os.replace(temp, path)

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if any(type(data.get(k)) is not type(v) or data[k] != v for k, v in METRIC.items()) or data.get("tags") != list(TAGS):
            raise ValueError("POS table must use individual reveal and TOTAL maximum-confidence change")
        table = cls(data["sums"], data["counts"], data.get("metadata"))
        if not table._counts.sum():
            raise ValueError("POS table contains no observations")
        saved = torch.tensor(data["weights"], dtype=torch.float64)
        if saved.shape != table.weights.shape or not torch.allclose(saved, table.weights, rtol=1e-12, atol=1e-12):
            raise ValueError("Saved weights do not match POS statistics")
        return table

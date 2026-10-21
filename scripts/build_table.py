"""Build fixed POS weights from isolated reveals on MATH/GSM8K train."""

import hashlib
import json
import os
from pathlib import Path

from common import config, parser


def training_data(settings, limit=0):
    """Validate both train sources before loading any model."""
    specs = settings.offline.datasets
    if len(specs) != 2 or {s.name for s in specs} != {"MATH", "GSM8K"}:
        raise ValueError("Provide both MATH and GSM8K training datasets")
    examples, sources = [], []
    for spec in specs:
        path = Path(spec.path)
        if spec.split != "train" or "test" in path.stem.lower() or "math500" in path.stem.lower():
            raise ValueError("POS construction accepts TRAIN splits only")
        content = path.read_bytes()
        rows = json.loads(content)
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"Expected nonempty training JSON: {path}")
        chosen = rows[:limit] if limit else rows
        for index, row in enumerate(chosen):
            if not isinstance(row.get("question"), str) or not row["question"].strip():
                raise ValueError(f"Missing question in {path}:{index}")
            examples.append({"dataset": spec.name, "index": index, "question": row["question"]})
        sources.append({"name": spec.name, "split": "train", "sha256": hashlib.sha256(content).hexdigest(), "prompts": len(chosen)})
    return examples, sources


def record_stats(row, example, response, length, special_ids):
    """Recompute POS statistics from validated individual-intervention records."""
    import torch
    from dtca.pos import IDS, aggregate

    if row.get("example") != example or row.get("response") != response:
        raise ValueError("Response record does not match its training example")
    fields = ("tokens", "pos", "before", "after", "contribution", "remaining")
    if any(not isinstance(row.get(key), list) or len(row[key]) != length for key in fields):
        raise ValueError("Response record has missing or misaligned token observations")
    if any(type(token) is not int or token < 0 for token in row["tokens"]):
        raise ValueError("Invalid response token IDs")
    if any(tag not in IDS for tag in row["pos"]):
        raise ValueError("Invalid response POS category")
    if any(type(count) is not int for count in row["remaining"]) or sorted(row["remaining"]) != sorted(list(range(1, length, 2))*2):
        raise ValueError("Response record must contain two individual probes at each denoising step")
    before, after, values = (torch.tensor(row[key], dtype=torch.float64)[None]
                             for key in ("before", "after", "contribution"))
    remaining = torch.tensor(row["remaining"])[None]
    if any(not torch.isfinite(tensor).all() or (tensor < 0).any() for tensor in (before, after, values)):
        raise ValueError("Response confidence observations must be finite and nonnegative")
    if (before > remaining + 1e-6).any() or (after > remaining + 1e-6).any():
        raise ValueError("Total confidence cannot exceed the remaining position count")
    if not torch.allclose(values, (after-before).clamp_min(0), atol=1e-10, rtol=1e-9):
        raise ValueError("Response contribution must equal the positive TOTAL confidence change")
    tokens = torch.tensor(row["tokens"])[None]
    valid = torch.ones_like(tokens, dtype=torch.bool)
    for token in special_ids:
        valid &= tokens.ne(token)
    pos = torch.tensor([[IDS[tag] for tag in row["pos"]]])
    sums, counts = aggregate(values, pos, valid)
    saved_sums, saved_counts = (torch.as_tensor(row[key], dtype=torch.float64) for key in ("sums", "counts"))
    if saved_sums.shape != sums.shape or saved_counts.shape != counts.shape:
        raise ValueError("Response POS statistics must contain all 18 categories")
    if not torch.allclose(saved_sums.double(), sums, atol=1e-10, rtol=1e-7) or not torch.equal(saved_counts.double(), counts.double()):
        raise ValueError("Response POS statistics disagree with token observations")
    return sums, counts


def main():
    cli = parser(__doc__)
    cli.add_argument("--limit", type=int, default=0, help="Prompts per dataset; zero uses all")
    cli.add_argument("--output", help="Override POS table output (also isolates response records)")
    args = cli.parse_args()
    settings = config(args)
    if not settings.dtca.enabled:
        cli.error("RLDF does not construct or load a POS table")
    if args.limit < 0 or (args.limit and not args.output):
        cli.error("Use --limit >= 0; limited collections require a separate --output")
    import torch
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from jinja2 import Template
    from dtca.contrib import reveal
    from dtca.pos import Tagger, aggregate, TAGS
    from dtca.table import Table, METRIC

    examples, sources = training_data(settings, args.limit)
    offline = settings.offline
    if offline.length < 2 or offline.length % 2 or offline.responses < 1:
        raise ValueError("Offline length must be even and responses must be positive")
    output = Path(args.output or settings.dtca.table)
    records = output.with_suffix("") / "records" if args.output else Path(offline.records)
    identity = {**METRIC, "model": settings.model.pretrained_model, "seed": settings.training.seed,
                "length": offline.length, "temperature": offline.temperature,
                "responses": offline.responses, "sources": sources, "pos_model": settings.dtca.pos_model}
    if output.exists():
        if Table.load(output).metadata != identity:
            raise ValueError("Existing table belongs to a different collection; choose another output")
        print(f"Validated existing table: {output}")
        return
    accelerator = Accelerator(mixed_precision=settings.training.mixed_precision)
    records.mkdir(parents=True, exist_ok=True)
    identity_path = records / "collection.json"
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != identity:
        raise ValueError("Response records belong to different data/model/settings")
    if accelerator.is_main_process:
        identity_path.write_text(json.dumps(identity, indent=2), encoding="utf-8")
    accelerator.wait_for_everyone()
    from trainer.model_utils import load_tokenizer, resolve_model_dtype
    from rollout.llada import LLaDAModelLM
    from trainer.prompts import get_default_system_prompt

    tokenizer, mask_id = load_tokenizer(settings.model.pretrained_model, "llada")
    model = LLaDAModelLM.from_pretrained(settings.model.pretrained_model,
                                        torch_dtype=resolve_model_dtype(settings)).to(accelerator.device).eval()
    tagger = Tagger(settings.dtca.pos_model)
    template = Template(get_default_system_prompt("llada", "math"))
    for index in range(accelerator.process_index, len(examples), accelerator.num_processes):
        example = examples[index]
        prompt = template.render(problem=example["question"])
        ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(accelerator.device)
        for response in range(offline.responses):
            path = records / f"{index:08d}_{response:03d}.json"
            if path.exists():
                saved = json.loads(path.read_text(encoding="utf-8"))
                record_stats(saved, example, response, offline.length, tokenizer.all_special_ids)
                continue
            set_seed(settings.training.seed + index * offline.responses + response)
            observation = reveal(model, ids, length=offline.length, temperature=offline.temperature, mask_id=mask_id)
            valid = observation.counts.gt(0)
            for special in tokenizer.all_special_ids:
                valid &= observation.tokens.ne(special)
            pos_ids = tagger.project(tokenizer, observation.tokens, valid)
            sums, counts = aggregate(observation.values, pos_ids, valid)
            record = {"example": example, "response": response, "sums": sums.tolist(), "counts": counts.tolist(),
                      "tokens": observation.tokens[0].tolist(), "pos": [TAGS[p] for p in pos_ids[0].tolist()],
                      "before": observation.before[0].tolist(), "after": observation.after[0].tolist(),
                      "contribution": observation.values[0].tolist(), "remaining": observation.counts[0].tolist()}
            record_stats(record, example, response, offline.length, tokenizer.all_special_ids)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(record), encoding="utf-8")
            os.replace(temporary, path)
        print(f"rank={accelerator.process_index} prompt={index + 1}/{len(examples)}", flush=True)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        sums = torch.zeros(len(TAGS), dtype=torch.float64)
        counts = torch.zeros(len(TAGS), dtype=torch.long)
        for index, example in enumerate(examples):
            for response in range(offline.responses):
                path = records / f"{index:08d}_{response:03d}.json"
                row = json.loads(path.read_text(encoding="utf-8"))
                part_sums, part_counts = record_stats(row, example, response, offline.length, tokenizer.all_special_ids)
                sums += part_sums
                counts += part_counts
        table = Table(sums, counts, identity)
        table.save(output)
        Table.load(output)
        print(f"Saved fixed POS table: {output}")
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()

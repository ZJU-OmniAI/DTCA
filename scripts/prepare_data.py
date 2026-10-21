"""Download public math datasets in the Drift JSON format."""

import argparse
import json
import os
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "787d875d7e992e92f883049cef58bcd49ef34844"


def write(path, rows):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", default=str(ROOT / "data"))
    args = cli.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for filename in ("MATH_rldf.json", "MATH500.json", "gsm8k_test.json"):
        path = output / filename
        if path.exists():
            print(f"Keep {path}")
            continue
        url = f"https://raw.githubusercontent.com/ant-research/Drift/{COMMIT}/data/{filename}"
        with urlopen(url, timeout=120) as response:
            rows = json.load(response)
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"Invalid dataset: {filename}")
        write(path, rows)
        print(f"Downloaded {filename}: {len(rows)} questions")
    path = output / "GSM8K_train.json"
    if not path.exists():
        from datasets import load_dataset

        rows = load_dataset("openai/gsm8k", "main", split="train")
        write(path, [{"question": r["question"], "ground_truth_answer": r["answer"].rsplit("####", 1)[-1].strip()} for r in rows])
        print(f"Downloaded GSM8K train: {len(rows)} questions")


if __name__ == "__main__":
    main()

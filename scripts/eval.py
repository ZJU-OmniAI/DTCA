"""Evaluate a model on MATH500/GSM8K at paired generation budgets."""

from common import ROOT, config, parser, upstream_main


def main():
    cli = parser(__doc__)
    cli.add_argument("--checkpoint", required=True)
    cli.add_argument("--output")
    args = cli.parse_args()
    from pathlib import Path

    checkpoint = str(Path(args.checkpoint).resolve()) if Path(args.checkpoint).exists() else args.checkpoint
    settings = config(args)
    if list(settings.evaluation.steps) != list(settings.evaluation.max_gen_length):
        raise ValueError("Evaluation steps and lengths must be paired")
    for name in settings.evaluation.eval_dataset:
        if not (ROOT / "data" / f"{name}.json").is_file():
            raise FileNotFoundError(f"Missing evaluation dataset: data/{name}.json")
    settings.model.pretrained_model = checkpoint
    settings.experiment.project = args.output or str(Path(settings.experiment.project) / "eval")
    upstream_main("trainer.eval", settings)


if __name__ == "__main__":
    main()

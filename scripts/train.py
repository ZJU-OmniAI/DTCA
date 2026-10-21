"""Train DTCA with the Drift training framework."""

from common import config, parser, upstream_main


def main():
    args = parser(__doc__).parse_args()
    settings = config(args)
    # Match Drift's MATH500/256 training monitor; scripts/eval.py uses the
    # full dataset and budget lists from the same configuration.
    settings.evaluation.steps = 256
    settings.evaluation.max_gen_length = 256
    settings.evaluation.eval_dataset = "MATH500"
    upstream_main("trainer.main_rl", settings)


if __name__ == "__main__":
    main()

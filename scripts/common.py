"""Shared CLI configuration. Run from any working directory."""

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("USE_TF", "0")


def parser(description):
    result = argparse.ArgumentParser(description=description)
    result.add_argument("--config", default="configs/dtca.yaml")
    result.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="OmegaConf overrides")
    return result


def config(args):
    from omegaconf import OmegaConf

    path = Path(args.config).resolve()
    if not path.exists():
        path = ROOT / args.config
    result = OmegaConf.merge(OmegaConf.load(path), OmegaConf.from_dotlist(args.set))
    os.chdir(ROOT)
    if result.model.model_type != "llada":
        raise ValueError("The released DTCA configuration targets LLaDA")
    if result.training.num_iterations != 1 or result.training.loss_norm != "token":
        raise ValueError("Use one policy iteration and token loss normalization")
    if result.training.entropy_coeff != 0:
        raise ValueError("Use entropy_coeff=0")
    if not 0 <= result.training.low_prob_threshold <= 1:
        raise ValueError("Training target threshold must be in [0,1]")
    if result.dtca.enabled:
        if not (0 < result.dtca.mass <= 1 and 1 <= result.dtca.min_steps <= result.dtca.max_steps):
            raise ValueError("Invalid DTCA selection bounds")
    elif result.training.mask_strategy != "sequential_masking":
        raise ValueError("RLDF uses native sequential masking")
    return result


def upstream_main(module, settings):
    # Drift's get_config consumes config=PATH and key=value arguments.
    from omegaconf import OmegaConf
    import importlib

    loaded = importlib.import_module(module)
    loaded.get_config = lambda: OmegaConf.create(OmegaConf.to_container(settings, resolve=True))
    loaded.main()

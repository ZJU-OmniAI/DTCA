"""Real two-process CPU collectives, including a rank with no active targets."""

from copy import deepcopy
from datetime import timedelta
from pathlib import Path
import socket
import time
from types import SimpleNamespace
from unittest.mock import patch
import logging

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from test_objective import Engine, example, paper_loss


def worker(rank, port, output, enabled):
    from dtca.select import schedule as select_schedule
    from trainer.masking_strategy import _sync_max_steps

    torch.set_num_threads(1)
    store = dist.TCPStore("127.0.0.1", port, 2, rank == 0, timeout=timedelta(seconds=45), use_libuv=False)
    dist.init_process_group("gloo", store=store, rank=rank, world_size=2, timeout=timedelta(seconds=45))
    try:
        assert _sync_max_steps(3 if rank == 0 else 0) == 3
        steps = torch.arange(4)[None]
        valid_steps = torch.full((1, 4), rank == 0)
        selected = select_schedule(steps, valid_steps, torch.tensor([[.5, .6, .7, .8]]), torch.ones(1, 4))
        assert len(selected.masking_inputs[0]) == 3
        if rank == 1:
            assert all(step == -1 for step, _, _ in selected.masking_inputs[0])
        trainer, model, reference, schedule, full, completion, valid, advantages = example(enabled=enabled)
        if rank == 1:
            valid.zero_()

        class Averaged(Engine):
            def step(self):
                for parameter in self.model.parameters():
                    dist.all_reduce(parameter.grad)
                    parameter.grad.div_(2)
                super().step()

        engine = Averaged(model)
        result = trainer._compute_loss(model, reference, schedule, full, 2, completion, valid,
                                        None, advantages, engine, 3, torch.device("cpu"))
        # One question across two ranks also exercises the evaluation empty shard.
        import trainer.eval as evaluation
        accelerator = SimpleNamespace(num_processes=2, process_index=rank,
                                       device=torch.device("cpu"), is_local_main_process=False)
        generated = SimpleNamespace(completions=[r"\boxed{2}"], ground_truths=["2"])
        with patch.object(evaluation, "logger", logging.getLogger("cpu_eval")), \
             patch.object(evaluation, "load_eval_data", return_value=(["question"], ["2"])), \
             patch.object(evaluation, "load_rollout_model", return_value=model), \
             patch.object(evaluation, "run_rollout", return_value=generated):
            evaluated = evaluation.run_single_eval(accelerator, trainer.config, None, "tiny", None,
                                                     "llada", torch.float32, 5, "MATH500", "math",
                                                     128, 128, 2, ["math"])
        torch.save({"parameters": [p.detach().clone() for p in model.parameters()],
                    "gradients": [p.grad.clone() for p in model.parameters()],
                    "count": result["total_mask_sum"], "updates": engine.steps,
                    "evaluated": evaluated["num_samples"], "accuracy": evaluated["accuracy"]},
                   Path(output) / f"rank{rank}.pt")
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="Requires PyTorch Gloo")
@pytest.mark.parametrize("enabled", [False, True])
def test_global_token_mean_matches_single_process_with_empty_rank(tmp_path, enabled):
    with socket.socket() as endpoint:
        endpoint.bind(("127.0.0.1", 0))
        port = endpoint.getsockname()[1]
    context = mp.spawn(worker, args=(port, str(tmp_path), enabled),
                       nprocs=2, join=False)
    deadline = time.monotonic() + 120
    try:
        while not context.join(timeout=5):
            if time.monotonic() > deadline:
                pytest.fail("CPU collective test exceeded 120 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
    trainer, model, reference, schedule, full, completion, valid, advantages = example(enabled=enabled)
    expected = deepcopy(model)
    loss, count = paper_loss(expected, reference, schedule, full, completion, valid,
                             advantages, trainer.weights, enabled, .01)
    loss.backward()
    for rank in range(2):
        saved = torch.load(tmp_path / f"rank{rank}.pt", weights_only=True)
        assert saved["count"] == (count if rank == 0 else 0)
        assert saved["updates"] == 1
        assert saved["evaluated"] == 1 and saved["accuracy"] == 1
        for actual, gradient, original in zip(saved["parameters"], saved["gradients"], expected.parameters()):
            assert torch.allclose(gradient, original.grad, atol=3e-7, rtol=3e-5)
            assert torch.allclose(actual, original.detach()-.03*original.grad, atol=3e-7, rtol=3e-5)

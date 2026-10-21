"""DTCA integration with the official Drift training engine.

The integration shares target filtering, global token normalization and
optimizer stepping with Drift, and adds POS credit and OPEFO for DTCA.
"""

import torch
import torch.distributed as dist

from dtca.credit import credit, targets
from dtca.entropy import calibrate
from trainer.main_rl import DriftTrainer


class Trainer(DriftTrainer):
    def __init__(self, config, mask_id, tokenizer=None, model_type="llada"):
        super().__init__(config, mask_id, tokenizer, model_type)
        self.enabled = bool(config.dtca.enabled)
        self.weights = None
        if self.enabled:
            from dtca.pos import Tagger
            from dtca.table import Table

            self.table = Table.load(config.dtca.table)
            self.tagger = Tagger(config.dtca.pos_model)

    def build_schedule(self, rollout):
        from trainer.masking_strategy import build_masking_schedule

        if self.enabled:
            from dtca.select import schedule

            ids = self.tagger.project(self.tokenizer, rollout.completion_ids, rollout.completion_mask)
            self.weights = self.table.lookup(ids)
            return schedule(rollout.step_map, rollout.completion_mask, rollout.token_probs,
                            self.weights, mass=self.config.dtca.mass,
                            minimum=self.config.dtca.min_steps, maximum=self.config.dtca.max_steps)
        return build_masking_schedule(
            rollout.step_map, rollout.completion_mask, strategy="sequential_masking",
            k=self.config.training.mask_num, token_probs=rollout.token_probs,
            sample_temperature=self.config.training.sample_temperature)

    def compute_loss(self, model, ref_model, rollout, main_schedule, old_logps=None, accelerator=None):
        self.attention = torch.cat((rollout.prompt_mask, torch.ones_like(rollout.completion_ids)), dim=1)
        try:
            return super().compute_loss(model, ref_model, rollout, main_schedule, old_logps, accelerator)
        finally:
            self.attention = None

    def _compute_loss(self, model, ref_model, sched, full_seq, prompt_len,
                      completion_ids, completion_mask, old_logps, advantages,
                      ds_engine, B, device):
        if self.num_iterations != 1 or self.loss_norm != "token":
            raise ValueError("Use one policy iteration and token normalization")
        size = int(self.config.training.micro_batch_size)
        if size < 1:
            raise ValueError("micro_batch_size must be positive")
        steps = len(sched.masking_inputs[0])
        attention = getattr(self, "attention", None)
        attention = attention.to(device) if attention is not None else torch.ones_like(full_seq)
        weights = self.weights.to(device) if self.enabled else torch.ones_like(completion_ids, dtype=torch.float32)
        parts = []
        # Replay the same RNG sequence for detached statistics and gradients.
        # Collect all statistics before updating parameters or optimizer state.
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            for step in range(steps):
                masked = torch.stack([sched.masking_inputs[b][step][1] for b in range(B)]).to(device)
                committed = torch.stack([sched.masking_inputs[b][step][2] for b in range(B)]).to(device)
                for start in range(0, B, size):
                    end = min(B, start + size)
                    logp, distribution, logits = self._forward_and_gather(
                        model, full_seq[start:end], prompt_len, completion_ids[start:end], masked[start:end],
                        attention_mask=attention[start:end])
                    active = targets(logp, masked[start:end], committed[start:end], completion_mask[start:end], self.low_prob_threshold)
                    signals = credit(weights[start:end], advantages[start:end], active)
                    entropy = -(distribution.exp() * distribution).sum(-1)
                    rows = torch.arange(start, end, device=device)[:, None].expand_as(logp)
                    parts.append({"step": step, "start": start, "end": end, "masked": masked[start:end],
                                  "active": active, "credits": signals, "logps": logp[active].clone(),
                                  "entropy": entropy[active].clone(), "rows": rows[active].clone()})
                    del logits, logp, distribution, entropy
        count = torch.tensor(sum(int(p["active"].sum()) for p in parts), dtype=torch.float64, device=device)
        global_count = count.clone()
        world = 1
        if dist.is_initialized():
            dist.all_reduce(global_count)
            world = dist.get_world_size()
        # DeepSpeed averages gradients across workers, hence world/global_count.
        scale = world / max(float(global_count), 1)
        if self.enabled:
            logps = torch.cat([p["logps"] for p in parts])
            entropy = torch.cat([p["entropy"] for p in parts])
            signals = torch.cat([p["credits"][p["active"]] for p in parts])
            rows = torch.cat([p["rows"] for p in parts])
            corrected = calibrate(logps, entropy, signals, rows,
                                  float(self.config.optimizer.params.learning_rate))
            offset = 0
            for part in parts:
                n = part["logps"].numel()
                part["credits"][part["active"]] = corrected[offset:offset+n]
                offset += n
        metrics = {name: 0.0 for name in ("loss", "adv_loss", "entropy", "kl_loss")}
        for index, part in enumerate(parts):
            start, end = part["start"], part["end"]
            logp, distribution, logits = self._forward_and_gather(
                model, full_seq[start:end], prompt_len, completion_ids[start:end], part["masked"],
                attention_mask=attention[start:end])
            active = part["active"]
            policy = -logp * part["credits"]
            divergence = torch.zeros_like(logp)
            if ref_model is not None and self.beta > 0:
                ref = self._forward_and_gather_ref(ref_model, full_seq[start:end], prompt_len,
                                                   completion_ids[start:end], part["masked"],
                                                   attention_mask=attention[start:end])
                divergence = self._compute_per_token_kl_loss(logp, ref)
            loss = ((policy + self.beta * divergence) * active).sum() * scale
            # Includes a zero-valued autograd path on dummy states/ranks.
            ds_engine.set_gradient_accumulation_boundary(index == len(parts) - 1)
            ds_engine.backward(loss)
            with torch.no_grad():
                metrics["loss"] += float(loss)
                metrics["adv_loss"] += float((policy * active).sum())
                metrics["kl_loss"] += float((divergence * active).sum())
                entropy = -(distribution.exp() * distribution).sum(-1)
                metrics["entropy"] += float((entropy * active).sum())
            del logits, distribution, logp, loss, policy, divergence
        ds_engine.step()
        denominator = max(float(count), 1)
        return {"mean_loss": metrics["loss"], "mean_adv_loss": metrics["adv_loss"] / denominator,
                "mean_clip": 0.0, "mean_ratio": 1.0, "mean_entropy": metrics["entropy"] / denominator,
                "mean_kl_loss": metrics["kl_loss"] / denominator, "total_mask_sum": float(count)}

"""Fault-tolerant distributed training entry point.

Run under torchrun, which restarts all workers when any worker dies:

    torchrun --nproc-per-node=2 --max-restarts=3 -m rtrain.train --steps 600

Each attempt resumes from the newest verified checkpoint. Because batches are
a pure function of (seed, step, rank) and the optimizer, scheduler and RNG
state are checkpointed, a resumed run follows the same trajectory as one that
was never interrupted.
"""

from __future__ import annotations

import dataclasses
import faulthandler
import json
import logging
import math
import os
import signal
import sys
import time
from collections import deque
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from . import data
from .checkpoint import CheckpointManager, SaveStats
from .config import TrainConfig, parse
from .metrics import Metrics
from .model import GPT, num_params
from .resilience import StopFlag, Watchdog, detect_stragglers


class Events:
    """Append-only JSON-lines event log shared by all ranks (O_APPEND writes)."""

    def __init__(self, path: Path, rank: int, restart: int):
        self.path, self.rank, self.restart = path, rank, restart
        path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: str, echo: bool = True, **fields) -> None:
        rec = {"t": round(time.time(), 4), "event": event, "rank": self.rank, "restart": self.restart, **fields}
        line = json.dumps(rec)
        with open(self.path, "a") as f:
            f.write(line + "\n")
        if echo:
            print(line, flush=True)


class Chaos:
    """Test-only fault injection, configured by environment variables. Faults
    fire only on the first attempt so the restarted job can make progress."""

    def __init__(self, rank: int, restart: int, events: Events):
        env = os.environ
        self.rank, self.events = rank, events
        first = restart == 0
        self.kill_step = int(env["CHAOS_KILL_STEP"]) if first and "CHAOS_KILL_STEP" in env else None
        self.kill_rank = int(env.get("CHAOS_KILL_RANK", "1"))
        self.hang_step = int(env["CHAOS_HANG_STEP"]) if first and "CHAOS_HANG_STEP" in env else None
        self.hang_rank = int(env.get("CHAOS_HANG_RANK", "1"))
        self.slow_rank = int(env["CHAOS_SLOW_RANK"]) if "CHAOS_SLOW_RANK" in env else None
        self.slow_s = float(env.get("CHAOS_SLOW_MS", "50")) / 1000

    def before_step(self, step: int) -> None:
        if self.kill_step == step and self.rank == self.kill_rank:
            self.events.emit("chaos_kill", step=step)
            os.kill(os.getpid(), signal.SIGKILL)
        if self.hang_step == step and self.rank == self.hang_rank:
            self.events.emit("chaos_hang", step=step)
            while True:  # simulate a wedged GPU / collective
                time.sleep(3600)
        if self.slow_rank == self.rank:
            time.sleep(self.slow_s)


def lr_at(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * min(1.0, progress)))


@torch.no_grad()
def eval_loss(model: torch.nn.Module, cfg: TrainConfig, device: torch.device, batches: int = 8) -> float:
    model.eval()
    total = 0.0
    for i in range(batches):  # held-out stream: steps far beyond training
        x, y = data.batch(cfg.seed, 10**9 + i, 0, cfg.batch_size, cfg.model.block_size)
        total += model(x.to(device), y.to(device))[1].item()
    model.train()
    return total / batches


def main(argv: list[str] | None = None) -> int:
    cfg = parse(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    restart = int(os.environ.get("TORCHELASTIC_RESTART_COUNT", 0))
    distributed = world > 1
    use_cuda = torch.cuda.is_available()
    faulthandler.enable()  # print the Python stack on SIGSEGV/SIGABRT/etc. in any worker
    if distributed:
        # Bound collective waits so a dead peer surfaces as an error, not a 30-minute hang.
        dist.init_process_group("nccl" if use_cuda else "gloo", timeout=timedelta(seconds=max(30.0, cfg.hang_timeout)))
    device = torch.device(f"cuda:{local_rank}" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.set_device(device)
    torch.set_num_threads(max(1, (os.cpu_count() or 1) // max(1, int(os.environ.get("LOCAL_WORLD_SIZE", 1)))))

    out = Path(cfg.out_dir)
    events = Events(out / "events.jsonl", rank, restart)
    is_main = rank == 0
    metrics = Metrics(rank)
    if cfg.metrics_port:
        metrics.serve(cfg.metrics_port + local_rank)
    metrics.restarts.labels(metrics.rank).set(restart)

    torch.manual_seed(cfg.seed)  # identical initial weights on every rank
    model = GPT(cfg.model).to(device)
    net = DDP(model, device_ids=[local_rank] if use_cuda else None) if distributed else model
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    ckpt = CheckpointManager(cfg.ckpt_dir, keep_last=cfg.keep_last, async_save=cfg.async_ckpt)

    def on_saved(s: SaveStats) -> None:
        metrics.ckpt_total.labels(metrics.rank).observe(s.total_s)
        metrics.ckpt_last_step.labels(metrics.rank).set(s.step)
        events.emit(
            "checkpoint",
            echo=False,
            step=s.step,
            stall_ms=round(s.stall_s * 1000, 2),
            total_ms=round(s.total_s * 1000, 2),
            bytes=s.bytes,
        )

    ckpt.on_saved = on_saved

    # Resume: rank 0 picks the newest verified checkpoint and broadcasts the
    # choice, so every rank loads the same one even on a lagging shared filesystem.
    if is_main:
        for stale in cfg.ckpt_dir.glob("*.tmp"):
            stale.unlink(missing_ok=True)  # partial write from a crashed save
    choice: list = [None]
    if is_main:
        latest = ckpt.latest_valid()
        choice = [str(latest[1]) if latest else None]
    if distributed:
        dist.broadcast_object_list(choice, src=0)
    start_step = 0
    if choice[0]:
        state = torch.load(choice[0], map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        torch.set_rng_state(state["rng"])
        start_step = state["step"]
    metrics.resumed_from.labels(metrics.rank).set(start_step)
    if is_main:
        events.emit(
            "attempt_start",
            resumed_from=start_step,
            world_size=world,
            params=num_params(model),
            backend=dist.get_backend() if distributed else "none",
        )

    chaos = Chaos(rank, restart, events)
    stop = StopFlag().install()
    watchdog = Watchdog(cfg.hang_timeout).start()
    window: deque[float] = deque(maxlen=cfg.straggler_every)
    tokens_per_step = cfg.batch_size * cfg.model.block_size
    last_saved = start_step
    stopped = False

    def state_dict(step: int) -> dict:
        return {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "step": step,
            "rng": torch.get_rng_state(),
            "config": dataclasses.asdict(cfg),
        }

    for step in range(start_step, cfg.steps):
        t0 = time.perf_counter()
        chaos.before_step(step)
        x, y = data.batch(cfg.seed, step, rank, cfg.batch_size, cfg.model.block_size)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        for g in opt.param_groups:
            g["lr"] = lr_at(step, cfg)
        _, loss = net(x, y)
        # Time to reach the gradient all-reduce. Synchronous DDP equalizes
        # wall-clock step time across ranks (fast ranks wait inside backward),
        # so only pre-collective time reveals which rank is the straggler.
        compute = time.perf_counter() - t0
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        done = step + 1
        watchdog.beat()

        dt = time.perf_counter() - t0
        window.append(compute)
        metrics.step.labels(metrics.rank).set(done)
        metrics.step_seconds.labels(metrics.rank).observe(dt)
        metrics.tokens_per_s.labels(metrics.rank).set(tokens_per_step / dt if dt > 0 else 0)

        if step == start_step and is_main:
            events.emit("first_step", step=done)

        if done % cfg.log_every == 0 or done == cfg.steps:
            l = loss.detach().clone()
            if distributed:
                dist.all_reduce(l, op=dist.ReduceOp.SUM)  # gloo has no AVG
                l /= world
            metrics.loss.labels(metrics.rank).set(l.item())
            if is_main:
                events.emit(
                    "train",
                    step=done,
                    loss=round(l.item(), 6),
                    lr=round(lr_at(step, cfg), 6),
                    step_ms=round(dt * 1000, 2),
                    tokens_per_s=round(tokens_per_step * world / dt),
                )

        if done % cfg.straggler_every == 0:
            slow, times = detect_stragglers(sum(window) / len(window), cfg.straggler_threshold)
            metrics.straggler.labels(metrics.rank).set(1 if rank in slow else 0)
            if slow and is_main:
                events.emit("stragglers", step=done, ranks=slow, compute_ms=[round(t * 1000, 2) for t in times])

        if (done % cfg.ckpt_every == 0 or done == cfg.steps) and is_main:
            stall = ckpt.save(done, state_dict(done))
            metrics.ckpt_stall.labels(metrics.rank).observe(stall)
            last_saved = done

        if stop.agreed():
            # Preemption: every rank stops at the same step boundary.
            if is_main:
                ckpt.wait()
                if last_saved != done:
                    ckpt.async_save = False
                    ckpt.save(done, state_dict(done))
                events.emit("graceful_stop", step=done, reason=stop.reason or "peer")
            metrics.stops.labels(metrics.rank).inc()
            stopped = True
            break
        watchdog.beat()  # checkpoint/straggler collectives count as progress too

    watchdog.stop()
    if is_main:
        ckpt.wait()
        if not stopped:
            events.emit("completed", step=cfg.steps, eval_loss=round(eval_loss(model, cfg, device), 6))
    if distributed:
        # Finish all work first, then tear down together. If one rank exits while
        # a peer is still inside gloo, the peer's I/O threads hit "connection
        # reset", the C++ exception calls std::terminate, and the process aborts
        # (exit -6) — which makes torchrun restart a job that already finished.
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())

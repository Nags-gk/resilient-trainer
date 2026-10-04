#!/usr/bin/env python3
"""Chaos benchmark: kill a worker at random steps and measure recovery.

For each trial a 2-process DDP job runs under torchrun with restarts enabled,
and one worker is SIGKILLed at a random step. Reported per trial:
  * recovery time: kill -> first optimizer step of the restarted attempt
  * lost work: steps between the last checkpoint and the kill
  * overhead: wall time vs. an uninterrupted baseline
  * exactness: whether every logged loss matches the baseline bit for bit
Then sync vs. async checkpoint stall on a larger model.

    python scripts/chaos_bench.py            # writes bench/RESULTS.md and bench/results.json
"""

from __future__ import annotations

import json
import os
import platform
import random
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS, CKPT_EVERY, TRIALS = 600, 50, 5
ARGS = ["--steps", str(STEPS), "--ckpt-every", str(CKPT_EVERY), "--log-every", "25"]


def run(out: Path, extra: list[str] = (), env: dict | None = None, restarts: int = 3) -> float:
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc-per-node=2",
        f"--max-restarts={restarts}",
        "--monitor-interval=1",
        "-m",
        "rtrain.train",
        "--out-dir",
        str(out),
        *ARGS,
        *extra,
    ]
    t0 = time.time()
    p = subprocess.run(
        cmd,
        cwd=ROOT,
        env={**os.environ, "OMP_NUM_THREADS": "1", **(env or {})},
        capture_output=True,
        text=True,
        timeout=900,
    )
    if p.returncode != 0:
        raise SystemExit(f"run failed:\n{p.stdout[-3000:]}\n{p.stderr[-3000:]}")
    return time.time() - t0


def events(out: Path) -> list[dict]:
    return [json.loads(x) for x in (out / "events.jsonl").read_text().splitlines() if x.strip()]


def losses(evs: list[dict]) -> dict[int, float]:
    return {e["step"]: e["loss"] for e in evs if e["event"] == "train"}


def main() -> None:
    rng = random.Random(7)
    tmp = Path(tempfile.mkdtemp(prefix="chaos-"))
    print(f"baseline: {STEPS} steps, 2 ranks")
    base_wall = run(tmp / "baseline", restarts=0)
    base = events(tmp / "baseline")
    base_losses = losses(base)
    base_eval = next(e["eval_loss"] for e in base if e["event"] == "completed")

    trials = []
    for i in range(TRIALS):
        kill_step = rng.randrange(60, STEPS - 60)
        kill_rank = i % 2  # alternate: rank 0 also owns checkpoint writes
        out = tmp / f"trial{i}"
        wall = run(out, env={"CHAOS_KILL_STEP": str(kill_step), "CHAOS_KILL_RANK": str(kill_rank)})
        evs = events(out)
        kill = next(e for e in evs if e["event"] == "chaos_kill")
        resumed = [e for e in evs if e["event"] == "attempt_start" and e["restart"] == 1][0]
        first = [e for e in evs if e["event"] == "first_step" and e["restart"] == 1][0]
        ev = next(e["eval_loss"] for e in evs if e["event"] == "completed")
        t = {
            "kill_step": kill_step,
            "kill_rank": kill_rank,
            "resumed_from": resumed["resumed_from"],
            "lost_steps": kill_step - resumed["resumed_from"],
            "recovery_s": round(first["t"] - kill["t"], 2),
            "overhead_s": round(wall - base_wall, 2),
            "identical": losses(evs) == base_losses and ev == base_eval,
        }
        trials.append(t)
        print(f"  trial {i}: {t}")

    # Checkpoint stall: sync vs async on a ~3M-parameter model (~38 MB with Adam state).
    big = [
        "--model-n-layer",
        "4",
        "--model-n-embd",
        "256",
        "--model-n-head",
        "4",
        "--steps",
        "120",
        "--ckpt-every",
        "10",
        "--batch-size",
        "8",
    ]
    stalls = {}
    for mode in ("false", "true"):
        out = tmp / f"ckpt-async-{mode}"
        run(out, big + ["--async-ckpt", mode], restarts=0)
        ck = [e for e in events(out) if e["event"] == "checkpoint"]
        stalls[mode] = {
            "stall_ms": round(statistics.median(e["stall_ms"] for e in ck), 1),
            "total_ms": round(statistics.median(e["total_ms"] for e in ck), 1),
            "mb": round(ck[0]["bytes"] / 1e6, 1),
        }
        print(f"  checkpoints async={mode}: {stalls[mode]}")

    res = {
        "baseline_wall_s": round(base_wall, 2),
        "steps": STEPS,
        "ckpt_every": CKPT_EVERY,
        "trials": trials,
        "checkpoint": stalls,
        "machine": f"{platform.machine()} {platform.system()}",
        "cpus": os.cpu_count(),
    }
    (ROOT / "bench").mkdir(exist_ok=True)
    (ROOT / "bench" / "results.json").write_text(json.dumps(res, indent=2))

    rec = [t["recovery_s"] for t in trials]
    lost = [t["lost_steps"] for t in trials]
    lines = [
        "# Chaos benchmark results\n",
        f"2-rank DDP (gloo, CPU) under `torchrun --max-restarts=3 --monitor-interval=1`, {STEPS} steps, "
        f"checkpoint every {CKPT_EVERY} steps (async). In each trial one worker is SIGKILLed at a random step. "
        f"Machine: {res['machine']}, {res['cpus']} CPUs. Reproduce with `python scripts/chaos_bench.py`.\n",
        "| Trial | Killed rank @ step | Resumed from | Lost steps | Recovery (kill → next step) | Wall overhead | "
        "Matches uninterrupted run |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, t in enumerate(trials):
        lines.append(
            f"| {i} | rank {t['kill_rank']} @ {t['kill_step']} | {t['resumed_from']} | {t['lost_steps']} | "
            f"{t['recovery_s']} s | {t['overhead_s']:+} s | {'yes, bit-identical' if t['identical'] else 'NO'} |"
        )
    lines += [
        "",
        f"- **Recovery time:** median {statistics.median(rec):.2f} s (max {max(rec):.2f} s) from the kill to the "
        "first optimizer step of the restarted job, including failure detection, process restart, rendezvous "
        "and checkpoint load.",
        f"- **Lost work:** median {statistics.median(lost)} steps, bounded by the checkpoint interval ({CKPT_EVERY}).",
        f"- **Correctness:** {sum(t['identical'] for t in trials)}/{len(trials)} interrupted runs reproduced every "
        "logged loss and the final eval loss of the uninterrupted run exactly.",
        "",
        "## Checkpoint stall (training loop blocked), ~3M-parameter model\n",
        "| Mode | Checkpoint size | Loop blocked (median) | Full save (median) |",
        "|---|---|---|---|",
        f"| synchronous | {stalls['false']['mb']} MB | {stalls['false']['stall_ms']} ms | "
        f"{stalls['false']['total_ms']} ms |",
        f"| async (snapshot + background write) | {stalls['true']['mb']} MB | {stalls['true']['stall_ms']} ms | "
        f"{stalls['true']['total_ms']} ms |",
    ]
    (ROOT / "bench" / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

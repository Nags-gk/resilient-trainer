"""Fault-injection tests that run real multi-process DDP jobs under torchrun.

Each test launches `torchrun --nproc-per-node=2` in a subprocess with a fault
injected (SIGKILL, hang, SIGTERM, slow rank, corrupted checkpoint) and checks
the outcome from the job's events log.
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STEPS = 120
COMMON = [
    "--steps",
    str(STEPS),
    "--ckpt-every",
    "20",
    "--log-every",
    "20",
    "--straggler-every",
    "20",
    "--model-n-layer",
    "1",
    "--batch-size",
    "16",
]


def torchrun(out: Path, *extra: str, env: dict | None = None, restarts: int = 3, wait: bool = True):
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc-per-node=2",
        f"--max-restarts={restarts}",
        "--monitor-interval=0.5",
        "-m",
        "rtrain.train",
        "--out-dir",
        str(out),
        *COMMON,
        *extra,
    ]
    full_env = {**os.environ, "OMP_NUM_THREADS": "1", **(env or {})}
    if not wait:
        return subprocess.Popen(
            cmd, cwd=ROOT, env=full_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True
        )
    p = subprocess.run(cmd, cwd=ROOT, env=full_env, capture_output=True, text=True, timeout=300)
    return p


def events(out: Path, kind: str | None = None) -> list[dict]:
    path = out / "events.jsonl"
    if not path.exists():
        return []
    evs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [e for e in evs if kind is None or e["event"] == kind]


def train_losses(out: Path) -> dict[int, float]:
    # Last value per step wins (a step may be logged before a crash and again after resume).
    return {e["step"]: e["loss"] for e in events(out, "train")}


@pytest.fixture(scope="module")
def baseline(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("baseline")
    p = torchrun(out, restarts=0)
    assert p.returncode == 0, p.stdout + p.stderr
    return out


def test_baseline_learns(baseline: Path):
    losses = train_losses(baseline)
    assert losses[STEPS] < losses[20] - 0.3, f"loss should fall: {losses}"
    assert events(baseline, "completed")


def test_sigkill_worker_resumes_with_identical_results(baseline: Path, tmp_path: Path):
    p = torchrun(tmp_path, env={"CHAOS_KILL_STEP": "53", "CHAOS_KILL_RANK": "1"})
    assert p.returncode == 0, p.stdout + p.stderr
    starts = events(tmp_path, "attempt_start")
    assert [s["resumed_from"] for s in starts] == [0, 40], "resumes from the last checkpoint before the kill"
    # Same seed, same data per (step, rank), same optimizer/RNG state: same trajectory.
    assert train_losses(tmp_path) == train_losses(baseline)
    assert events(tmp_path, "completed")[0]["eval_loss"] == events(baseline, "completed")[0]["eval_loss"]


def test_hang_is_detected_by_watchdog_and_recovered(baseline: Path, tmp_path: Path):
    t0 = time.time()
    p = torchrun(tmp_path, "--hang-timeout", "5", env={"CHAOS_HANG_STEP": "47", "CHAOS_HANG_RANK": "1"})
    assert p.returncode == 0, p.stdout + p.stderr
    assert "WATCHDOG" in p.stderr + p.stdout, "the watchdog, not a 30-minute backend timeout, ended the hang"
    assert [s["resumed_from"] for s in events(tmp_path, "attempt_start")] == [0, 40]
    assert train_losses(tmp_path)[STEPS] == train_losses(baseline)[STEPS]
    assert time.time() - t0 < 120


def test_corrupted_latest_checkpoint_falls_back(baseline: Path, tmp_path: Path):
    p = torchrun(tmp_path, "--steps", "60", restarts=0)
    assert p.returncode == 0, p.stdout + p.stderr
    newest = tmp_path / "checkpoints" / "ckpt-00000060.pt"
    raw = bytearray(newest.read_bytes())
    raw[len(raw) // 3] ^= 0xFF
    newest.write_bytes(bytes(raw))
    p = torchrun(tmp_path, restarts=0)
    assert p.returncode == 0, p.stdout + p.stderr
    assert events(tmp_path, "attempt_start")[-1]["resumed_from"] == 40, "skips the corrupted step-60 checkpoint"
    assert train_losses(tmp_path)[STEPS] == train_losses(baseline)[STEPS]


def test_sigterm_preemption_checkpoints_and_resumes_exactly(baseline: Path, tmp_path: Path):
    proc = torchrun(tmp_path, env={"CHAOS_SLOW_RANK": "0", "CHAOS_SLOW_MS": "30"}, restarts=0, wait=False)
    deadline = time.time() + 60
    while time.time() < deadline and not any(e["step"] >= 30 for e in events(tmp_path, "train")):
        time.sleep(0.2)
    os.killpg(proc.pid, signal.SIGTERM)  # what Kubernetes sends on pod eviction / spot reclaim
    proc.wait(timeout=60)
    stop = events(tmp_path, "graceful_stop")
    assert stop, proc.stdout.read().decode()
    stopped_at = stop[0]["step"]
    assert (tmp_path / "checkpoints" / f"ckpt-{stopped_at:08d}.pt").exists(), "checkpoint written at the stop step"

    p = torchrun(tmp_path, restarts=0)
    assert p.returncode == 0, p.stdout + p.stderr
    assert events(tmp_path, "attempt_start")[-1]["resumed_from"] == stopped_at, "no work lost"
    assert train_losses(tmp_path)[STEPS] == train_losses(baseline)[STEPS]


def test_straggler_is_flagged(tmp_path: Path):
    p = torchrun(tmp_path, "--steps", "40", env={"CHAOS_SLOW_RANK": "1", "CHAOS_SLOW_MS": "80"}, restarts=0)
    assert p.returncode == 0, p.stdout + p.stderr
    flagged = events(tmp_path, "stragglers")
    assert flagged and all(e["ranks"] == [1] for e in flagged), flagged

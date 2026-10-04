"""Fast unit tests: checkpoints, data, config, watchdog, stop flag."""

import json
import os
import signal
import threading
import time
from pathlib import Path

import pytest
import torch

from rtrain import data
from rtrain.checkpoint import CheckpointManager, snapshot
from rtrain.config import parse
from rtrain.resilience import StopFlag, Watchdog, detect_stragglers, flag_slow


def state(v: float) -> dict:
    return {"w": torch.full((4, 4), v), "step": int(v), "nested": {"b": torch.tensor([v])}}


# ---------------- checkpoints ----------------


@pytest.mark.parametrize("async_save", [False, True])
def test_roundtrip_and_pruning(tmp_path: Path, async_save: bool):
    m = CheckpointManager(tmp_path, keep_last=2, async_save=async_save)
    for s in (10, 20, 30):
        m.save(s, state(s))
    m.wait()
    assert [p.name for p in m.list()] == ["ckpt-00000030.pt", "ckpt-00000020.pt"]
    assert not (tmp_path / "ckpt-00000010.pt.json").exists(), "manifests are pruned with their checkpoints"
    step, st = m.load_latest()
    assert step == 30 and torch.equal(st["w"], torch.full((4, 4), 30.0))
    meta = json.loads((tmp_path / "ckpt-00000030.pt.json").read_text())
    assert meta["step"] == 30 and len(meta["sha256"]) == 64


def test_corrupted_newest_falls_back(tmp_path: Path):
    m = CheckpointManager(tmp_path, keep_last=3, async_save=False)
    m.save(10, state(10))
    m.save(20, state(20))
    newest = tmp_path / "ckpt-00000020.pt"
    raw = bytearray(newest.read_bytes())
    raw[len(raw) // 2] ^= 0xFF  # bit rot / torn write
    newest.write_bytes(bytes(raw))
    assert not m.verify(newest)
    step, st = m.load_latest()
    assert step == 10 and st["step"] == 10


def test_missing_manifest_and_truncation(tmp_path: Path):
    m = CheckpointManager(tmp_path, keep_last=3, async_save=False)
    m.save(10, state(10))
    m.save(20, state(20))
    m.save(30, state(30))
    (tmp_path / "ckpt-00000030.pt.json").unlink()  # crashed between rename and manifest
    p20 = tmp_path / "ckpt-00000020.pt"
    p20.write_bytes(p20.read_bytes()[:100])  # truncated
    assert m.latest_valid()[0] == 10
    assert m.load_latest()[0] == 10


def test_no_checkpoints(tmp_path: Path):
    m = CheckpointManager(tmp_path / "empty")
    assert m.latest_valid() is None and m.load_latest() is None


def test_temp_files_are_ignored(tmp_path: Path):
    m = CheckpointManager(tmp_path, async_save=False)
    m.save(10, state(10))
    (tmp_path / "ckpt-00000020.pt.tmp").write_bytes(b"partial")
    assert m.latest_valid()[0] == 10


def test_async_snapshot_is_isolated_from_training(tmp_path: Path):
    m = CheckpointManager(tmp_path, async_save=True)
    w = torch.zeros(1000, 100)
    m.save(1, {"w": w})
    w += 1  # training keeps mutating parameters in place
    m.wait()
    assert torch.count_nonzero(m.load_latest()[1]["w"]) == 0


def test_async_failure_surfaces(tmp_path: Path):
    m = CheckpointManager(tmp_path, async_save=True)
    m.save(1, {"bad": threading.Lock()})  # not picklable
    with pytest.raises(RuntimeError, match="background checkpoint save failed"):
        m.wait()


def test_snapshot_copies():
    t = torch.ones(3)
    s = snapshot({"a": [t, (t,)]})
    t.zero_()
    assert s["a"][0].sum() == 3 and isinstance(s["a"][1], tuple)


# ---------------- data ----------------


def test_batches_are_deterministic_and_distinct():
    a = data.batch(1, 5, 0, 4, 32)
    b = data.batch(1, 5, 0, 4, 32)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    assert not torch.equal(a[0], data.batch(1, 6, 0, 4, 32)[0]), "steps differ"
    assert not torch.equal(a[0], data.batch(1, 5, 1, 4, 32)[0]), "ranks differ"
    x, y = a
    assert x.shape == (4, 32) and torch.equal(x[:, 1:], y[:, :-1]), "targets are inputs shifted by one"
    assert int(x.max()) < len(data.VOCAB)


def test_text_is_valid_arithmetic():
    g = torch.Generator().manual_seed(0)
    for prob in data._sample_text(g, 200).split(";")[1:-1]:
        lhs, rhs = prob.split("=")
        a, b = lhs.split("+")
        assert int(a) + int(b) == int(rhs)


# ---------------- config ----------------


def test_config_yaml_and_cli_override(tmp_path: Path):
    f = tmp_path / "c.yaml"
    f.write_text("steps: 100\nckpt_every: 20\nmodel: {n_layer: 3}\n")
    cfg = parse(["--config", str(f), "--steps", "50", "--model-n-embd", "32", "--async-ckpt", "false"])
    assert (cfg.steps, cfg.ckpt_every, cfg.model.n_layer, cfg.model.n_embd, cfg.async_ckpt) == (50, 20, 3, 32, False)
    with pytest.raises(ValueError):
        parse(["--model-n-embd", "30", "--model-n-head", "4"])


# ---------------- resilience ----------------


def test_watchdog_fires_without_progress_and_not_with_it():
    fired = threading.Event()
    wd = Watchdog(0.3, on_timeout=fired.set).start()
    for _ in range(8):
        time.sleep(0.1)
        wd.beat()
    assert not fired.is_set()
    assert fired.wait(2.0), "watchdog must fire once beats stop"


def test_stop_flag_on_sigterm():
    s = StopFlag().install()
    try:
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(0.05)
        assert s.requested and s.reason == "SIGTERM" and s.agreed()
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)


def test_stragglers_single_process():
    assert detect_stragglers(0.1, 1.5) == ([], [0.1])


def test_flag_slow_uses_lower_median():
    assert flag_slow([0.10, 0.20], 1.5) == [1], "2 ranks: the slow one must be detectable"
    assert flag_slow([0.1, 0.1, 0.1, 0.4], 1.5) == [3]
    assert flag_slow([0.1, 0.12, 0.11], 1.5) == []
    assert flag_slow([], 1.5) == [] and flag_slow([0.0, 0.0], 1.5) == []


@pytest.mark.parametrize("async_save", [False, True])
def test_stall_accounting(tmp_path: Path, async_save: bool):
    m = CheckpointManager(tmp_path, async_save=async_save)
    blocked = m.save(1, {"w": torch.zeros(500, 500)})
    m.wait()
    s = m.last_stats
    assert s.total_s >= s.stall_s > 0
    if not async_save:
        assert s.stall_s == s.total_s, "a sync save blocks for the whole write"
        assert blocked >= s.total_s * 0.9


def test_watchdog_stop_joins_thread():
    wd = Watchdog(30).start()
    wd.stop()
    assert not wd._thread.is_alive(), "no daemon thread may outlive training"

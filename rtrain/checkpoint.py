"""Crash-safe checkpointing.

Guarantees:
* **Atomic**: write to a temp file, fsync, then os.replace(). A crash at any
  moment leaves either the old checkpoint or the new one, never half of one.
* **Verified**: each checkpoint has a sidecar manifest with its SHA-256. On
  resume, checkpoints that fail verification or fail to load are skipped and
  the next older one is used.
* **Async** (optional): the training loop only pays for snapshotting tensors
  to host memory; serialization and disk I/O run on a background thread. At
  most one save is in flight; a new save waits for the previous one.
* **Bounded**: only the newest `keep_last` checkpoints are kept.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

log = logging.getLogger(__name__)
NAME = re.compile(r"^ckpt-(\d{8})\.pt$")


@dataclass
class SaveStats:
    step: int
    stall_s: float  # time the training loop was blocked
    total_s: float  # snapshot + serialize + fsync + rename
    bytes: int


def snapshot(obj: Any) -> Any:
    """Deep-copy a state tree, moving tensors to CPU so training can continue
    mutating its own copies while the background thread serializes these."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().to("cpu", copy=True)
    if isinstance(obj, dict):
        return {k: snapshot(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(snapshot(v) for v in obj)
    return obj


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fsync_dir(d: Path) -> None:
    fd = os.open(d, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class CheckpointManager:
    def __init__(self, directory: Path, keep_last: int = 3, async_save: bool = True):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.keep_last = keep_last
        self.async_save = async_save
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self.last_stats: SaveStats | None = None
        self.on_saved = None  # optional callback(SaveStats)

    # ---------- saving ----------
    def save(self, step: int, state: dict) -> float:
        """Save `state` for `step`. Returns seconds the caller was blocked."""
        t0 = time.perf_counter()
        self.wait()  # backpressure: one save in flight
        snap = snapshot(state)
        stall = time.perf_counter() - t0
        if self.async_save:
            self._thread = threading.Thread(target=self._write_guarded, args=(step, snap, t0, stall), daemon=True)
            self._thread.start()
        else:
            self._write(step, snap, t0, stall)
            stall = time.perf_counter() - t0
        return stall

    def wait(self) -> None:
        """Block until any in-flight save finishes; re-raise its error."""
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        if self._error is not None:
            err, self._error = self._error, None
            raise RuntimeError("background checkpoint save failed") from err

    def _write_guarded(self, step: int, snap: dict, t0: float, stall: float) -> None:
        try:
            self._write(step, snap, t0, stall)
        except BaseException as e:  # surfaced on the next save()/wait()
            log.exception("checkpoint save failed")
            self._error = e

    def _write(self, step: int, snap: dict, t0: float, stall: float) -> None:
        final = self.dir / f"ckpt-{step:08d}.pt"
        tmp = final.with_suffix(".pt.tmp")
        with open(tmp, "wb") as f:
            torch.save(snap, f)
            f.flush()
            os.fsync(f.fileno())
        digest = _sha256(tmp)
        size = tmp.stat().st_size
        os.replace(tmp, final)
        manifest = {"step": step, "file": final.name, "sha256": digest, "bytes": size, "time": time.time()}
        mtmp = self.dir / f".{final.name}.json.tmp"
        mtmp.write_text(json.dumps(manifest))
        os.replace(mtmp, final.with_suffix(".pt.json"))
        _fsync_dir(self.dir)
        total = time.perf_counter() - t0
        # A synchronous save blocks the caller for the whole write, not just the snapshot.
        blocked = stall if self.async_save else total
        self.last_stats = SaveStats(step=step, stall_s=blocked, total_s=total, bytes=size)
        if self.on_saved:
            self.on_saved(self.last_stats)
        self._prune()

    def _prune(self) -> None:
        for path in self.list()[self.keep_last :]:
            for p in (path, path.with_suffix(".pt.json")):
                p.unlink(missing_ok=True)

    # ---------- loading ----------
    def list(self) -> list[Path]:
        """Checkpoint files, newest first."""
        files = [p for p in self.dir.iterdir() if NAME.match(p.name)]
        return sorted(files, key=lambda p: int(NAME.match(p.name).group(1)), reverse=True)

    def verify(self, path: Path) -> bool:
        manifest = path.with_suffix(".pt.json")
        if not manifest.exists():
            log.warning("checkpoint %s has no manifest; skipping", path.name)
            return False
        try:
            meta = json.loads(manifest.read_text())
        except (OSError, json.JSONDecodeError):
            log.warning("checkpoint %s has an unreadable manifest; skipping", path.name)
            return False
        if _sha256(path) != meta.get("sha256"):
            log.warning("checkpoint %s failed its checksum; skipping", path.name)
            return False
        return True

    def latest_valid(self) -> tuple[int, Path] | None:
        for path in self.list():
            if self.verify(path):
                return int(NAME.match(path.name).group(1)), path
        return None

    def load_latest(self, map_location: str | torch.device = "cpu") -> tuple[int, dict] | None:
        for path in self.list():
            if not self.verify(path):
                continue
            try:
                return int(NAME.match(path.name).group(1)), torch.load(
                    path, map_location=map_location, weights_only=False
                )
            except Exception:  # truncated or otherwise unloadable despite a manifest
                log.exception("checkpoint %s failed to load; trying an older one", path.name)
        return None

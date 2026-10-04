"""Runtime safety nets for distributed training.

* `Watchdog`: aborts the process if training makes no progress for
  `timeout` seconds. A rank stuck in a collective (a peer died or a NCCL
  hang) would otherwise sit there until the backend's 30-minute default
  timeout. Exiting non-zero lets torchrun or Kubernetes restart the job,
  which then resumes from the last checkpoint.
* `StopFlag`: records SIGTERM (spot/preemptible VM reclaim, node drain).
  Ranks agree on stopping with an all-reduce, so they all checkpoint and exit
  together instead of one rank leaving the others blocked in a collective.
* `detect_stragglers`: compares per-rank compute time across the job.
"""

from __future__ import annotations

import faulthandler
import logging
import os
import signal
import sys
import threading
import time

import torch
import torch.distributed as dist

log = logging.getLogger(__name__)
EXIT_HANG = 86


class Watchdog:
    def __init__(self, timeout: float, on_timeout=None):
        self.timeout = timeout
        self._last = time.monotonic()
        self._stop = threading.Event()
        self._on_timeout = on_timeout or self._abort
        self._thread = threading.Thread(target=self._run, name="watchdog", daemon=True)

    def start(self) -> Watchdog:
        self._thread.start()
        return self

    def beat(self) -> None:
        self._last = time.monotonic()

    def stop(self) -> None:
        self._stop.set()

    def idle_for(self) -> float:
        return time.monotonic() - self._last

    def _run(self) -> None:
        interval = max(0.05, min(1.0, self.timeout / 4))
        while not self._stop.wait(interval):
            if self.idle_for() > self.timeout:
                self._on_timeout()
                return

    def _abort(self) -> None:
        sys.stderr.write(f"WATCHDOG: no training progress for {self.timeout:.0f}s; dumping stacks and aborting\n")
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
        sys.stderr.flush()
        os._exit(EXIT_HANG)  # skip atexit/finalizers: they may block on the hung collective


class StopFlag:
    def __init__(self) -> None:
        self.requested = False
        self.reason = ""

    def install(self) -> StopFlag:
        signal.signal(signal.SIGTERM, self._handler)
        return self

    def _handler(self, signum, _frame) -> None:
        self.requested = True
        self.reason = signal.Signals(signum).name

    def agreed(self) -> bool:
        """True on every rank if any rank was asked to stop."""
        if not (dist.is_available() and dist.is_initialized()):
            return self.requested
        t = torch.tensor([1 if self.requested else 0], dtype=torch.int32)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return bool(t.item())


def detect_stragglers(local_step_time: float, threshold: float) -> tuple[list[int], list[float]]:
    """All-gather each rank's recent mean pre-collective compute time; return
    ranks slower than `threshold` x the median, plus all the times.

    Pass compute time, not wall-clock step time: in synchronous data
    parallelism every rank's step ends together at the gradient all-reduce."""
    if not (dist.is_available() and dist.is_initialized()):
        return [], [local_step_time]
    world = dist.get_world_size()
    t = torch.tensor([local_step_time], dtype=torch.float64)
    out = [torch.zeros_like(t) for _ in range(world)]
    dist.all_gather(out, t)
    times = [float(x.item()) for x in out]
    return flag_slow(times, threshold), times


def flag_slow(times: list[float], threshold: float) -> list[int]:
    """Indices whose time exceeds `threshold` x the lower median. The lower
    median matters for even world sizes: with 2 ranks the upper median *is*
    the slow rank, so nothing could ever exceed it."""
    if not times:
        return []
    median = sorted(times)[(len(times) - 1) // 2]
    return [r for r, v in enumerate(times) if median > 0 and v > threshold * median]

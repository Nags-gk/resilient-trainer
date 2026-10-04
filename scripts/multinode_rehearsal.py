#!/usr/bin/env python3
"""Local rehearsal of the Kubernetes gang-restart pattern.

Two torchrun agents (one per simulated node) run under a bash loop that mimics
the kubelet's `restartPolicy: OnFailure`. Near step 150, node 1 loses its agent
and its worker (SIGKILL, like a deleted pod). Every agent runs with
--max-restarts=0, so the survivor exits too; both are restarted, rendezvous
fresh, and the job must resume from a checkpoint and complete.

    python scripts/multinode_rehearsal.py          # node crash (SIGKILL)
    python scripts/multinode_rehearsal.py delete   # pod deletion (SIGTERM, grace, replacement)
"""

from __future__ import annotations

import contextlib
import json
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = Path(tempfile.mkdtemp(prefix="gang-"))
OUT = WORK / "run"
ENV = {**os.environ, "OMP_NUM_THREADS": "1", "GLOO_SOCKET_IFNAME": "lo"}
MODE = sys.argv[1] if len(sys.argv) > 1 else "kill"  # kill: node crash; delete: pod deletion (SIGTERM)


def agent_cmd() -> list[str]:
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes=2",
        "--nproc-per-node=1",
        "--max-restarts=0",
        "--rdzv-backend=c10d",
        "--rdzv-endpoint=127.0.0.1:29411",
        "--rdzv-id=gang",
        "--local-addr=127.0.0.1",
        "--monitor-interval=1",
        "-m",
        "rtrain.train",
        "--out-dir",
        str(OUT),
        "--steps",
        "400",
        "--ckpt-every",
        "50",
        "--log-every",
        "25",
        "--hang-timeout",
        "30",
    ]


def kubelet(i: int) -> subprocess.Popen:
    """Restart the 'container' whenever it exits non-zero."""
    log = WORK / f"node{i}.log"
    cmd = " ".join(shlex.quote(c) for c in agent_cmd())
    script = f"until {cmd} >> {log} 2>&1; do echo restart >> {log}; sleep 1; done"
    return subprocess.Popen(["bash", "-c", script], cwd=ROOT, env=ENV, start_new_session=True)


def events() -> list[dict]:
    path = OUT / "events.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def children(pid: int) -> list[int]:
    out = []
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            ppid = int(Path(f"/proc/{d}/stat").read_text().rsplit(")", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            continue
        if ppid == pid:
            out += [int(d), *children(int(d))]
    return out


def main() -> int:
    k0 = kubelet(0)
    time.sleep(0.5)
    k1 = kubelet(1)
    try:
        while not any(e["event"] == "train" and e["step"] >= 150 for e in events()):
            time.sleep(0.3)
        agent = next(p for p in children(k1.pid) if "torch.distributed.run" in Path(f"/proc/{p}/cmdline").read_text())
        victims = [agent, *children(agent)]
        t_kill = time.time()
        if MODE == "delete":
            # Pod deletion: SIGTERM to the container's PID 1 (torchrun), grace period,
            # then the pod is gone and the Job controller starts a replacement.
            os.kill(agent, signal.SIGTERM)
            time.sleep(5)
            os.killpg(k1.pid, signal.SIGKILL)
            for v in victims:
                with contextlib.suppress(ProcessLookupError):
                    os.kill(v, signal.SIGKILL)
            k1 = kubelet(1)
            print("node 1 pod deleted (SIGTERM, grace, replacement started)")
        else:
            for v in victims:
                os.kill(v, signal.SIGKILL)
            print(f"node 1 lost (pids {victims})")

        deadline = time.time() + 300
        while time.time() < deadline and not any(e["event"] == "completed" for e in events()):
            time.sleep(0.5)
        evs = events()
        starts = [e["resumed_from"] for e in evs if e["event"] == "attempt_start"]
        firsts = [e for e in evs if e["event"] == "first_step"]
        done = [e for e in evs if e["event"] == "completed"]
        print(f"attempts resumed_from={starts} recovery_s={firsts[-1]['t'] - t_kill:.2f} completed={bool(done)}")
        return 0 if done and len(starts) >= 2 and starts[-1] > 0 else 1
    finally:
        for k in (k0, k1):
            os.killpg(k.pid, signal.SIGKILL)


if __name__ == "__main__":
    sys.exit(main())

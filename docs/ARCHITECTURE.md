# Architecture and design decisions

## Training step

```
for step in range(resumed_step, steps):
    chaos.before_step(step)                  # test-only fault injection
    x, y = data.batch(seed, step, rank)      # pure function → reproducible after restart
    loss = ddp_model(x, y)                   # compute time measured up to here (straggler signal)
    loss.backward()                          # DDP all-reduces gradients during backward
    clip, optimizer.step()
    watchdog.beat()
    every log_every:       all-reduce loss for logging
    every straggler_every: all-gather compute times, flag slow ranks
    every ckpt_every:      rank 0 snapshots state → background write
    stop.agreed():         all-reduce SIGTERM flag → checkpoint + exit together
```

## Failure model

| Failure | Detection | Recovery | Work lost |
|---|---|---|---|
| Worker process crash (OOM, segfault, Xid) | Peer's collective errors immediately; torchrun sees a dead worker | torchrun restarts all workers (single node) or the gang restarts (multi-node); resume | Steps since the last checkpoint |
| Hung collective / wedged device | Watchdog: no progress for `hang_timeout` | Abort with stack dump (exit 86) → restart → resume | Steps since the last checkpoint, plus the timeout |
| Preemption (SIGTERM) | Signal handler + all-reduce agreement | Checkpoint at the current step, exit 0 | None |
| Corrupted / partial checkpoint | Missing manifest, checksum mismatch, load error | Fall back to the next older checkpoint | One extra interval |
| Crash during a checkpoint write | Temp file never renamed | Ignored on resume; removed by rank 0 at start | None beyond the previous checkpoint |
| Node loss | Peers' collectives fail; kubelet / Job controller notice | Gang restart: every container exits and restarts, fresh rendezvous | Steps since the last checkpoint |
| Slow rank (thermal throttling, bad link, noisy neighbor) | Pre-collective compute time > threshold × lower median | Flagged (metric + event) for cordoning; not auto-remediated | n/a |

## Decisions and trade-offs

| Decision | Why | Trade-off |
|---|---|---|
| Step-indexed data instead of a stateful dataloader | No iterator position to checkpoint; makes exact resume trivially testable | Real datasets need a deterministic sharded sampler keyed by step (same idea: map step → sample indices) |
| Rank 0 writes full checkpoints | DDP replicas are identical; one writer avoids contention | For FSDP / tensor parallel, use sharded checkpoints (e.g. `torch.distributed.checkpoint`): each rank writes its shard |
| Async save with a single in-flight write | Hides serialization and I/O (7.4 ms vs 104.8 ms stall); backpressure prevents unbounded memory | Needs host memory for one snapshot; the newest checkpoint is durable slightly later |
| fsync + rename + directory fsync + checksum manifest | Durability and torn-write detection across crashes and flaky network filesystems | Extra I/O per checkpoint |
| Rank 0 broadcasts the resume choice | Shared filesystems can be eventually consistent across nodes | One small collective at startup |
| All-reduce the stop flag every step | All ranks must leave the collective schedule at the same point | A tiny collective per step (negligible next to gradient all-reduce) |
| Gang restart for multi-node | Avoids torchrun's attempt-prefix mismatch with replacement nodes; matches JobSet / PyTorchJob semantics | A single failure restarts every rank (correct for synchronous DDP anyway) |
| Indexed Job + headless Service instead of an operator | Runs on any cluster; stable hostnames for the rendezvous endpoint | No elastic scaling of world size; for that, use the Kubeflow Training Operator or JobSet |

## Not in scope (yet)

- Sharded / distributed checkpoints for FSDP and tensor parallelism.
- In-memory peer checkpoints for sub-second recovery.
- Automatic straggler eviction (cordon the node, restart without it).
- Elastic world size (continue on fewer nodes, rescale the batch).

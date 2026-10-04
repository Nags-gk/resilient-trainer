# Study guide: owning this codebase

## Reading order (about 3 hours)

1. `rtrain/data.py`: why step-indexed batches make resume exact.
2. `rtrain/checkpoint.py` + its tests: atomic writes, manifests, fallback, async.
3. `rtrain/resilience.py`: watchdog, stop flag, straggler detection.
4. `rtrain/train.py`: the loop; note where each collective happens.
5. `tests/test_distributed.py`: how faults are injected into real torchrun jobs.
6. `deploy/k8s/trainer-job.yaml`: why Indexed Job + headless Service + gang restart.

## Concepts to be able to explain

- **DDP**: replicated model, gradient all-reduce during backward (bucketed, overlapped with compute), why every rank must call the same collectives in the same order.
- **torchrun / TorchElastic**: agents, c10d rendezvous, `RANK` / `LOCAL_RANK` / `WORLD_SIZE`, `--max-restarts`, what happens when a worker dies.
- **gloo vs NCCL**: CPU vs GPU collectives; op support differences (no AVG in gloo).
- **Checkpoint durability**: page cache vs disk, why `fsync` before `rename`, why fsync the directory, torn writes.
- **Determinism**: what is needed to reproduce a trajectory (data order, initial weights, optimizer state, RNG, fixed world size).
- **Failure detection timescales**: collective timeouts, watchdogs, heartbeats; why the backend default (30 minutes) is too long.
- **Preemption**: SIGTERM → grace period → SIGKILL in Kubernetes; spot VM eviction notices.

## Likely interview questions

1. *How do you know resume is correct?* The data is a function of (seed, step, rank) and every piece of optimizer state is checkpointed, so an interrupted run must match an uninterrupted one exactly. The tests assert equality of every logged loss and the final eval loss, and 5/5 chaos trials matched bit-for-bit.
2. *What's the cost of checkpointing, and how did you reduce it?* A synchronous 38 MB save blocked the loop for ~105 ms; async snapshot + background write cut that to ~7 ms. The remaining cost is the device→host copy. Next steps: pinned memory, sharded checkpoints, in-memory peer replication.
3. *A rank hangs in all-reduce. What happens?* The watchdog sees no progress for `hang_timeout`, dumps every thread's stack (for debugging), and exits 86. torchrun or Kubernetes restarts the job, which resumes from the last checkpoint. Without it, the job sits until the backend timeout.
4. *Why did your first straggler detector fail?* Synchronous all-reduce equalizes step times; you have to measure time to reach the collective. And with two ranks the upper median is the slow rank itself, so use the lower median.
5. *Why gang restart on Kubernetes?* torchrun's per-node restart counter is part of the store key prefix; a replacement node starts at 0 while survivors are at N, so they never rendezvous. Restarting every container keeps them aligned; synchronous training needs all ranks anyway.
6. *How would this change for a 1,000-GPU FSDP job?* Sharded checkpoints written in parallel by every rank; a manifest committed last (two-phase); faster detection (NCCL async error handling, heartbeats); hot spares and in-memory checkpoints to cut recovery to seconds; straggler eviction feeding the scheduler.
7. *What does preemption handling guarantee?* Zero lost work as long as the grace period covers one step plus one checkpoint write; the stop step's checkpoint is the resume point.

## Exercises (do these yourself)

1. Switch the checkpoint format to `torch.distributed.checkpoint` (sharded) and keep the tests green.
2. Add pinned-memory snapshots and measure the stall again.
3. Add a `--max-lost-steps` mode that checkpoints by time instead of by step count.
4. Run on one real GPU (e.g. an Azure NC-series VM for an hour) with NCCL and record the chaos benchmark.
5. Feed the straggler metric to GPU Fleet Sentinel's controller to cordon slow nodes.

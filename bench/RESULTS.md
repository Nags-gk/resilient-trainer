# Chaos benchmark results

2-rank DDP (gloo, CPU) under `torchrun --max-restarts=3 --monitor-interval=1`, 600 steps, checkpoint every 50 steps (async). In each trial one worker is SIGKILLed at a random step. Machine: x86_64 Linux, 2 CPUs. Reproduce with `python scripts/chaos_bench.py`.

| Trial | Killed rank @ step | Resumed from | Lost steps | Recovery (kill → next step) | Wall overhead | Matches uninterrupted run |
|---|---|---|---|---|---|---|
| 0 | rank 0 @ 225 | 200 | 25 | 2.18 s | +1.97 s | yes, bit-identical |
| 1 | rank 1 @ 137 | 100 | 37 | 1.99 s | +2.08 s | yes, bit-identical |
| 2 | rank 0 @ 262 | 250 | 12 | 2.44 s | +2.97 s | yes, bit-identical |
| 3 | rank 1 @ 393 | 350 | 43 | 2.6 s | +2.0 s | yes, bit-identical |
| 4 | rank 0 @ 84 | 50 | 34 | 1.99 s | +2.05 s | yes, bit-identical |

- **Recovery time:** median 2.18 s (max 2.60 s) from the kill to the first optimizer step of the restarted job, including failure detection, process restart, rendezvous and checkpoint load.
- **Lost work:** median 34 steps, bounded by the checkpoint interval (50).
- **Correctness:** 5/5 interrupted runs reproduced every logged loss and the final eval loss of the uninterrupted run exactly.

## Checkpoint stall (training loop blocked), ~3M-parameter model

| Mode | Checkpoint size | Loop blocked (median) | Full save (median) |
|---|---|---|---|
| synchronous | 38.3 MB | 104.8 ms | 104.8 ms |
| async (snapshot + background write) | 38.3 MB | 7.4 ms | 151.8 ms |

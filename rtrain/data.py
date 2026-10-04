"""Deterministic, step-indexed training data.

Every batch is a pure function of (seed, global step, rank). There is no
dataloader position to save: after a restart the trainer regenerates exactly
the batches it would have seen, which is what makes resumed training
reproduce the uninterrupted loss curve bit for bit.

The task is learnable but non-trivial: streams of addition problems such as
"417+85=502;". The model has to learn digit-level carrying to predict the sums.
"""

from __future__ import annotations

import torch

VOCAB = "0123456789+=; "
STOI = {c: i for i, c in enumerate(VOCAB)}


def _seed(seed: int, step: int, rank: int) -> int:
    # SplitMix64-style mixing so neighboring (step, rank) pairs get unrelated streams.
    x = (seed * 0x9E3779B97F4A7C15 + step * 0xBF58476D1CE4E5B9 + rank * 0x94D049BB133111EB) & (2**64 - 1)
    x ^= x >> 31
    return x & (2**63 - 1)


def _sample_text(g: torch.Generator, length: int) -> str:
    out = []
    n = 0
    while n < length:
        a, b = (int(v) for v in torch.randint(0, 1000, (2,), generator=g))
        s = f"{a}+{b}={a + b};"
        out.append(s)
        n += len(s)
    return "".join(out)[:length]


def batch(seed: int, step: int, rank: int, batch_size: int, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (inputs, targets) of shape [batch_size, block_size]."""
    g = torch.Generator().manual_seed(_seed(seed, step, rank))
    rows = []
    for _ in range(batch_size):
        # Random offset so sequences don't always start at a problem boundary.
        off = int(torch.randint(0, 8, (1,), generator=g))
        text = _sample_text(g, block_size + 1 + off)[off:]
        rows.append([STOI[c] for c in text])
    t = torch.tensor(rows, dtype=torch.long)
    return t[:, :-1].contiguous(), t[:, 1:].contiguous()

"""Prometheus metrics, one endpoint per rank (port = base + LOCAL_RANK)."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, start_http_server


class Metrics:
    def __init__(self, rank: int):
        self.registry = CollectorRegistry()
        r = self.registry
        lbl = ["rank"]
        self.rank = str(rank)
        self.step = Gauge("train_step", "Last completed optimizer step.", lbl, registry=r)
        self.loss = Gauge("train_loss", "Training loss at the last step.", lbl, registry=r)
        self.step_seconds = Histogram(
            "train_step_seconds",
            "Wall time per optimizer step.",
            lbl,
            buckets=(0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10),
            registry=r,
        )
        self.tokens_per_s = Gauge("train_tokens_per_second", "Training throughput on this rank.", lbl, registry=r)
        self.ckpt_stall = Histogram(
            "train_checkpoint_stall_seconds",
            "Time the training loop was blocked by a checkpoint save.",
            lbl,
            buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 30),
            registry=r,
        )
        self.ckpt_total = Histogram(
            "train_checkpoint_save_seconds",
            "End-to-end checkpoint save time (snapshot, serialize, fsync, rename).",
            lbl,
            buckets=(0.01, 0.05, 0.1, 0.5, 1, 5, 30, 120),
            registry=r,
        )
        self.ckpt_last_step = Gauge(
            "train_checkpoint_last_step", "Step of the newest durable checkpoint.", lbl, registry=r
        )
        self.restarts = Gauge("train_restarts", "torchrun restart count for this job attempt.", lbl, registry=r)
        self.straggler = Gauge("train_straggler", "1 if this rank was flagged as a straggler.", lbl, registry=r)
        self.resumed_from = Gauge("train_resumed_from_step", "Step the current attempt resumed from.", lbl, registry=r)
        self.stops = Counter("train_graceful_stops_total", "Coordinated stops after SIGTERM.", lbl, registry=r)

    def serve(self, port: int) -> None:
        start_http_server(port, registry=self.registry)

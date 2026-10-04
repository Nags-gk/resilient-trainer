"""Training configuration. Every field can be set from the CLI or a YAML file."""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    vocab_size: int = 16
    block_size: int = 64
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 64


@dataclass
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    steps: int = 600
    batch_size: int = 32  # per rank
    lr: float = 3e-3
    min_lr: float = 3e-4
    warmup_steps: int = 50
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    seed: int = 1234

    out_dir: str = "runs/default"
    ckpt_every: int = 50
    keep_last: int = 3
    async_ckpt: bool = True

    log_every: int = 10
    metrics_port: int = 0  # 0 disables; rank r serves on metrics_port + local_rank
    hang_timeout: float = 60.0  # seconds without progress before the watchdog aborts
    straggler_every: int = 20
    straggler_threshold: float = 1.5  # step time vs. median across ranks

    @property
    def ckpt_dir(self) -> Path:
        return Path(self.out_dir) / "checkpoints"


def _add_fields(parser: argparse.ArgumentParser, cls, prefix: str = "") -> None:
    for f in dataclasses.fields(cls):
        if dataclasses.is_dataclass(f.type) or f.name == "model":
            continue
        name = f"--{prefix}{f.name.replace('_', '-')}"
        default = f.default if f.default is not dataclasses.MISSING else None
        if isinstance(default, bool):
            parser.add_argument(name, type=lambda s: s.lower() in ("1", "true", "yes"), default=None)
        else:
            parser.add_argument(name, type=type(default), default=None)


def parse(argv: list[str] | None = None) -> TrainConfig:
    p = argparse.ArgumentParser(description="Fault-tolerant distributed trainer")
    p.add_argument("--config", type=Path, help="YAML file; CLI flags override it")
    _add_fields(p, TrainConfig)
    _add_fields(p, ModelConfig, prefix="model-")
    args = p.parse_args(argv)

    cfg = TrainConfig()
    if args.config:
        raw = yaml.safe_load(args.config.read_text()) or {}
        model_raw = raw.pop("model", {}) or {}
        cfg = dataclasses.replace(cfg, **raw, model=ModelConfig(**model_raw))
    overrides = {k: v for k, v in vars(args).items() if v is not None and k != "config"}
    model_over = {k[len("model_") :]: v for k, v in overrides.items() if k.startswith("model_")}
    top_over = {k: v for k, v in overrides.items() if not k.startswith("model_")}
    cfg = dataclasses.replace(cfg, **top_over, model=dataclasses.replace(cfg.model, **model_over))
    validate(cfg)
    return cfg


def validate(cfg: TrainConfig) -> None:
    if cfg.model.n_embd % cfg.model.n_head:
        raise ValueError("model.n_embd must be divisible by model.n_head")
    if cfg.steps <= 0 or cfg.batch_size <= 0:
        raise ValueError("steps and batch_size must be positive")
    if cfg.ckpt_every <= 0 or cfg.keep_last <= 0:
        raise ValueError("ckpt_every and keep_last must be positive")
    if cfg.straggler_threshold <= 1.0:
        raise ValueError("straggler_threshold must be > 1.0")

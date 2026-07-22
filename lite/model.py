"""Compact action-conditioned world model (the part experiments optimize).

EDITABLE: this file and config.yaml are the levers. Experiments change the
architecture / conditioning / precision here and are judged by the frozen
benchmark. The model predicts the next frame from the current frame + action;
rolled out autoregressively it simulates the world.

Design knobs (see config.yaml), each mapped to a hypothesis:
  base_ch, depth      -> backbone width/depth        (hyp-003)
  refine_steps        -> iterative refinement steps   (hyp-001)
  cond_mode           -> heavy context vs compact action conditioning (hyp-002)
  dtype               -> compute/param precision      (hyp-004)
"""
from __future__ import annotations

import dataclasses
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import N_ACTIONS


@dataclasses.dataclass
class ModelConfig:
    base_ch: int = 48
    depth: int = 2                 # number of down/up stages
    refine_steps: int = 3          # bottleneck refinement iterations
    action_emb: int = 32
    cond_mode: str = "action_context"   # "action" | "action_context"
    context_dim: int = 256         # size of the heavy context branch
    dtype: str = "float32"         # "float32" | "float16"
    # training knobs (read by train.py)
    train_steps: int = 900
    batch_size: int = 64
    lr: float = 2.0e-3
    ep_len: int = 12
    seed: int = 0

    @staticmethod
    def load(path: str) -> "ModelConfig":
        with open(path) as f:
            d = yaml.safe_load(f) or {}
        fields = {f.name for f in dataclasses.fields(ModelConfig)}
        return ModelConfig(**{k: v for k, v in d.items() if k in fields})


def torch_dtype(name: str):
    return {"float32": torch.float32, "float16": torch.float16,
            "bfloat16": torch.bfloat16}[name]


class FiLM(nn.Module):
    def __init__(self, cond_dim, ch):
        super().__init__()
        self.fc = nn.Linear(cond_dim, ch * 2)

    def forward(self, x, cond):
        g, b = self.fc(cond).chunk(2, dim=1)
        return x * (1 + g[:, :, None, None]) + b[:, :, None, None]


def _gn(ch):
    return nn.GroupNorm(min(8, ch), ch)


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, cond_dim):
        super().__init__()
        self.n1 = _gn(cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.n2 = _gn(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.film = FiLM(cond_dim, cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, cond):
        h = self.c1(F.silu(self.n1(x)))
        h = self.film(h, cond)
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class ContextEncoder(nn.Module):
    """Heavy-ish global context branch, analogous to the repo's text encoder.
    Present in the baseline; hyp-002 removes it to save memory."""
    def __init__(self, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128, out_dim), nn.SiLU(),
            nn.Linear(out_dim, out_dim), nn.SiLU(),
        )

    def forward(self, frame):
        return self.net(frame)


class WorldModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.act_emb = nn.Embedding(N_ACTIONS, cfg.action_emb)
        cond_dim = cfg.action_emb
        self.ctx = None
        if cfg.cond_mode == "action_context":
            self.ctx = ContextEncoder(cfg.context_dim)
            cond_dim += cfg.context_dim
        self.cond_dim = cond_dim

        chs = [cfg.base_ch * (2 ** i) for i in range(cfg.depth + 1)]
        self.stem = nn.Conv2d(3, chs[0], 3, padding=1)
        self.downs = nn.ModuleList()
        self.down_samp = nn.ModuleList()
        for i in range(cfg.depth):
            self.downs.append(ConvBlock(chs[i], chs[i], cond_dim))
            self.down_samp.append(nn.Conv2d(chs[i], chs[i + 1], 3, 2, 1))
        self.mid = ConvBlock(chs[-1], chs[-1], cond_dim)   # weight-shared refine
        self.ups = nn.ModuleList()
        self.up_samp = nn.ModuleList()
        for i in reversed(range(cfg.depth)):
            self.up_samp.append(nn.Conv2d(chs[i + 1], chs[i], 3, 1, 1))
            self.ups.append(ConvBlock(chs[i] * 2, chs[i], cond_dim))
        self.head = nn.Conv2d(chs[0], 3, 3, padding=1)
        nn.init.zeros_(self.head.weight)   # start at identity (delta=0)
        nn.init.zeros_(self.head.bias)

    def _cond(self, frame, action):
        c = self.act_emb(action)
        if self.ctx is not None:
            c = torch.cat([c, self.ctx(frame)], dim=1)
        return c

    def forward(self, frame, action):
        cond = self._cond(frame, action)
        h = self.stem(frame)
        skips = []
        for blk, ds in zip(self.downs, self.down_samp):
            h = blk(h, cond)
            skips.append(h)
            h = ds(h)
        for _ in range(self.cfg.refine_steps):     # iterative refinement
            h = self.mid(h, cond)
        for us, blk, sk in zip(self.up_samp, self.ups, reversed(skips)):
            h = F.interpolate(h, scale_factor=2, mode="nearest")
            h = us(h)
            h = blk(torch.cat([h, sk], dim=1), cond)
        delta = self.head(h)
        return (frame + delta).clamp(0.0, 1.0)      # residual next-frame

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build(cfg: ModelConfig) -> WorldModel:
    return WorldModel(cfg)

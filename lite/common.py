"""Shared constants and metrics for the lingbot-lite proxy world model.

PROTECTED: experiments must not edit this file. It fixes the evaluation
resolution, action space, and the fidelity metrics so results stay
comparable across the baseline and every experiment variant.
"""
from __future__ import annotations

import os
import numpy as np

# ---- Fixed evaluation settings (frozen; do not change per-experiment) ----
RES = 64                      # frames are RES x RES RGB
N_ACTIONS = 4                 # 0=forward 1=back 2=turn-left 3=turn-right
ACTION_NAMES = ("forward", "back", "turn_left", "turn_right")

# Held-out evaluation episodes are generated from these fixed seeds so the
# benchmark measures the same world every time.
EVAL_SEEDS = tuple(range(9000, 9024))      # 24 eval episodes (full)
EVAL_SEEDS_SCREEN = tuple(range(9000, 9006))  # 6 eval episodes (screening)
EVAL_ROLLOUT = 16             # autoregressive steps scored per episode

# Training data comes from a disjoint seed range (owned by worldgen).
TRAIN_SEEDS = tuple(range(1000, 1240))     # 240 training episodes


def select_device(prefer: str = "auto"):
    import torch
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer in ("mps", "auto") and torch.backends.mps.is_available():
        return torch.device("mps")
    if prefer in ("cuda", "auto") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------- Metrics ----------------------------
def _to_gray(x: np.ndarray) -> np.ndarray:
    # x: (...,H,W,3) float [0,1] -> (...,H,W)
    w = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return (x * w).sum(-1)


def psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    mse = float(np.mean((pred - gt) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * np.log10(1.0 / mse))


def mse(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean((pred - gt) ** 2))


def ssim(pred: np.ndarray, gt: np.ndarray, win: int = 7) -> float:
    """Windowed SSIM on luminance, uniform window. Self-contained (no skimage).

    pred/gt: (N,H,W,3) or (H,W,3) float in [0,1]. Returns mean SSIM.
    """
    if pred.ndim == 3:
        pred = pred[None]
        gt = gt[None]
    p = _to_gray(pred.astype(np.float32))
    g = _to_gray(gt.astype(np.float32))
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2
    k = win
    pad = k // 2

    def box(a):
        # mean filter via cumulative sum, reflect-pad
        a = np.pad(a, ((0, 0), (pad, pad), (pad, pad)), mode="reflect")
        cs = np.cumsum(np.cumsum(a, axis=1), axis=2)
        cs = np.pad(cs, ((0, 0), (1, 0), (1, 0)), mode="constant")
        H = a.shape[1] - 2 * pad
        W = a.shape[2] - 2 * pad
        out = (cs[:, k:k + H, k:k + W] - cs[:, 0:H, k:k + W]
               - cs[:, k:k + H, 0:W] + cs[:, 0:H, 0:W])
        return out / (k * k)

    mu_p = box(p)
    mu_g = box(g)
    mu_p2 = mu_p * mu_p
    mu_g2 = mu_g * mu_g
    mu_pg = mu_p * mu_g
    sig_p = box(p * p) - mu_p2
    sig_g = box(g * g) - mu_g2
    sig_pg = box(p * g) - mu_pg
    num = (2 * mu_pg + C1) * (2 * sig_pg + C2)
    den = (mu_p2 + mu_g2 + C1) * (sig_p + sig_g + C2)
    return float(np.mean(num / den))

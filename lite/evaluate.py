"""Benchmark the trained world model -- the frozen evaluation.

PROTECTED: defines the metrics and writes the machine-readable result the
ResearchForge contract reads. Experiments may not edit this file.

Metrics (all on held-out episodes with fixed seeds):
  ssim, psnr, mse      -- autoregressive rollout fidelity vs ground truth
  action_following     -- fraction of steps where the model's predicted next
                          frame (given the true current frame + taken action)
                          is closest to the correct action's ground-truth
                          outcome among all candidate actions (What-If style)
  latency_ms_per_frame -- mean wall-clock per predicted frame on the device
  peak_ram_mb          -- peak process RSS during the benchmark
  params, model_mb     -- model size
"""
from __future__ import annotations

import argparse
import json
import os
import time
import numpy as np
import torch

from .common import (EVAL_SEEDS, EVAL_SEEDS_SCREEN, EVAL_ROLLOUT, N_ACTIONS,
                     select_device, ssim as ssim_fn, psnr as psnr_fn, mse as mse_fn)
from .model import ModelConfig, build, torch_dtype
from . import worldgen

try:
    import psutil
    _PROC = psutil.Process()
except Exception:
    _PROC = None


def _peak_rss_mb():
    if _PROC is not None:
        return _PROC.memory_info().rss / 1e6
    import resource
    m = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return m / 1e6 if m > 1e6 else m / 1e3  # linux KB vs mac bytes


@torch.no_grad()
def evaluate(ckpt_path, device_str="auto", screening=False):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ck["config"])
    device = select_device(device_str)
    dtype = torch_dtype(cfg.dtype)
    model = build(cfg).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    cast = dtype in (torch.float16, torch.bfloat16)
    if cast:
        model = model.to(dtype)

    seeds = EVAL_SEEDS_SCREEN if screening else EVAL_SEEDS
    peak = _peak_rss_mb()

    def predict(frame_chw, action_int):
        x = frame_chw.unsqueeze(0).to(device)
        if cast:
            x = x.to(dtype)
        a = torch.tensor([action_int], device=device)
        out = model(x, a).float().clamp(0, 1)
        return out[0].cpu()

    # ---- action-following (teacher-forced, single step) ----
    af_correct = af_total = 0
    for s in seeds:
        frames, actions, cands = worldgen.make_episode(s, EVAL_ROLLOUT)
        for t in range(len(actions)):
            cur = torch.from_numpy(frames[t]).permute(2, 0, 1).contiguous()
            pred = predict(cur, int(actions[t])).permute(1, 2, 0).numpy()
            cand = cands[t]  # (N_ACTIONS,H,W,3)
            d = ((cand - pred[None]) ** 2).reshape(N_ACTIONS, -1).mean(1)
            if int(np.argmin(d)) == int(actions[t]):
                af_correct += 1
            af_total += 1
        peak = max(peak, _peak_rss_mb())

    # ---- autoregressive rollout fidelity + latency ----
    preds, gts = [], []
    # warmup
    _ = predict(torch.from_numpy(worldgen.make_episode(seeds[0], 1)[0][0])
                .permute(2, 0, 1).contiguous(), 0)
    n_frames = 0
    t0 = time.time()
    for s in seeds:
        frames, actions, _ = worldgen.make_episode(s, EVAL_ROLLOUT)
        cur = torch.from_numpy(frames[0]).permute(2, 0, 1).contiguous()
        for t in range(len(actions)):
            cur = predict(cur, int(actions[t]))
            preds.append(cur.permute(1, 2, 0).numpy())
            gts.append(frames[t + 1])
            n_frames += 1
        peak = max(peak, _peak_rss_mb())
    if device.type in ("mps", "cuda"):
        getattr(torch, device.type).synchronize()
    latency_ms = (time.time() - t0) / max(1, n_frames) * 1000.0

    preds = np.stack(preds); gts = np.stack(gts)
    params = model.num_params()
    # ResearchForge result schema v1: primary_metric + secondary_metrics dict.
    result = {
        "schema_version": 1,
        # primary = params (minimize): the "lightweight" objective. Latency is
        # overhead-bound at this scale, so it is a secondary/constraint metric.
        "primary_metric": {
            "name": "params",
            "value": float(params),
        },
        "secondary_metrics": {
            "latency_ms_per_frame": round(latency_ms, 4),
            "ssim": round(ssim_fn(preds, gts), 5),
            "action_following": round(af_correct / max(1, af_total), 5),
            "peak_ram_mb": round(peak, 2),
            "psnr": round(psnr_fn(preds, gts), 4),
            "mse": round(mse_fn(preds, gts), 6),
            "model_mb": round(params * (2 if cast else 4) / 1e6, 3),
        },
        "sample_count": int(n_frames),
        "seed": int(cfg.seed),
        "metadata": {
            "device": device.type,
            "screening": bool(screening),
            "cond_mode": cfg.cond_mode,
            "dtype": cfg.dtype,
        },
    }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="lite/_ckpt/model.pt")
    ap.add_argument("--out", default="lite/_out/result.json")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--screening", action="store_true")
    args = ap.parse_args()

    res = evaluate(args.ckpt, args.device, args.screening)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

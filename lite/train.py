"""Train the world model on ground-truth transitions.

EDITABLE: experiments may change the training recipe. Produces a checkpoint
consumed by evaluate.py. Kept fast so the full screening->benchmark funnel is
runnable on a MacBook (MPS/CPU).
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import time
import numpy as np
import torch

from .common import TRAIN_SEEDS, select_device
from .model import ModelConfig, build, torch_dtype
from . import worldgen


def get_dataset(seeds, ep_len, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    key = f"train_{len(seeds)}_{ep_len}_{seeds[0]}.npz"
    path = os.path.join(cache_dir, key)
    if os.path.exists(path):
        d = np.load(path)
        return d["f0"], d["a"], d["f1"]
    f0, a, f1 = worldgen.build_transition_dataset(seeds, ep_len)
    np.savez_compressed(path, f0=f0, a=a, f1=f1)
    return f0, a, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="lite/config.yaml")
    ap.add_argument("--ckpt", default="lite/_ckpt/model.pt")
    ap.add_argument("--data-dir", default="lite/_data")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--screening", action="store_true")
    args = ap.parse_args()

    cfg = ModelConfig.load(args.config)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = select_device(args.device)
    dtype = torch_dtype(cfg.dtype)

    seeds = TRAIN_SEEDS[:40] if args.screening else TRAIN_SEEDS
    # Screening must train long enough for SMALLER models to converge, or the
    # hard-constraint check false-rejects them (they need more steps than the
    # baseline to reach the same action-following). 600 is a fair filter budget.
    steps = min(cfg.train_steps, 600) if args.screening else cfg.train_steps

    f0, a, f1 = get_dataset(seeds, cfg.ep_len, args.data_dir)
    f0 = torch.from_numpy(f0).permute(0, 3, 1, 2).contiguous()
    f1 = torch.from_numpy(f1).permute(0, 3, 1, 2).contiguous()
    a = torch.from_numpy(a)
    N = f0.shape[0]

    model = build(cfg).to(device)
    if dtype == torch.float16 and device.type == "mps":
        pass  # keep params fp32 on mps; dtype used for eval-time cast only
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    print(f"[train] device={device} params={model.num_params():,} "
          f"transitions={N} steps={steps} screening={args.screening}", flush=True)
    model.train()
    t0 = time.time()
    for it in range(steps):
        idx = torch.randint(0, N, (cfg.batch_size,))
        x = f0[idx].to(device)
        y = f1[idx].to(device)
        ac = a[idx].to(device)
        pred = model(x, ac)
        loss = torch.nn.functional.l1_loss(pred, y) + \
            torch.nn.functional.mse_loss(pred, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if it % max(1, steps // 6) == 0 or it == steps - 1:
            print(f"[train] step {it:4d}/{steps} loss={loss.item():.4f} "
                  f"({time.time()-t0:.1f}s)", flush=True)

    os.makedirs(os.path.dirname(args.ckpt), exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "config": dataclasses.asdict(cfg)}, args.ckpt)
    print(f"[train] saved {args.ckpt} in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()

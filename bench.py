#!/usr/bin/env python3
"""Canonical benchmark for the LingBot-World Fast inference pipeline.

One configuration, fixed seed, one number: feed every optimization through
this script so commits can be compared apples-to-apples.

Usage:
    MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 \\
        .venv/bin/torchrun --nproc_per_node=8 \\
        --master_addr=127.0.0.1 --master_port=29500 bench.py

Output:
    bench/results/<short-git-sha>.json    timings + output MD5
    bench/videos/<short-git-sha>.mp4      the generated clip (gitignored)
"""

import hashlib
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
from PIL import Image

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

import wan  # noqa: E402
from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS  # noqa: E402
from wan.distributed.util import init_distributed_group  # noqa: E402
from wan.utils.utils import save_video  # noqa: E402


# ── Fixed canonical configuration ───────────────────────────────────────
# Change these only if the entire benchmark series is being re-baselined.
CONFIG = {
    "task": "i2v-A14B",
    "size": "480*832",
    "ckpt_dir": "lingbot-world-base-cam",
    "image": "examples/03/image.jpg",
    "action_path": "examples/03",
    "frame_num": 81,
    "base_seed": 42,
    "prompt": (
        "A serene lakeside scene with a lone tree standing in calm water, "
        "surrounded by distant snow-capped mountains under a bright blue "
        "sky with drifting white clouds — gentle ripples reflect the tree "
        "and sky, creating a tranquil, meditative atmosphere."
    ),
    "hardware": "8x H100 80GB",
}


class _PhaseCapture(logging.Handler):
    """Scrapes '[PROFILE] name: 1234.5 ms' lines into a dict, rank-0 only."""

    def __init__(self):
        super().__init__(level=logging.INFO)
        self.phases: dict[str, float] = {}

    def emit(self, record):
        msg = record.getMessage()
        marker = "[PROFILE] "
        if marker not in msg:
            return
        payload = msg.split(marker, 1)[1].strip()
        try:
            name, value = payload.rsplit(":", 1)
            ms = float(value.strip().split()[0])
            self.phases[name.strip()] = ms
        except Exception:
            # If the format ever drifts, fail soft — bench should still complete.
            pass


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True
        ).strip()
    except Exception:
        return "unknown"


def _is_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO, text=True
        )
        return bool(out.strip())
    except Exception:
        return False


def main():
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    init_distributed_group()

    # Logging: rank 0 prints everything + captures PROFILE lines; others stay quiet.
    capture = _PhaseCapture()
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)],
        )
        logging.getLogger().addHandler(capture)
    else:
        logging.basicConfig(level=logging.ERROR)

    sha = _git_sha()
    dirty = _is_dirty()
    if rank == 0:
        logging.info(f"bench.py @ {sha}{' (dirty)' if dirty else ''}")
        logging.info(f"config: {json.dumps(CONFIG, indent=2)}")

    cfg = WAN_CONFIGS[CONFIG["task"]]
    img = Image.open(CONFIG["image"]).convert("RGB")

    pipe = wan.WanI2VFast(
        config=cfg,
        checkpoint_dir=CONFIG["ckpt_dir"],
        device_id=local_rank,
        rank=rank,
        t5_fsdp=True,
        dit_fsdp=True,
        use_sp=True,
    )

    # Total generate() timer — wall-clock across all ranks (sync first).
    if dist.is_initialized():
        torch.cuda.synchronize()
        dist.barrier()
    import time
    t0 = time.perf_counter()

    video = pipe.generate(
        CONFIG["prompt"],
        img,
        action_path=CONFIG["action_path"],
        max_area=MAX_AREA_CONFIGS[CONFIG["size"]],
        frame_num=CONFIG["frame_num"],
        shift=cfg.sample_shift,
        seed=CONFIG["base_seed"],
        offload_model=False,
    )

    if dist.is_initialized():
        torch.cuda.synchronize()
        dist.barrier()
    total_ms = (time.perf_counter() - t0) * 1000.0

    if rank == 0:
        bench_dir = REPO / "bench"
        results_dir = bench_dir / "results"
        videos_dir = bench_dir / "videos"
        results_dir.mkdir(parents=True, exist_ok=True)
        videos_dir.mkdir(parents=True, exist_ok=True)

        mp4_path = videos_dir / f"{sha}.mp4"
        save_video(
            tensor=video[None],
            save_file=str(mp4_path),
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )
        md5 = hashlib.md5(mp4_path.read_bytes()).hexdigest()

        result = {
            "git_sha": sha,
            "git_dirty": dirty,
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "config": CONFIG,
            "phases_ms": dict(sorted(capture.phases.items())),
            "total_generate_ms": round(total_ms, 1),
            "output_md5": md5,
            "output_mp4": str(mp4_path.relative_to(REPO)),
        }
        result_path = results_dir / f"{sha}.json"
        result_path.write_text(json.dumps(result, indent=2) + "\n")

        logging.info("")
        logging.info("=== bench result ===")
        logging.info(json.dumps(result, indent=2))
        logging.info(f"wrote {result_path.relative_to(REPO)}")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

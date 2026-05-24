"""Persistent-pipeline example: amortize model load + warmup across many prompts.

Pattern borrowed from LLM serving (vLLM, SGLang): don't construct the engine
per request. Construct WanI2VFast once, call prewarm() once, then loop over
prompts. Cold-start cost (~100 s model load + ~7 s prewarm) is paid once;
each subsequent generate() runs at steady-state speed.

Usage:
    MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 \\
        torchrun --nproc_per_node=8 \\
        --master_addr=127.0.0.1 --master_port=29500 \\
        examples/persistent_inference.py \\
        --ckpt_dir lingbot-world-base-cam \\
        --image examples/03/image.jpg \\
        --action_path examples/03 \\
        --save_dir output

Three prompts run back-to-back. Output: output/persistent_{0,1,2}.mp4.

Wall-clock expectation (8xH100, 480*832, 81 frames):
    Naive (3 separate `generate_fast.py` invocations): ~3 x 126 s ~= 378 s.
    Persistent (this script):                          ~104 s + 7 s + 3 x 15.5 s ~= 158 s.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Allow running this file directly from the repo root: add the parent dir
# (repo root) to sys.path so `import wan` resolves to the in-tree package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.distributed as dist
from PIL import Image

import wan  # noqa: E402
from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS  # noqa: E402
from wan.distributed.util import init_distributed_group  # noqa: E402
from wan.utils.utils import save_video  # noqa: E402


PROMPTS = [
    "A serene lakeside scene with a lone tree standing in calm water, "
    "surrounded by distant snow-capped mountains under a bright blue sky "
    "with drifting white clouds — gentle ripples reflect the tree and "
    "sky, creating a tranquil, meditative atmosphere.",
    "A sweeping cinematic journey along the Great Wall of China, winding "
    "through golden autumn hills under a brilliant blue sky — stone "
    "pathways stretch into the distance, watchtowers stand sentinel.",
    "Aerial flight over a vast tropical rainforest at dawn, mist rising "
    "from the canopy, sunlight breaking through tall trees, a winding "
    "river snaking through the green expanse below.",
]


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--task", default="i2v-A14B",
                        choices=list(WAN_CONFIGS.keys()))
    parser.add_argument("--size", default="480*832")
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--action_path", default=None)
    parser.add_argument("--frame_num", type=int, default=81)
    parser.add_argument("--chunk_size", type=int, default=3)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--save_dir", default="output")
    parser.add_argument("--ulysses_size", type=int, default=8)
    return parser.parse_args()


def _init_distributed():
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        init_distributed_group()
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)],
        )
    else:
        logging.basicConfig(level=logging.ERROR)
    return rank, local_rank, world_size


def main():
    args = _parse_args()
    rank, local_rank, world_size = _init_distributed()

    cfg = WAN_CONFIGS[args.task]
    img = Image.open(args.image).convert("RGB")

    if rank == 0:
        logging.info(f"Persistent inference: {len(PROMPTS)} prompts, "
                     f"size={args.size}, frame_num={args.frame_num}")

    # ── Cold start: construct pipe once. Paid before the timed window. ──
    t_construct = time.perf_counter()
    pipe = wan.WanI2VFast(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=local_rank,
        rank=rank,
        t5_fsdp=True,
        dit_fsdp=True,
        use_sp=(args.ulysses_size > 1),
    )
    construct_ms = (time.perf_counter() - t_construct) * 1000.0
    if rank == 0:
        logging.info(f"WanI2VFast() construction: {construct_ms:.0f} ms")

    # ── One-time prewarm. After this, generate() runs at steady state. ──
    t_warm = time.perf_counter()
    pipe.prewarm(
        img,
        max_area=MAX_AREA_CONFIGS[args.size],
        frame_num=args.frame_num,
        chunk_size=args.chunk_size,
    )
    warm_ms = (time.perf_counter() - t_warm) * 1000.0
    if rank == 0:
        logging.info(f"prewarm(): {warm_ms:.0f} ms")

    # ── Generate loop. Each call should run at amortized steady-state cost. ──
    save_dir = Path(args.save_dir)
    if rank == 0:
        save_dir.mkdir(parents=True, exist_ok=True)

    per_call_ms = []
    for i, prompt in enumerate(PROMPTS):
        if dist.is_initialized():
            torch.cuda.synchronize()
            dist.barrier()
        t0 = time.perf_counter()

        video = pipe.generate(
            prompt,
            img,
            action_path=args.action_path,
            chunk_size=args.chunk_size,
            max_area=MAX_AREA_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=cfg.sample_shift,
            seed=args.base_seed + i,
            offload_model=False,
        )

        if dist.is_initialized():
            torch.cuda.synchronize()
            dist.barrier()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        per_call_ms.append(elapsed_ms)

        if rank == 0:
            out_path = save_dir / f"persistent_{i}.mp4"
            save_video(
                tensor=video[None],
                save_file=str(out_path),
                fps=cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1),
            )
            logging.info(
                f"prompt[{i}]: generate() {elapsed_ms:.0f} ms -> {out_path}")

    if rank == 0:
        total = construct_ms + warm_ms + sum(per_call_ms)
        logging.info("=" * 60)
        logging.info(f"SUMMARY  prompts={len(PROMPTS)}  hardware=8xH100")
        logging.info(f"  construct:        {construct_ms:>8.0f} ms (once)")
        logging.info(f"  prewarm:          {warm_ms:>8.0f} ms (once)")
        for i, ms in enumerate(per_call_ms):
            logging.info(f"  generate[{i}]:      {ms:>8.0f} ms")
        avg = sum(per_call_ms) / len(per_call_ms)
        logging.info(f"  generate avg:     {avg:>8.0f} ms")
        logging.info(f"  total wall-clock: {total:>8.0f} ms")
        naive_est = construct_ms + warm_ms + len(PROMPTS) * (construct_ms + avg)
        logging.info(f"  naive estimate (separate invocations): {naive_est:>8.0f} ms")
        logging.info(f"  speedup vs naive: {naive_est / total:.2f}x")
        logging.info("=" * 60)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

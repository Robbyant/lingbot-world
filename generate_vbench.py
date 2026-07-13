import argparse
import csv
import json
import logging
import os
import re
import sys
import time
import warnings
from datetime import datetime

# Disable torch compile to avoid inductor import errors
os.environ['TORCHDYNAMO_DISABLE'] = '1'

warnings.filterwarnings('ignore')

import random

import torch
import torch.distributed as dist
from PIL import Image

import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.distributed.util import init_distributed_group
from wan.utils.utils import merge_video_audio, save_video, str2bool


_SCRIPT_DIR        = os.path.dirname(os.path.abspath(__file__))
_VBENCH_ROOT       = os.path.join(_SCRIPT_DIR, "..", "VBench", "vbench2_beta_i2v")
_DEFAULT_INFO_JSON = os.path.join(_VBENCH_ROOT, "vbench2_i2v_full_info.json")
_DEFAULT_CROP_DIR  = os.path.join(_VBENCH_ROOT, "vbench2_beta_i2v", "data", "crop")
def _safe(s):
    return re.sub(r'[<>:"/\\|?*]', "_", s)[:150]


EXAMPLE_PROMPT = {
    "i2v-A14B": {
        "prompt":
            "The video presents a cinematic, first-person wandering experience through a hyper-realistic urban environment rendered in a video game engine. It begins with a static, sun-drenched alley framed by graffiti-laden industrial walls and overhead power lines, immediately establishing a gritty, lived-in atmosphere. As the camera pans right and tilts upward, it reveals a sprawling cityscape dominated by towering skyscrapers and industrial infrastructure, all bathed in warm, late-afternoon light that casts long shadows and produces dramatic lens flares. The perspective then transitions into a smooth forward tracking shot along a cracked sidewalk, passing weathered fences, palm trees, and distant pedestrians, creating a sense of immersion and exploration. Midway, the camera briefly follows a walking figure before refocusing on the broader streetscape, culminating in a stabilized view of a small blue van parked at an intersection surrounded by urban elements like parking garages and traffic lights. The entire sequence is characterized by its photorealistic detail, dynamic lighting, and deliberate pacing, evoking the feel of a quiet, sunlit afternoon in a futuristic metropolis.",
        "image":
            "examples/02/image.jpg",
    },
}


def _validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"
    assert args.task in EXAMPLE_PROMPT, f"Unsupport task: {args.task}"

    if args.prompt is None:
        args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
    if args.image is None and "image" in EXAMPLE_PROMPT[args.task]:
        args.image = EXAMPLE_PROMPT[args.task]["image"]

    if args.task == "i2v-A14B":
        assert args.image is not None, "Please specify the image path for i2v."

    cfg = WAN_CONFIGS[args.task]

    if args.sample_steps is None:
        args.sample_steps = cfg.sample_steps

    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift

    if args.sample_guide_scale is None:
        args.sample_guide_scale = cfg.sample_guide_scale

    if args.frame_num is None:
        args.frame_num = cfg.frame_num

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(
        0, sys.maxsize)
    # Size check
    if not 's2v' in args.task:
        assert args.size in SUPPORTED_SIZES[
            args.
            task], f"Unsupport size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="i2v-A14B",
        choices=list(WAN_CONFIGS.keys()),
        help="The task to run.")
    parser.add_argument(
        "--size",
        type=str,
        default="1280*720",
        choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image."
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=81,
        help="How many frames of video are generated. The number should be 4n+1 (default: 81)"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage.")
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.")
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.")
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="The file to save the generated video to.")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="The prompt to generate the video from.")
    parser.add_argument(
        "--use_prompt_extend",
        action="store_true",
        default=False,
        help="Whether to use prompt extend.")
    parser.add_argument(
        "--prompt_extend_method",
        type=str,
        default="local_qwen",
        choices=["dashscope", "local_qwen"],
        help="The prompt extend method to use.")
    parser.add_argument(
        "--prompt_extend_model",
        type=str,
        default=None,
        help="The prompt extend model to use.")
    parser.add_argument(
        "--prompt_extend_target_lang",
        type=str,
        default="zh",
        choices=["zh", "en"],
        help="The target language of prompt extend.")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=42,
        help="The seed to use for generating the video.")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="The image to generate the video from.")
    parser.add_argument(
        "--action_path",
        type=str,
        default=None,
        help="The camera path to generate the video from.")
    parser.add_argument(
        "--sample_solver",
        type=str,
        default='unipc',
        choices=['unipc', 'dpm++'],
        help="The solver used to sample.")
    parser.add_argument(
        "--sample_steps", type=int, default=None, help="The sampling steps.")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=None,
        help="Classifier free guidance scale.")
    parser.add_argument(
        "--convert_model_dtype",
        action="store_true",
        default=False,
        help="Whether to convert model paramerters dtype.")
    # ---- VBench batch args ----
    parser.add_argument("--vbench", action="store_true", default=True,
        help="Run VBench batch generation instead of single-video mode.")
    parser.add_argument("--image_types", type=str, default="indoor,scenery",
        help="Comma-separated image_type values to include (default: scenery,indoor).")
    parser.add_argument("--vbench_output_dir", type=str, default="results_vbench/videos",
        help="Output directory for vbench videos.")
    parser.add_argument("--num_samples", type=int, default=5,
        help="Number of samples per prompt.")
    parser.add_argument("--vbench_info_json", type=str, default=None,
        help="Path to vbench2_i2v_full_info.json.")
    parser.add_argument("--crop_dir", type=str, default=None,
        help="Path to VBench crop directory.")
    parser.add_argument("--resolution", type=str, default="1-1",
        help="Crop resolution subfolder.")

    args = parser.parse_args()
    if args.vbench:
        assert args.ckpt_dir is not None, "Please specify --ckpt_dir (path to Wan checkpoint directory)."
    else:
        _validate_args(args)

    return args


def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


def generate(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    logging.info("Starting the generation process...")
    
    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(
            f"offload_model is not specified, set to {args.offload_model}.")
    if world_size > 1:
        logging.info("Initializing distributed environment...")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)
        logging.info("Distributed environment initialized.")
    else:
        assert not (
            args.t5_fsdp or args.dit_fsdp
        ), f"t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (
            args.ulysses_size > 1
        ), f"sequence parallel are not supported in non-distributed environments."

    if args.ulysses_size > 1:
        assert args.ulysses_size == world_size, f"The number of ulysses_size should be equal to the world size."
        init_distributed_group()

    logging.info("Loading model configuration...")
    cfg = WAN_CONFIGS[args.task]
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, f"`{cfg.num_heads=}` cannot be divided evenly by `{args.ulysses_size=}`."

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    logging.info(f"Input prompt: {args.prompt}")
    img = None
    if args.image is not None:
        logging.info(f"Loading input image from {args.image}...")
        img = Image.open(args.image).convert("RGB")
        logging.info("Input image loaded.")

    # prompt extend
    if args.use_prompt_extend:
        logging.info("Extending prompt...")
        if rank == 0:
            input_prompt = args.prompt
            input_prompt = [input_prompt]
        else:
            input_prompt = [None]
        if dist.is_initialized():
            dist.broadcast_object_list(input_prompt, src=0)
        args.prompt = input_prompt[0]
        logging.info(f"Extended prompt: {args.prompt}")
    
    logging.info("Creating WanI2V pipeline...")
    wan_i2v = wan.WanI2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=device,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_sp=(args.ulysses_size > 1),
        t5_cpu=args.t5_cpu,
        convert_model_dtype=args.convert_model_dtype,
    )
    logging.info("WanI2V pipeline created.")

    logging.info("Generating video...")
    video = wan_i2v.generate(
        args.prompt,
        img,
        action_path=args.action_path,
        max_area=MAX_AREA_CONFIGS[args.size],
        frame_num=args.frame_num,
        shift=args.sample_shift,
        sample_solver=args.sample_solver,
        sampling_steps=args.sample_steps,
        guide_scale=args.sample_guide_scale,
        seed=args.base_seed,
        offload_model=args.offload_model)
    logging.info("Video generation completed.")

    if rank == 0:
        if args.save_file is None:
            formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            formatted_prompt = args.prompt.replace(" ", "_").replace("/",
                                                                     "_")[:50]
            suffix = '.mp4'
            args.save_file = f"{args.task}_{args.size.replace('*','x') if sys.platform=='win32' else args.size}_{args.ulysses_size}_{formatted_prompt}_{formatted_time}" + suffix

        logging.info(f"Saving generated video to {args.save_file}...")
        save_video(
            tensor=video[None],
            save_file=args.save_file,
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1))
        if "s2v" in args.task:
            if args.enable_tts is False:
                merge_video_audio(video_path=args.save_file, audio_path=args.audio)
            else:
                merge_video_audio(video_path=args.save_file, audio_path="tts.wav")
        logging.info("Video saved successfully.")
    del video

    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    logging.info("Generation process finished.")


def vbench_batch(args):
    info_json = os.path.abspath(args.vbench_info_json or _DEFAULT_INFO_JSON)
    crop_base = os.path.abspath(args.crop_dir or _DEFAULT_CROP_DIR)
    image_dir = os.path.join(crop_base, args.resolution)
    out_dir   = os.path.abspath(args.vbench_output_dir)
    os.makedirs(out_dir, exist_ok=True)

    stats_path = os.path.join(os.path.dirname(out_dir), 'vbench_stats.csv')
    stats_f    = open(stats_path, 'w', newline='', encoding='utf-8')
    stats_w    = csv.writer(stats_f)
    stats_w.writerow(['task_idx', 'prompt', 'sample_idx', 'duration_s', 'gen_fps', 'out_path', 'status'])

    if not os.path.isfile(info_json):
        print(f'[vbench] ERROR: info JSON not found: {info_json}'); return
    if not os.path.isdir(image_dir):
        print(f'[vbench] ERROR: crop dir not found: {image_dir}'); return

    with open(info_json, encoding='utf-8') as f:
        entries = json.load(f)

    allowed  = {t.strip() for t in args.image_types.split(',') if t.strip()} if args.image_types else None
    populate = None

    seen, prompts = set(), []
    for e in entries:
        name = e['image_name']
        if name in seen: continue
        if allowed and e.get('image_type') not in allowed: continue
        if populate is not None and (e.get('image_type') in _POPULATED_TYPES) != populate: continue
        seen.add(name)
        prompts.append((name, e['prompt_en']))

    print(f'[vbench] {len(prompts)} prompts × {args.num_samples} samples = {len(prompts) * args.num_samples} total')

    cfg = WAN_CONFIGS[args.task]
    wan_i2v = wan.WanI2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=args.t5_cpu,
        convert_model_dtype=args.convert_model_dtype,
    )

    skipped = generated = errors = 0
    total   = len(prompts) * args.num_samples
    done    = 0
    t_start = time.time()
    print(f'[vbench] {len(prompts)} prompts × {args.num_samples} samples = {total} total')

    for task_idx, (image_name, prompt) in enumerate(prompts):
        image_path = os.path.join(image_dir, image_name)
        if not os.path.isfile(image_path):
            print(f'[vbench] skip {task_idx}: image not found — {image_path}')
            continue

        img = Image.open(image_path).convert("RGB")

        for sample_idx in range(args.num_samples):
            out_path = os.path.join(out_dir, f'{_safe(prompt)}-{sample_idx}.mp4')
            if os.path.exists(out_path):
                skipped += 1
                done += 1
                stats_w.writerow([task_idx, prompt, sample_idx, '', '', out_path, 'skipped'])
                stats_f.flush()
                continue

            pct = 100 * done / total if total else 0
            eta = ''
            if done > 0:
                secs_left = (time.time() - t_start) / done * (total - done)
                eta = f'  ETA {int(secs_left//3600):02d}h{int(secs_left%3600//60):02d}m{int(secs_left%60):02d}s'
            print(f'[vbench] [{done+1}/{total}  {pct:.0f}%{eta}]  prompt {task_idx+1}/{len(prompts)}  sample {sample_idx+1}/{args.num_samples}: {prompt[:50]}')
            seed = args.base_seed + sample_idx
            try:
                with torch.inference_mode():
                    t0 = time.time()
                    video = wan_i2v.generate(
                        prompt, img,
                        max_area=MAX_AREA_CONFIGS[args.size],
                        frame_num=args.frame_num,
                        shift=args.sample_shift or cfg.sample_shift,
                        sample_solver=args.sample_solver,
                        sampling_steps=args.sample_steps or cfg.sample_steps,
                        guide_scale=args.sample_guide_scale or cfg.sample_guide_scale,
                        seed=seed,
                        offload_model=True,
                    )
                    elapsed = time.time() - t0
                frame_num = args.frame_num
                gen_fps = frame_num / elapsed if elapsed > 0 else 0.0
                from wan.utils.utils import save_video as _save_video
                _save_video(tensor=video[None], save_file=out_path, fps=cfg.sample_fps,
                            nrow=1, normalize=True, value_range=(-1, 1))
                print(f'[vbench] saved {out_path}  ({gen_fps:.1f} gen-fps)')
                stats_w.writerow([task_idx, prompt, sample_idx, f'{elapsed:.2f}', f'{gen_fps:.2f}', out_path, 'ok'])
                stats_f.flush()
                generated += 1
            except Exception as exc:
                print(f'[vbench] ERROR task {task_idx} sample {sample_idx}: {exc}')
                stats_w.writerow([task_idx, prompt, sample_idx, '', '', out_path, 'error'])
                stats_f.flush()
                errors += 1
            done += 1

    elapsed_total = time.time() - t_start
    stats_f.close()
    print(f'\n[vbench] done — generated={generated}  skipped={skipped}  errors={errors}  elapsed={elapsed_total/60:.1f}m')
    print(f'[vbench] stats → {stats_path}')


if __name__ == "__main__":
    args = _parse_args()
    if args.vbench:
        vbench_batch(args)
    else:
        generate(args)

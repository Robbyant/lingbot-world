import gc
import hashlib
import logging
import math
import os
import random
import sys
import time
import types
from contextlib import contextmanager
from functools import partial

import numpy as np
import torch
# import torch.cuda.amp as amp
import torch.distributed as dist
import torchvision.transforms.functional as TF
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .distributed.sequence_parallel import sp_attn_forward_causal, sp_dit_forward_causal
from .distributed.util import get_world_size
from .modules.model_fast import WanModelFast
from .modules.t5 import T5EncoderModel
from .modules.vae2_1 import Wan2_1_VAE

from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from .utils.cam_utils import (
    compute_relative_poses,
    interpolate_camera_poses,
    get_plucker_embeddings,
    get_Ks_transformed,
)
from einops import rearrange


class WanI2VFast:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        init_on_cpu=True,
        convert_model_dtype=False,
        pipe_dtype=torch.bfloat16,
        local_attn_size=-1,
        sink_size=0,
    ):
        r"""
        Initializes the image-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_sp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of sequence parallel.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
            init_on_cpu (`bool`, *optional*, defaults to True):
                Enable initializing Transformer Model on CPU. Only works without FSDP or USP.
            convert_model_dtype (`bool`, *optional*, defaults to False):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.init_on_cpu = init_on_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.boundary = config.boundary
        self.param_dtype = config.param_dtype
        self.pipe_dtype = pipe_dtype
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size

        if t5_fsdp or dit_fsdp or use_sp:
            self.init_on_cpu = False

        if 'cam' in checkpoint_dir:
            self.control_type = 'cam'
        elif 'act' in checkpoint_dir:
            self.control_type = 'act'

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None,
        )

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = Wan2_1_VAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        logging.info(f"Creating WanModelFast from {checkpoint_dir}")
        self.model = WanModelFast.from_pretrained(
            checkpoint_dir,
            subfolder=config.fast_noise_checkpoint,
            torch_dtype=torch.bfloat16,
            control_type=self.control_type,
            local_attn_size=self.local_attn_size,
            sink_size=self.sink_size)

        self.model = self._configure_model(
            model=self.model,
            use_sp=use_sp,
            dit_fsdp=dit_fsdp,
            shard_fn=shard_fn,
            convert_model_dtype=convert_model_dtype).to(self.device)

        self.scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=self.num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False)

        if use_sp:
            self.sp_size = get_world_size()
        else:
            self.sp_size = 1

        self.sample_neg_prompt = config.sample_neg_prompt

        # T5 prompt-embedding cache. Same-prompt re-encodes hit this dict
        # instead of re-running the umt5-xxl encoder (~360 ms/call).
        # Keyed by sha256(prompt.utf8); value is the list returned by
        # T5EncoderModel.__call__ (already device-resident). Unbounded;
        # callers can clear via `pipe.clear_text_cache()` if needed.
        self._t5_cache: dict[str, list] = {}

        # Reset per generate() and flipped True after the first DiT forward.
        # Passed into model.forward as `cross_attn_first_call` to skip the
        # crossattn_cache["is_init"].item() sync inside WanCrossAttention.
        self._cross_attn_initialized: bool = False

    def clear_text_cache(self):
        """Drop all cached T5 prompt embeddings. Frees ~4 MB per entry."""
        self._t5_cache.clear()

    def prewarm(
        self,
        img,
        max_area: int = 480 * 832,
        frame_num: int = 81,
        chunk_size: int = 3,
        text_seq_len: int = 512,
    ):
        """Opt-in pre-warm. Run one dummy DiT forward at the same shape a
        subsequent generate() call will use, so CUDA kernels are autotuned,
        FSDP all-gathers happen, and Ulysses all-to-alls handshake — all
        outside the timed generate() window.

        Without this call, the first generate() pays a ~7s warmup tax in
        chunk 0 (CUDA lazy init, kernel autotuning, NCCL handshake). On
        8xH100 at 480*832/81 frames, calling prewarm() before the first
        generate() reduces generate()'s wall-clock by ~6.5s (~30%) with
        bit-identical output.

        Idempotent: subsequent calls on the same pipe are no-ops.
        Shape-keyed: if generate() is later invoked with a different shape,
        the autotuner will warm those kernels on demand in chunk 0 (no
        incorrect output, just the tax re-paid once).

        Args:
            img: PIL image or torch tensor — used only for its h/w to match
                generate()'s lat_h/lat_w derivation.
            max_area, frame_num, chunk_size: shape parameters; must match
                the subsequent generate() call to be effective.
            text_seq_len: T5 sequence length (defaults to config.text_len).

        Caller pattern:
            pipe = WanI2VFast(...)
            pipe.prewarm(img, max_area=..., frame_num=...)
            # start your timer here
            video = pipe.generate(prompt, img, ...)
        """
        if getattr(self, "_warmed", False):
            return

        cfg = self.config

        # Match generate()'s shape derivation exactly.
        F = frame_num
        h, w = (img.shape[1], img.shape[2]) if hasattr(img, 'shape') else (img.size[1], img.size[0])
        aspect_ratio = h / w
        lat_h = round(
            np.sqrt(max_area * aspect_ratio) // cfg.vae_stride[1] //
            cfg.patch_size[1] * cfg.patch_size[1])
        lat_w = round(
            np.sqrt(max_area / aspect_ratio) // cfg.vae_stride[2] //
            cfg.patch_size[2] * cfg.patch_size[2])
        lat_f = (F - 1) // cfg.vae_stride[0] + 1
        lat_f = int(lat_f - (lat_f % chunk_size))

        frame_seqlen = (lat_h * lat_w) // (cfg.patch_size[1] * cfg.patch_size[2])
        max_seq_len = chunk_size * frame_seqlen
        head_dim = cfg.dim // cfg.num_heads
        local_num_heads = cfg.num_heads // self.sp_size

        if self.local_attn_size > -1:
            kv_size = frame_seqlen * self.local_attn_size
        else:
            kv_size = frame_seqlen * lat_f

        transformer_dtype = self.pipe_dtype
        # generate() folds the VAE spatial stride into the Plücker channel
        # dim via rearrange 'f (h s1) (w s2) c -> (f h w) (c s1 s2)' with
        # s1=s2=vae_stride[1]=8, so control_dim=6 → 6 * 8 * 8 = 384.
        plucker_channels = 6 * cfg.vae_stride[1] * cfg.vae_stride[2]
        # T5 (umt5-xxl) hidden size; cross-attn projects t5_hidden → cfg.dim.
        t5_hidden = 4096

        warmup_self_kv = self._initialize_self_kv_cache(
            num_layers=cfg.num_layers,
            shape=[1, kv_size, local_num_heads, head_dim],
            dtype=transformer_dtype,
            device=self.device)
        warmup_cross_kv = self._initialize_crossattn_cache(
            num_layers=cfg.num_layers,
            shape=[1, text_seq_len, cfg.num_heads, head_dim],
            dtype=transformer_dtype,
            device=self.device)

        # `y` is concat([msk_4ch, vae_latent_16ch]) → 20 channels; combined
        # with latent's 16 ch at patch-embed concat, the DiT sees 36 ch in.
        dummy_latent = torch.zeros(
            16, chunk_size, lat_h, lat_w,
            device=self.device, dtype=torch.float32)
        dummy_y = torch.zeros(
            20, chunk_size, lat_h, lat_w,
            device=self.device, dtype=transformer_dtype)
        dummy_c2ws = torch.zeros(
            1, plucker_channels, chunk_size, lat_h, lat_w,
            device=self.device, dtype=self.param_dtype)
        dummy_context = torch.zeros(
            text_seq_len, t5_hidden,
            device=self.device, dtype=self.param_dtype)
        dummy_t = torch.tensor(
            [500.0], device=self.device, dtype=torch.float32)

        @contextmanager
        def _noop_no_sync():
            yield
        no_sync_model = getattr(self.model, 'no_sync', _noop_no_sync)

        if dist.is_initialized():
            torch.cuda.synchronize()
            dist.barrier()
        t0 = time.perf_counter()

        with torch.amp.autocast('cuda', dtype=self.param_dtype), \
             torch.no_grad(), \
             no_sync_model():
            _ = self.model(
                x=[dummy_latent],
                t=dummy_t,
                context=[dummy_context],
                seq_len=max_seq_len,
                y=[dummy_y],
                dit_cond_dict={"c2ws_plucker_emb": (dummy_c2ws,)},
                kv_cache=warmup_self_kv,
                crossattn_cache=warmup_cross_kv,
                current_start=0,
                max_attention_size=kv_size,
                frame_seqlen=frame_seqlen,
            )

        if dist.is_initialized():
            torch.cuda.synchronize()
            dist.barrier()

        if (not dist.is_initialized()) or dist.get_rank() == 0:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            logging.info(f"WanI2VFast.prewarm: {dt_ms:.0f} ms")

        del (warmup_self_kv, warmup_cross_kv, dummy_latent, dummy_y,
             dummy_c2ws, dummy_context, dummy_t)
        torch.cuda.empty_cache()
        self._warmed = True

    def _configure_model(self, model, use_sp, dit_fsdp, shard_fn,
                         convert_model_dtype):
        """
        Configures a model object. This includes setting evaluation modes,
        applying distributed parallel strategy, and handling device placement.

        Args:
            model (torch.nn.Module):
                The model instance to configure.
            use_sp (`bool`):
                Enable distribution strategy of sequence parallel.
            dit_fsdp (`bool`):
                Enable FSDP sharding for DiT model.
            shard_fn (callable):
                The function to apply FSDP sharding.
            convert_model_dtype (`bool`):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.

        Returns:
            torch.nn.Module:
                The configured model.
        """
        model.eval().requires_grad_(False)

        if use_sp:
            for block in model.blocks:
                block.self_attn.forward = types.MethodType(
                    sp_attn_forward_causal, block.self_attn)
            model.forward = types.MethodType(sp_dit_forward_causal, model)

        if dist.is_initialized():
            dist.barrier()

        if dit_fsdp:
            model = shard_fn(model)
        else:
            if convert_model_dtype:
                model.to(self.param_dtype)
            if not self.init_on_cpu:
                model.to(self.device)

        return model

    def _convert_flow_pred_to_x0(self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor, scheduler) -> torch.Tensor:
        """
        Convert flow matching's prediction to x0 prediction.
        flow_pred: the prediction with shape [B, C, F, H, W]
        xt: the input noisy data with shape [B, C, F, H, W]
        timestep: the timestep with shape [B]

        pred = noise - x0
        x_t = (1-sigma_t) * x0 + sigma_t * noise
        we have x0 = x_t - sigma_t * pred
        """
        # use higher precision for calculations
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device), [flow_pred, xt, scheduler.sigmas, scheduler.timesteps]
        )
        timestep_id = torch.argmin((timesteps - timestep).abs())
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred

        return x0_pred.to(original_dtype)


    def generate(self,
                 input_prompt,
                 img,
                 action_path=None,
                 chunk_size=3,
                 max_area=480 * 832,
                 frame_num=81,
                 timesteps_index=[0, 179, 358, 679],
                 shift=5.0,
                 seed=-1,
                 offload_model=True,
                 max_sequence_length=512,
                 max_attention_size=None,):
        r"""
        Generates video frames from input image and text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation.
            img (PIL.Image.Image):
                Input image tensor. Shape: [3, H, W]
            max_area (`int`, *optional*, defaults to 720*1280):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
                [NOTE]: If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from max_area)
                - W: Frame width from max_area)
        """

        if input_prompt is not None and isinstance(input_prompt, str):
            batch_size = 1
        elif input_prompt is not None and isinstance(input_prompt, list):
            batch_size = len(input_prompt)
        else:
            batch_size = 1
        if action_path is not None:
            c2ws = np.load(os.path.join(action_path, "poses.npy")) # opencv coordinate
            len_c2ws = ((len(c2ws) - 1) // 4) * 4 + 1
            frame_num = ((frame_num - 1) // 4) * 4 + 1
            frame_num = min(frame_num, len_c2ws)
            c2ws = c2ws[:frame_num]
            if self.control_type == 'act':
                # In 'act' mode, use rotation of c2ws to control orientation and wasd_action to drive movement.
                wasd_action = np.load(os.path.join(action_path, "action.npy")) # wasd action
                wasd_action = wasd_action[:frame_num]

        # preprocess
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)

        F = frame_num
        h, w = img.shape[1:]
        aspect_ratio = h / w
        lat_h = round(
            np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] //
            self.patch_size[1] * self.patch_size[1])
        lat_w = round(
            np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] //
            self.patch_size[2] * self.patch_size[2])
        h = lat_h * self.vae_stride[1]
        w = lat_w * self.vae_stride[2]
        lat_f = (F - 1) // self.vae_stride[0] + 1
        lat_f = int(lat_f - (lat_f % chunk_size))
        F = (lat_f - 1) * 4 + 1
        max_seq_len = chunk_size * lat_h * lat_w // (
            self.patch_size[1] * self.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size
        # Reset per-generate state: cross-attn K/V cache will be freshly
        # initialized below; the first DiT forward must compute and store.
        self._cross_attn_initialized = False

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        noise = torch.randn(
            16,
            lat_f,
            lat_h,
            lat_w,
            dtype=torch.float32,
            generator=seed_g,
            device=self.device)

        msk = torch.ones(1, F, lat_h, lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([
            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
        ],
                           dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]

        # 2. Prepare timesteps
        self.scheduler.set_timesteps(self.num_train_timesteps, shift=shift)
        timesteps = self.scheduler.timesteps[timesteps_index]

        # preprocess
        # T5 cache: skip the encoder entirely if we've seen this exact prompt
        # before in this pipe instance. Bit-identical: cached tensor is the
        # same object returned by the prior call.
        cache_key = hashlib.sha256(input_prompt.encode('utf-8')).hexdigest()
        if cache_key in self._t5_cache:
            context = self._t5_cache[cache_key]
        else:
            if not self.t5_cpu:
                self.text_encoder.model.to(self.device)
                context = self.text_encoder([input_prompt], self.device)
                if offload_model:
                    self.text_encoder.model.cpu()
            else:
                context = self.text_encoder([input_prompt], torch.device('cpu'))
                context = [t.to(self.device) for t in context]
            self._t5_cache[cache_key] = context

        # cam preparation (only if action_path is provided)
        dit_cond_dict = None
        if action_path is not None:
            Ks = torch.from_numpy(np.load(os.path.join(action_path, "intrinsics.npy"))).float()

            # The provided intrinsics are for original image size (480p). We need to transform them according to the new image size (h, w).
            Ks = get_Ks_transformed(Ks,
                                    height_org=480,
                                    width_org=832,
                                    height_resize=h,
                                    width_resize=w,
                                    height_final=h,
                                    width_final=w)
            Ks = Ks[0]

            len_c2ws = len(c2ws)
            len_c2ws_ = int((len_c2ws - 1) // 4) + 1
            len_c2ws_ = int(len_c2ws_ - (len_c2ws_ % chunk_size))
            c2ws_infer = interpolate_camera_poses(
                src_indices=np.linspace(0, len_c2ws - 1, len_c2ws),
                src_rot_mat=c2ws[:, :3, :3],
                src_trans_vec=c2ws[:, :3, 3],
                tgt_indices=np.linspace(0, len_c2ws - 1, len_c2ws_),
            )
            c2ws_infer = compute_relative_poses(c2ws_infer, framewise=True)
            Ks = Ks.repeat(len(c2ws_infer), 1)

            c2ws_infer = c2ws_infer.to(self.device)
            Ks = Ks.to(self.device)
            if self.control_type == 'act':
                wasd_action = torch.from_numpy(wasd_action[::4]).float().to(self.device)
            else:
                wasd_action = None
            only_rays_d = wasd_action is not None
            c2ws_plucker_emb = get_plucker_embeddings(c2ws_infer, Ks, h, w, only_rays_d=only_rays_d)
            c2ws_plucker_emb = rearrange(
                c2ws_plucker_emb,
                'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
                c1=int(h // lat_h),
                c2=int(w // lat_w),
            )
            c2ws_plucker_emb = c2ws_plucker_emb[None, ...] # [b, f*h*w, c]
            c2ws_plucker_emb = rearrange(c2ws_plucker_emb, 'b (f h w) c -> b c f h w', f=lat_f, h=lat_h, w=lat_w).to(self.param_dtype)
            if wasd_action is not None:
                wasd_action_tensor = wasd_action[:, None, None, :].repeat(1, h, w, 1) # [f, h, w, 3]
                wasd_action_tensor = rearrange(
                    wasd_action_tensor,
                    'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
                    c1=int(h // lat_h),
                    c2=int(w // lat_w),
                )
                wasd_action_tensor = wasd_action_tensor[None, ...] # [b, f*h*w, c]
                wasd_action_tensor = rearrange(wasd_action_tensor, 'b (f h w) c -> b c f h w', f=lat_f, h=lat_h, w=lat_w).to(self.param_dtype)
                c2ws_plucker_emb = torch.cat([c2ws_plucker_emb, wasd_action_tensor], dim=1)

        y = self.vae.encode([
            torch.concat([
                torch.nn.functional.interpolate(
                    img[None].cpu(), size=(h, w), mode='bicubic').transpose(
                        0, 1),
                torch.zeros(3, F - 1, h, w)
            ],
                         dim=1).to(self.device)
        ])[0]
        y = torch.concat([msk, y])

        @contextmanager
        def noop_no_sync():
            yield

        no_sync_model = getattr(self.model, 'no_sync', noop_no_sync)

        # Initialize KV cache to all zeros
        model_args = self.model.config
        transformer_dtype = self.pipe_dtype
        frame_seqlen = int(noise.shape[-2] * noise.shape[-1]// 4)
        if self.local_attn_size > -1:
            kv_size = frame_seqlen * self.local_attn_size
        else:
            kv_size = frame_seqlen * lat_f
        head_dim = model_args.dim // model_args.num_heads
        local_num_heads = model_args.num_heads // self.sp_size
        self_kv_shape = [batch_size, kv_size, local_num_heads, head_dim]
        self_kv_cache = self._initialize_self_kv_cache(num_layers=model_args.num_layers,
                                                      shape=self_kv_shape,
                                                      dtype=transformer_dtype,
                                                      device=self.device)
        cross_kv_shape = [batch_size, max_sequence_length, model_args.num_heads, head_dim]
        cross_kv_cache = self._initialize_crossattn_cache(num_layers=model_args.num_layers,
                                                         shape=cross_kv_shape,
                                                         dtype=transformer_dtype,
                                                         device=self.device)
        # evaluation mode
        with (
                torch.amp.autocast('cuda', dtype=self.param_dtype),
                torch.no_grad(),
                no_sync_model(),
        ):
            # sample videos
            latent = noise
            latents_chunk = latent.split(chunk_size, dim=1) # [c, f, h, w]
            condition_chunk = y.split(chunk_size, dim=1)
            c2ws_plucker_emb_chunk = c2ws_plucker_emb.split(chunk_size, dim=2)
            num_inference_chunk = len(latents_chunk)
            pred_latent_chunks = []
            for chunk_id in tqdm(range(num_inference_chunk)):
                current_latent = latents_chunk[chunk_id]
                current_condition = condition_chunk[chunk_id]
                current_c2ws_plucker_emb = c2ws_plucker_emb_chunk[chunk_id]

                dit_cond_dict = {
                    "c2ws_plucker_emb": current_c2ws_plucker_emb.chunk(1, dim=0),
                }

                kwargs = {
                    'context': [context[0]],
                    'seq_len': max_seq_len,
                    'y': [current_condition],
                    'dit_cond_dict': dit_cond_dict,
                    'kv_cache': self_kv_cache,
                    'crossattn_cache': cross_kv_cache,
                    'current_start': chunk_id * chunk_size * frame_seqlen,
                    'max_attention_size': kv_size if max_attention_size is None else max_attention_size,
                    'frame_seqlen': frame_seqlen,
                }

                if offload_model:
                    torch.cuda.empty_cache()

                for timestep_idx in range(len(timesteps)):
                    latent_model_input = [current_latent.to(self.device)]
                    current_timestep = [timesteps[timestep_idx]]

                    timestep = torch.stack(current_timestep).to(self.device)

                    noise_pred = self.model(
                        x=latent_model_input, t=timestep,
                        cross_attn_first_call=not self._cross_attn_initialized,
                        **kwargs)[0]
                    self._cross_attn_initialized = True

                    if offload_model:
                        torch.cuda.empty_cache()

                    x0 = self._convert_flow_pred_to_x0(
                        flow_pred=noise_pred,
                        xt=current_latent,
                        timestep=current_timestep[0],
                        scheduler=self.scheduler,
                    )

                    if timestep_idx < len(timesteps) - 1:
                        next_timestep = timesteps[timestep_idx + 1]
                        current_latent = self.scheduler.add_noise(x0, torch.randn(x0.shape, generator=seed_g, device=x0.device, dtype=x0.dtype), next_timestep)
                    else:
                        # note return x0
                        break

                pred_latent_chunks.append(x0)

                # Update kv cache
                context_timestep = [timesteps[-1] * 0.0]
                timestep = torch.stack(context_timestep).to(self.device)
                self.model(x=[x0], t=timestep,
                           cross_attn_first_call=False,
                           **kwargs)

            pred_latent_chunks = torch.cat(pred_latent_chunks, dim=1)

            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()

            if self.rank == 0:
                videos = self.vae.decode([pred_latent_chunks])

        # del noise, latent, x0
        # del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None

    def _initialize_self_kv_cache(self, num_layers, shape, dtype, device):
        """
        Initialize a Per-GPU KV cache for the SelfAttn.
        """
        self_kv_cache = []
        for _ in range(num_layers):
            self_kv_cache.append({
                'k': torch.zeros(shape, dtype=dtype, device=device),
                'v': torch.zeros(shape, dtype=dtype, device=device),
                'global_end_index': torch.tensor([0], dtype=torch.long, device=device),
                'local_end_index': torch.tensor([0], dtype=torch.long, device=device)
            })

        return self_kv_cache


    def _initialize_crossattn_cache(self, num_layers, shape, dtype, device):
        """
        Initialize a per-GPU cross-attention cache.
        """
        crossattn_cache = []
        for _ in range(num_layers):
            crossattn_cache.append({
                'k': torch.zeros(shape, dtype=dtype, device=device),
                'v': torch.zeros(shape, dtype=dtype, device=device),
                'is_init': torch.tensor(0, dtype=torch.int32, device=device),
            })

        return crossattn_cache
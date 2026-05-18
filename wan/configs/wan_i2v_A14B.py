import torch
from easydict import EasyDict

from .shared_config import wan_shared_cfg

#------------------------ Wan I2V A14B ------------------------#

i2v_A14B = EasyDict(__name__='Config: Wan I2V A14B')
i2v_A14B.update(wan_shared_cfg)

i2v_A14B.t5_checkpoint = 'models_t5_umt5-xxl-enc-bf16.pth'
i2v_A14B.t5_tokenizer = 'google/umt5-xxl'

# vae
i2v_A14B.vae_checkpoint = 'Wan2.1_VAE.pth'
i2v_A14B.vae_stride = (4, 8, 8)

# transformer
i2v_A14B.patch_size = (1, 2, 2)
i2v_A14B.dim = 5120
i2v_A14B.ffn_dim = 13824
i2v_A14B.freq_dim = 256
i2v_A14B.num_heads = 40
i2v_A14B.num_layers = 40
i2v_A14B.window_size = (-1, -1)
i2v_A14B.qk_norm = True
i2v_A14B.cross_attn_norm = True
i2v_A14B.eps = 1e-6
i2v_A14B.low_noise_checkpoint = 'low_noise_model'
i2v_A14B.high_noise_checkpoint = 'high_noise_model'
i2v_A14B.fast_noise_checkpoint = 'lingbot_world_fast'

# inference
i2v_A14B.sample_shift = 10.0
i2v_A14B.sample_steps = 70
i2v_A14B.boundary = 0.947
i2v_A14B.sample_guide_scale = (5.0, 5.0)  # low noise, high noise
i2v_A14B.sample_neg_prompt = '画面突变，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走，镜头晃动，画面闪烁，模糊，噪点，水印，签名，文字，变形，扭曲，液化，不合逻辑的结构，卡顿，PPT幻灯片感，过暗，欠曝，低对比度，霓虹灯光感，过度锐化，3D渲染感，人物，行人，游客，身体，皮肤，肢体，面部特征，汽车，电线'

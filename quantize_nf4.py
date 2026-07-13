"""
Quantize a WanModel checkpoint to NF4 (4-bit NormalFloat) using bitsandbytes.
Saves a reload-compatible checkpoint via diffusers save_pretrained.

Usage:
    python quantize_nf4.py \
        --src base-act/high_noise_model \
        --dst base-act/high_noise_model_nf4
"""
import argparse
import sys
import torch
import torch.nn as nn

sys.path.insert(0, ".")

from diffusers import BitsAndBytesConfig
from wan.modules.model import WanModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="base-act/high_noise_model")
    p.add_argument("--dst", default="base-act/high_noise_model_nf4")
    return p.parse_args()


def main():
    args = parse_args()

    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading {args.src} ...")
    model = WanModel.from_pretrained(
        args.src,
        quantization_config=nf4_config,
        torch_dtype=torch.bfloat16,
    )

    print(f"Saving NF4 model to {args.dst} ...")
    model.save_pretrained(args.dst)
    print("Done.")


if __name__ == "__main__":
    main()

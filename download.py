import argparse
from huggingface_hub import snapshot_download

if __name__ == "__main__":
    MODELS = {
        #"base-cam": "robbyant/lingbot-world-base-cam",
        "base-cam-nf4": "cahlen/lingbot-world-base-cam-nf4",
        "base-act": "robbyant/lingbot-world-base-act",
        "fast": "robbyant/lingbot-world-fast",
    }

    parser = argparse.ArgumentParser(description="Download Lingbot World models from Hugging Face")
    parser.add_argument(
        "--model",
        type=str,
        nargs="+",
        choices=list(MODELS.keys()),
        default=["base-act", "base-cam-nf4"],
        help=f"Model(s) to download. Available options: {', '.join(MODELS.keys())} (default: base-act base-cam-nf4)"
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        default=None,
        help="Optional flat local directory. By default, model lands in the shared HF cache at ~/.cache/huggingface/hub/ — load with from_pretrained(repo_id) anywhere."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files exist in cache (passes force_download=True to snapshot_download)."
    )

    args = parser.parse_args()

    for model in args.model:
        repo_id = MODELS[model]

        print(f"Downloading model: {model}")
        print(f"Repository: {repo_id}")
        if args.local_dir:
            print(f"Local directory: {args.local_dir}")
        else:
            print(f"Cache directory: ~/.cache/huggingface/hub/")
        if args.force:
            print("Force re-download: ON")
        print()

        path = snapshot_download(repo_id=repo_id, local_dir=args.local_dir, force_download=args.force)
        print(f"Model '{model}' available at {path}")

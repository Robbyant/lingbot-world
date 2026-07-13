import argparse
from huggingface_hub import snapshot_download

if __name__ == "__main__":
    # Available models
    MODELS = {
        #"base-cam": "robbyant/lingbot-world-base-cam",
        "base-cam-nf4": "cahlen/lingbot-world-base-cam-nf4",
        "base-act": "robbyant/lingbot-world-base-act"
    }

    # Set up argument parser
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
        help="Local directory to save the model (default: ./model-name)"
    )

    args = parser.parse_args()

    for model in args.model:
        repo_id = MODELS[model]
        local_dir = args.local_dir if args.local_dir else f"./{model}"

        print(f"Downloading model: {model}")
        print(f"Repository: {repo_id}")
        print(f"Local directory: {local_dir}")
        print()

        snapshot_download(repo_id=repo_id, repo_type="model", local_dir=local_dir)
        print(f"Model '{model}' downloaded to {local_dir}")

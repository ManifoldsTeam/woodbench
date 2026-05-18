#!/usr/bin/env python3
"""Download woodbench dataset from HuggingFace Hub."""

import os
import sys
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download, login
except ImportError:
    print("Installing huggingface_hub...")
    os.system("pip install huggingface_hub -q")
    from huggingface_hub import hf_hub_download, login

HF_TASK_A_REPO = "ny0g507h3p/wood-task-A"
PROJECT_ROOT = Path("/workspace/woodbench")
DATA_DIR = PROJECT_ROOT / "data" / "task-A"


def main():
    # Get HF token
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Enter your HuggingFace token: ", end="")
        sys.stdout.flush()
        token = input().strip()

    if not token:
        print("Error: HF token is required.")
        sys.exit(1)

    # Login
    print("Logging in to HuggingFace...")
    login(token=token)

    # Create data directory
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Download all files from the repo
    print(f"Downloading dataset to {DATA_DIR}...")

    from huggingface_hub import snapshot_download

    # Create data directory
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Download all files from the repo
    print(f"Downloading dataset to {DATA_DIR}...")

    snapshot_download(
        repo_id=HF_TASK_A_REPO,
        repo_type="dataset",
        local_dir=DATA_DIR,
        local_dir_use_symlinks=False,
    )

    print(f"\nDataset ready at {DATA_DIR}")


if __name__ == "__main__":
    main()

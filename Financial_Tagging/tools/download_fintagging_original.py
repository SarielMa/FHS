#!/usr/bin/env python3
"""Download TheFinAI/FinTagging_Original from Hugging Face."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


REPO_ID = "TheFinAI/FinTagging_Original"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Download the full Hugging Face dataset repo: {REPO_ID}"
    )
    parser.add_argument(
        "--local-dir",
        default="FinTagging_Original",
        help="Destination directory. Default: %(default)s",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Dataset revision, branch, tag, or commit. Default: %(default)s",
    )
    parser.add_argument(
        "--cache-dir",
        default=".hf_cache",
        help=(
            "Hugging Face cache directory. Default keeps cache metadata in this "
            "workspace instead of using a shared home/scratch cache."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    local_dir = Path(args.local_dir).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve()

    local_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", str(cache_dir))

    downloaded_path = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        revision=args.revision,
        local_dir=str(local_dir),
        cache_dir=str(cache_dir),
    )

    print(f"Downloaded {REPO_ID} to: {downloaded_path}")
    print(f"Local directory: {local_dir}")


if __name__ == "__main__":
    main()

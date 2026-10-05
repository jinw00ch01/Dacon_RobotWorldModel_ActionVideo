"""Download public model weights into the Hugging Face cache.

python -m wmgen.fetch Wan-AI/Wan2.1-VACE-1.3B-diffusers [--include 'transformer/*' ...]
"""
from __future__ import annotations

import argparse

from huggingface_hub import snapshot_download


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("--include", nargs="*", default=None)
    ap.add_argument("--revision", default=None)
    args = ap.parse_args()
    path = snapshot_download(args.repo, allow_patterns=args.include, revision=args.revision, max_workers=4)
    print(path)


if __name__ == "__main__":
    main()

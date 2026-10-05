"""S0: repeat the eval input image for all 16 frames.

python -m wmgen.s0_first_frame --eval-root open/data/eval --out C:/Dacon/WM_Shared/s0_first_frame/videos
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from wmgen.video_io import NUM_FRAMES, read_mp4, write_mp4


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", type=Path, default=Path("open/data/eval"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    t0 = time.time()
    images = sorted((args.eval_root / "images").glob("*.png"))
    if not images:
        raise SystemExit(f"no images under {args.eval_root / 'images'}")
    worst = 0.0
    for img_path in images:
        image = np.asarray(Image.open(img_path).convert("RGB"))
        out = args.out / f"{img_path.stem}.mp4"
        write_mp4(np.repeat(image[None], NUM_FRAMES, axis=0), out)
        decoded = read_mp4(out)
        if decoded.shape != (NUM_FRAMES, *image.shape):
            raise RuntimeError(f"{out}: decoded shape {decoded.shape}")
        worst = max(worst, float(np.abs(decoded[0].astype(np.int16) - image).mean()))
    manifest = {
        "method": "first_frame_repeat",
        "count": len(images),
        "frames": NUM_FRAMES,
        "max_mean_abs_err_frame0": round(worst, 4),
        "seconds": round(time.time() - t0, 1),
    }
    (args.out.parent / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()

"""Apply background anchoring to a folder of generated mp4s (CPU).

python -m wmgen.apply_anchor --pred <dir> --eval-root C:/Dacon/WM_Shared/holdout_v1 --out <dir> --threshold 14
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from wmgen.anchor import AnchorParams, anchor
from wmgen.video_io import read_mp4, write_mp4


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--eval-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    d = AnchorParams()
    ap.add_argument("--threshold", type=float, default=d.threshold)
    ap.add_argument("--blur-sigma", type=float, default=d.blur_sigma)
    ap.add_argument("--dilate", type=int, default=d.dilate)
    ap.add_argument("--feather-sigma", type=float, default=d.feather_sigma)
    ap.add_argument("--no-color-correct", action="store_true")
    ap.add_argument("--accumulate", action="store_true")
    args = ap.parse_args()

    p = AnchorParams(threshold=args.threshold, blur_sigma=args.blur_sigma, dilate=args.dilate,
                     feather_sigma=args.feather_sigma, color_correct=not args.no_color_correct,
                     accumulate=args.accumulate)
    args.out.mkdir(parents=True, exist_ok=True)
    kept = []
    for mp4 in sorted(args.pred.glob("*.mp4")):
        image = np.asarray(Image.open(args.eval_root / "images" / f"{mp4.stem}.png").convert("RGB"))
        frames = read_mp4(mp4)
        out = anchor(frames, image, p)
        kept.append(float((out[1:] != image[None]).any(axis=3).mean()))
        write_mp4(out, args.out / mp4.name)
    stats = {"params": asdict(p), "n": len(kept), "mean_changed_pixel_fraction": float(np.mean(kept))}
    (args.out.parent / "anchor.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats))


if __name__ == "__main__":
    main()

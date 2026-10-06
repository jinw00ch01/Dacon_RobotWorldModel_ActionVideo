"""Materialise the holdout windows in the eval layout so any generator runs on them unchanged.

out/images/<id>.png      frame 0 (640x480 like eval)
out/actions/<id>.npy     (16, 6) float32 raw actions
out/gt_videos/<id>.mp4   ground-truth 16 frames
out/windows.csv          id -> source window

python -m wmscore.build_holdout --out C:/Dacon/WM_Shared/holdout_v1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from wmgen.video_io import write_mp4
from wmscore.data import NUM_FRAMES, REPO, decode_frames, read_actions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="holdout_v1")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--windows", type=Path, default=None, help="any windows csv (default: <split>_val_windows.csv)")
    ap.add_argument("--prefix", default="hold")
    args = ap.parse_args()

    windows = pd.read_csv(args.windows or REPO / "configs" / "splits" / f"{args.split}_val_windows.csv")
    for sub in ("images", "actions", "gt_videos"):
        (args.out / sub).mkdir(parents=True, exist_ok=True)
    rows = []
    for i, w in windows.iterrows():
        sid = f"{args.prefix}_{i:06d}"
        idx = list(range(int(w.start_frame), int(w.start_frame) + NUM_FRAMES))
        frames = decode_frames(w.video, idx)
        if frames.shape[1:3] != (480, 640):
            frames = np.stack([cv2.resize(f, (640, 480), interpolation=cv2.INTER_AREA) for f in frames])
        actions = read_actions(w.parquet)[idx]
        Image.fromarray(frames[0]).save(args.out / "images" / f"{sid}.png")
        np.save(args.out / "actions" / f"{sid}.npy", actions)
        write_mp4(frames, args.out / "gt_videos" / f"{sid}.mp4")
        rows.append({"sample_id": sid, **w.to_dict()})
    pd.DataFrame(rows).to_csv(args.out / "windows.csv", index=False)
    print(f"wrote {len(rows)} holdout samples to {args.out}")


if __name__ == "__main__":
    main()

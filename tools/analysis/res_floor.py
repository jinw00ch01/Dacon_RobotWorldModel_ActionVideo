"""How much DINO / R3D distance generation resolution alone costs (CPU, holdout ground truth only).

Each holdout GT clip is resized to H x W (INTER_AREA) and back to 640x480 (bicubic), as gen_cosmos_ac does with its output,
frame 0 kept as the original image, and scored against the untouched clip with wmscore's feature models. The result is a
floor: a perfect generator at that resolution still pays it.

python tools/analysis/res_floor.py --holdout C:/Dacon/WM_Shared/holdout_v1_sub64 --sizes 240x320 288x384 320x432 480x640
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from wmgen.video_io import read_mp4  # noqa: E402
from wmscore.features import FeatureModels, cosine_distance, framewise_cosine_distance, to_eval_frames  # noqa: E402


def round_trip(video: np.ndarray, h: int, w: int) -> np.ndarray:
    out = video.copy()
    for t in range(1, len(video)):
        small = cv2.resize(video[t], (w, h), interpolation=cv2.INTER_AREA)
        out[t] = cv2.resize(small, (video.shape[2], video.shape[1]), interpolation=cv2.INTER_CUBIC)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", type=Path, required=True)
    ap.add_argument("--sizes", nargs="+", default=["240x320", "288x384", "320x432", "480x640"])
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    models = FeatureModels(torch.device("cpu"))
    ids = sorted(p.stem for p in (args.holdout / "gt_videos").glob("*.mp4"))
    res = {s: {"dino": [], "r3d": []} for s in args.sizes}
    for i in ids:
        gt = read_mp4(args.holdout / "gt_videos" / f"{i}.mp4")
        f_gt = to_eval_frames(gt)
        d_gt, r_gt = models.dino_features(f_gt), models.r3d_features(f_gt)
        for s in args.sizes:
            h, w = (int(x) for x in s.split("x"))
            f = to_eval_frames(round_trip(gt, h, w))
            res[s]["dino"].append(framewise_cosine_distance(models.dino_features(f), d_gt))
            res[s]["r3d"].append(cosine_distance(models.r3d_features(f), r_gt))
    out = {s: {k: float(np.mean(v)) for k, v in d.items()} for s, d in res.items()}
    print(json.dumps({"n": len(ids), "floor": out}, indent=1))


if __name__ == "__main__":
    main()

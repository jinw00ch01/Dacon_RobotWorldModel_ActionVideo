"""Score generated videos on the holdout with our own models.

score = 0.3 * DINO + 0.3 * R3D + 0.4 * Action, each averaged over samples.
DINO is reported two ways (per-frame mean cosine distance, and cosine distance of the flattened
16x384 feature) because the leaderboard's exact aggregation is not published.

python -m wmscore.score --pred C:/Dacon/WM_Shared/holdout_v1/pred/s0 --holdout C:/Dacon/WM_Shared/holdout_v1 \
    --idm C:/Dacon/WM_Shared/idm/idm_v1.pt --out runs/score_s0.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from wmgen.video_io import read_mp4
from wmscore.cache_frames import IDM_H, IDM_W
from wmscore.data import NUM_FRAMES, letterbox, load_action_stats
from wmscore.features import FeatureModels, cosine_distance, framewise_cosine_distance, to_eval_frames
from wmscore.idm import load_idm

WEIGHTS = {"dino": 0.3, "r3d": 0.3, "action": 0.4}


def video_features(models: FeatureModels, video: np.ndarray) -> dict:
    frames = to_eval_frames(video)
    return {"dino": models.dino_features(frames), "r3d": models.r3d_features(frames)}


def gt_features(models: FeatureModels, holdout: Path, ids: list[str]) -> dict:
    cache = holdout / "gt_features.pt"
    if cache.exists():
        feats = torch.load(cache, weights_only=True)
        if all(i in feats for i in ids):
            return feats
    feats = {i: video_features(models, read_mp4(holdout / "gt_videos" / f"{i}.mp4")) for i in ids}
    torch.save(feats, cache)
    return feats


@torch.no_grad()
def idm_actions(idm, video: np.ndarray, device) -> np.ndarray:
    small = np.stack([letterbox(f, IDM_H, IDM_W) for f in video])
    x = torch.from_numpy(small).permute(0, 3, 1, 2).float().div(255.0)[None].to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return idm(x).float().cpu().numpy()[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--holdout", type=Path, required=True)
    ap.add_argument("--idm", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    device = torch.device("cuda")
    ids = sorted(p.stem for p in (args.holdout / "images").glob("*.png"))
    missing = [i for i in ids if not (args.pred / f"{i}.mp4").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} predictions missing, e.g. {missing[:3]}")
    models = FeatureModels(device)
    gt = gt_features(models, args.holdout, ids)
    idm = load_idm(args.idm, device)
    mean, std = load_action_stats()

    rows = []
    for i in ids:
        video = read_mp4(args.pred / f"{i}.mp4")
        if len(video) != NUM_FRAMES:
            raise SystemExit(f"{i}: {len(video)} frames")
        f = video_features(models, video)
        target = (np.load(args.holdout / "actions" / f"{i}.npy") - mean) / std
        rows.append({
            "sample_id": i,
            "dino": framewise_cosine_distance(f["dino"], gt[i]["dino"]),
            "dino_flat": cosine_distance(f["dino"], gt[i]["dino"]),
            "r3d": cosine_distance(f["r3d"], gt[i]["r3d"]),
            "action": float(np.abs(idm_actions(idm, video, device) - target).mean()),
        })
    df = pd.DataFrame(rows)
    windows = pd.read_csv(args.holdout / "windows.csv")[["sample_id", "user", "dataset"]]
    df = df.merge(windows, on="sample_id")
    means = df[["dino", "dino_flat", "r3d", "action"]].mean()
    summary = {
        "pred": str(args.pred), "idm": str(args.idm), "n": len(df),
        "score": float(sum(WEIGHTS[k] * means[k] for k in WEIGHTS)),
        "score_dino_flat": float(0.3 * means["dino_flat"] + 0.3 * means["r3d"] + 0.4 * means["action"]),
        **{k: float(v) for k, v in means.items()},
        "by_user": df.groupby("user")[["dino", "r3d", "action"]].mean().round(4).to_dict(orient="index"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=1))
    df.to_csv(args.out.with_suffix(".csv"), index=False)
    print(json.dumps({k: summary[k] for k in ("score", "score_dino_flat", "dino", "dino_flat", "r3d", "action")}))


if __name__ == "__main__":
    main()

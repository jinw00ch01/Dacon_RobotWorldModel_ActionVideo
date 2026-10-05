"""Decode training episodes once into small letterboxed uint8 arrays for inverse-dynamics training.

C:/Dacon/WM_Shared/idm_cache/<split>/<user>__<dataset>__<ep>.frames.npy  (T,H,W,3) uint8, memory-mapped in training
C:/Dacon/WM_Shared/idm_cache/<split>/<user>__<dataset>__<ep>.actions.npy (T,6) float32

python -m wmscore.cache_frames --out C:/Dacon/WM_Shared/idm_cache --train-eps 30 --val-eps 10
"""
from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from wmscore.data import NUM_FRAMES, REPO, decode_frames, letterbox, load_split, read_actions

IDM_H, IDM_W = 128, 208
INDEX = Path(r"C:\Dacon\WM_Shared\data_index\episodes.parquet")


def _work(job):
    row, out_path = job
    actions_path = Path(str(out_path) + ".actions.npy")
    if actions_path.exists():
        return str(out_path), 0
    frames = decode_frames(row["video"])
    actions = read_actions(row["parquet"])
    n = min(len(frames), len(actions))
    small = np.stack([letterbox(f, IDM_H, IDM_W) for f in frames[:n]])
    save_episode(out_path, small, actions[:n])
    return str(out_path), n


def save_episode(base: Path, frames: np.ndarray, actions: np.ndarray) -> None:
    """Write frames then actions; the actions file marks the episode complete."""
    for suffix, arr in ((".frames.npy", frames), (".actions.npy", actions)):
        tmp = Path(str(base) + suffix + ".tmp")
        with open(tmp, "wb") as f:
            np.save(f, arr)
        tmp.replace(Path(str(base) + suffix))


def load_episode(base: Path) -> tuple[np.ndarray, np.ndarray]:
    return (np.load(str(base) + ".frames.npy", mmap_mode="r"), np.load(str(base) + ".actions.npy"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", default="holdout_v1")
    ap.add_argument("--train-eps", type=int, default=30, help="episodes sampled per train dataset")
    ap.add_argument("--val-eps", type=int, default=10, help="episodes sampled per val dataset")
    ap.add_argument("--valtrain-eps", type=int, default=0,
                    help="extra episodes per val dataset for a scorer-only IDM; excludes the val part and the "
                         "holdout window episodes (the generator never trains on these)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    split = load_split(args.split)
    eps = pd.read_parquet(INDEX)
    eps = eps[eps.parquet_ok & eps.video_ok & ~eps.nan_any & (eps.rows >= NUM_FRAMES)]
    eps["key"] = eps.user + "/" + eps.dataset
    jobs, manifest = [], {}
    for part, keys, k in (("train", split["train_datasets"], args.train_eps), ("val", split["val_datasets"], args.val_eps)):
        chosen = []
        for key in keys:
            d = eps[eps.key == key]
            chosen.append(d.sample(n=min(k, len(d)), random_state=args.seed))
        chosen = pd.concat(chosen)
        (args.out / part).mkdir(parents=True, exist_ok=True)
        names = []
        for _, r in chosen.iterrows():
            name = f"{r.user}__{r.dataset}__{int(r.episode_index):06d}"
            jobs.append((r[["video", "parquet"]].to_dict(), args.out / part / name))
            names.append(name)
        manifest[part] = names

    if args.valtrain_eps:
        windows = pd.read_csv(REPO / "configs" / "splits" / f"{args.split}_val_windows.csv")
        held = {f"{w.user}__{w.dataset}__{int(w.episode_index):06d}" for w in windows.itertuples()}
        held |= set(manifest["val"])
        (args.out / "valtrain").mkdir(parents=True, exist_ok=True)
        names = []
        for key in split["val_datasets"]:
            d = eps[eps.key == key].copy()
            d["name"] = d.user + "__" + d.dataset + "__" + d.episode_index.astype(int).map("{:06d}".format)
            d = d[~d.name.isin(held)]
            for _, r in d.sample(n=min(args.valtrain_eps, len(d)), random_state=args.seed).iterrows():
                jobs.append((r[["video", "parquet"]].to_dict(), args.out / "valtrain" / r["name"]))
                names.append(r["name"])
        manifest["valtrain"] = names

    t0, frames = time.time(), 0
    with Pool(args.workers) as pool:
        for i, (_, n) in enumerate(pool.imap_unordered(_work, jobs, chunksize=4)):
            frames += n
            if i % 200 == 0:
                print(f"{i}/{len(jobs)} episodes, {frames} new frames, {time.time() - t0:.0f}s", flush=True)
    meta = {"split": args.split, "size": [IDM_H, IDM_W], "seed": args.seed, "train_eps": args.train_eps,
            "val_eps": args.val_eps, "valtrain_eps": args.valtrain_eps, **manifest}
    (args.out / "manifest.json").write_text(json.dumps(meta, indent=1))
    print(f"done: {len(jobs)} episodes, {frames} new frames, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

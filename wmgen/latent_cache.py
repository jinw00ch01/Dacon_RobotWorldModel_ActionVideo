"""Encode 17-frame training clips to Wan2.1-VAE latents once, so DiT training never runs the VAE.

Per train-split dataset: C:/Dacon/WM_Shared/latents_240x320/<user>__<dataset>.pt with
  latents (N,16,5,30,40) bf16, actions (N,16,6) f32 raw, episode (N,), start (N,)
Clip k uses frames start..start+16 and actions start..start+15 (action[t] drives frame t -> t+1).

python -m wmgen.latent_cache --out C:/Dacon/WM_Shared/latents_240x320 --per-dataset 500
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from wmgen import cosmos_ac as ca
from wmscore.data import decode_frames, load_split, read_actions

INDEX = Path(r"C:\Dacon\WM_Shared\data_index\episodes.parquet")
CLIP = 17


def plan_windows(eps: pd.DataFrame, per_dataset: int, per_episode: int, stride: int, rng: random.Random):
    """Pick (episode row, start) pairs spread over episodes."""
    candidates = []
    for _, r in eps.iterrows():
        starts = list(range(0, int(r.rows) - CLIP + 1, stride))
        rng.shuffle(starts)
        candidates.append((r, sorted(starts[:per_episode])))
    rng.shuffle(candidates)
    chosen, total = [], 0
    while total < per_dataset and any(s for _, s in candidates):
        for r, starts in candidates:  # round-robin so every episode contributes
            if starts and total < per_dataset:
                chosen.append((r, starts.pop(0)))
                total += 1
    return chosen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", default="holdout_v1")
    ap.add_argument("--part", default="train", choices=["train", "val"])
    ap.add_argument("--per-dataset", type=int, default=500)
    ap.add_argument("--per-episode", type=int, default=6)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda")
    vae, mean, inv_std = ca.load_vae(device)
    split = load_split(args.split)
    eps = pd.read_parquet(INDEX)
    eps = eps[eps.parquet_ok & eps.video_ok & ~eps.nan_any & (eps.rows >= CLIP)]
    eps["key"] = eps.user + "/" + eps.dataset
    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    t0, clips = time.time(), 0
    for key in split[f"{args.part}_datasets"]:
        path = args.out / f"{key.replace('/', '__')}.pt"
        if path.exists():
            continue
        windows = plan_windows(eps[eps.key == key], args.per_dataset, args.per_episode, args.stride, rng)
        by_ep: dict[int, list] = {}
        for r, s in windows:
            by_ep.setdefault(int(r.episode_index), [r, []])[1].append(s)
        lat_out, act_out, ep_out, st_out = [], [], [], []
        for ep, (r, starts) in by_ep.items():
            last = max(starts) + CLIP
            frames = decode_frames(r.video, list(range(last)))
            small = np.stack([cv2.resize(f, (args.width, args.height), interpolation=cv2.INTER_AREA) for f in frames])
            actions = read_actions(r.parquet)
            for i in range(0, len(starts), args.batch):
                group = starts[i : i + args.batch]
                x = torch.from_numpy(np.stack([small[s : s + CLIP] for s in group])).to(device)
                x = x.permute(0, 4, 1, 2, 3).float() / 127.5 - 1  # B,3,T,H,W
                lat_out.append(ca.encode_frames(vae, mean, inv_std, x).to(torch.bfloat16).cpu())
                act_out.extend(actions[s : s + CLIP - 1] for s in group)
                ep_out.extend([ep] * len(group))
                st_out.extend(group)
        torch.save({"latents": torch.cat(lat_out), "actions": torch.from_numpy(np.stack(act_out)),
                    "episode": torch.tensor(ep_out), "start": torch.tensor(st_out)}, path)
        clips += len(st_out)
        print(f"{key}: {len(st_out)} clips, total {clips}, {time.time() - t0:.0f}s", flush=True)
    meta = {k: getattr(args, k) for k in ("split", "part", "per_dataset", "per_episode", "stride", "height", "width", "seed")}
    (args.out / f"manifest_{args.part}.json").write_text(json.dumps(meta, indent=1))
    print(f"done {clips} clips in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

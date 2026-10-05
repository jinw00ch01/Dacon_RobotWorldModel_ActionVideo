"""Encode 17-frame training clips to Wan2.1-VAE latents once, so DiT training never runs the VAE.

Per split dataset: <out>/<user>__<dataset>.pt with
  latents (N,16,5,H/8,W/8) bf16, actions (N,16,6) f32 raw, episode (N,), start (N,)
Clip k uses frames start..start+16 and actions start..start+15 (action[t] drives frame t -> t+1).
Episodes are decoded by DataLoader workers while the GPU encodes. Finished datasets are skipped, so
the job can be stopped and restarted.

python -m wmgen.latent_cache --out C:/Dacon/WM_Shared/latents_240x320 --per-dataset 500
(on Colab: --train-root /content/open/data/train --index-from-meta)
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
from torch.utils.data import DataLoader, Dataset

from wmgen import cosmos_ac as ca
from wmscore import data as wdata

INDEX = Path(r"C:\Dacon\WM_Shared\data_index\episodes.parquet")
CLIP = 17


def episodes_from_meta(train_root: Path) -> pd.DataFrame:
    """Episode table straight from LeRobot meta files (for machines without the data index)."""
    rows = []
    for info_path in sorted(train_root.glob("*/*/meta/info.json")):
        ds = info_path.parent.parent
        info = json.loads(info_path.read_text(encoding="utf-8"))
        video_key = next(k for k, v in info["features"].items() if v.get("dtype") == "video")
        chunk = int(info.get("chunks_size", 1000))
        for line in (ds / "meta" / "episodes.jsonl").read_text(encoding="utf-8").splitlines():
            ep = json.loads(line)
            i = int(ep["episode_index"])
            rel = ds.relative_to(train_root).as_posix()
            rows.append({"user": ds.parent.name, "dataset": ds.name, "episode_index": i, "rows": int(ep["length"]),
                         "parquet": f"{rel}/" + info["data_path"].format(episode_chunk=i // chunk, episode_index=i),
                         "video": f"{rel}/" + info["video_path"].format(episode_chunk=i // chunk, episode_index=i,
                                                                         video_key=video_key)})
    return pd.DataFrame(rows)


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


class EpisodeClips(Dataset):
    """One item per episode: its resized clips (n,17,H,W,3) uint8 and actions (n,16,6)."""

    def __init__(self, jobs, height, width, train_root: Path):
        self.jobs, self.height, self.width, self.root = jobs, height, width, train_root

    def __len__(self):
        return len(self.jobs)

    def __getitem__(self, i):
        r, starts = self.jobs[i]
        frames = wdata.decode_frames(self.root / r["video"], list(range(max(starts) + CLIP)))
        small = np.stack([cv2.resize(f, (self.width, self.height), interpolation=cv2.INTER_AREA) for f in frames])
        actions = wdata.read_actions(self.root / r["parquet"])  # absolute: workers may not see TRAIN_ROOT overrides
        clips = np.stack([small[s : s + CLIP] for s in starts])
        acts = np.stack([actions[s : s + CLIP - 1] for s in starts])
        return torch.from_numpy(clips), torch.from_numpy(acts), int(r["episode_index"]), torch.tensor(starts)


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
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--train-root", type=Path, default=None)
    ap.add_argument("--index-from-meta", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.train_root is not None:
        wdata.TRAIN_ROOT = args.train_root
    device = torch.device("cuda")
    vae, mean, inv_std = ca.load_vae(device)
    split = wdata.load_split(args.split)
    if args.index_from_meta:
        eps = episodes_from_meta(wdata.TRAIN_ROOT)
    else:
        eps = pd.read_parquet(INDEX)
        eps = eps[eps.parquet_ok & eps.video_ok & ~eps.nan_any]
    eps = eps[eps.rows >= CLIP].copy()
    eps["key"] = eps.user + "/" + eps.dataset
    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    t0, clips = time.time(), 0
    for key in split[f"{args.part}_datasets"]:
        windows = plan_windows(eps[eps.key == key], args.per_dataset, args.per_episode, args.stride, rng)
        path = args.out / f"{key.replace('/', '__')}.pt"
        if path.exists():
            continue
        by_ep: dict[int, list] = {}
        for r, s in windows:
            by_ep.setdefault(int(r.episode_index), [r[["video", "parquet", "episode_index"]].to_dict(), []])[1].append(s)
        loader = DataLoader(EpisodeClips(list(by_ep.values()), args.height, args.width, wdata.TRAIN_ROOT), batch_size=None,
                            num_workers=args.workers, prefetch_factor=4 if args.workers else None)
        lat_out, act_out, ep_out, st_out = [], [], [], []
        t_enc = 0.0
        for video, acts, ep, starts in loader:
            for i in range(0, len(video), args.batch):
                x = video[i : i + args.batch].to(device, non_blocking=True)
                x = x.permute(0, 4, 1, 2, 3).float() / 127.5 - 1  # B,3,T,H,W
                te = time.time()
                lat_out.append(ca.encode_frames(vae, mean, inv_std, x).to(torch.bfloat16).cpu())
                t_enc += time.time() - te
            act_out.append(acts)
            ep_out.extend([ep] * len(starts))
            st_out.append(starts)
        tmp = path.with_suffix(".tmp")
        torch.save({"latents": torch.cat(lat_out), "actions": torch.cat(act_out).float(),
                    "episode": torch.tensor(ep_out), "start": torch.cat(st_out)}, tmp)
        tmp.replace(path)
        clips += len(ep_out)
        print(f"{key}: {len(ep_out)} clips, total {clips}, {time.time() - t0:.0f}s (encode {t_enc:.0f}s)", flush=True)
    meta = {k: getattr(args, k) for k in ("split", "part", "per_dataset", "per_episode", "stride", "height", "width", "seed")}
    (args.out / f"manifest_{args.part}.json").write_text(json.dumps(meta, indent=1))
    print(f"done {clips} clips in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

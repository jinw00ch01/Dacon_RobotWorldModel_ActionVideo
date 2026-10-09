"""Copy a holdout with a simulated joint-calibration offset (CPU).

Adds sigma * train_std[j] to the listed joints of every action file, so the generator input and the scoring
targets both carry the offset, as the submission kit would score a robot whose joint zero differs.
Only holdout (training-data) folders are accepted; the competition eval folder is refused.

python -m wmgen.make_sim_holdout --src C:/Dacon/WM_Shared/holdout_v1_sub64 --out C:/Dacon/WM_Shared/holdout_v1_sub64_simoff25 \
    --joints 1 2 --sigma -2.5
python -m wmgen.make_sim_holdout --src C:/Dacon/WM_Shared/holdout_v1 --out C:/Dacon/WM_Shared/holdout_v1_route5 --ids route.json
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from wmscore.data import load_action_stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--joints", type=int, nargs="*", default=[])
    ap.add_argument("--sigma", type=float, default=0.0)
    ap.add_argument("--ids", type=Path, help="json list of sample ids (or a route.json with a 'routed' dict)")
    args = ap.parse_args()
    for p in (args.src, args.out):
        if "open" in {part.lower() for part in p.resolve().parts} or "eval" in p.resolve().name.lower():
            raise SystemExit(f"refusing a competition data path: {p}")

    ids = sorted(p.stem for p in (args.src / "images").glob("*.png"))
    if args.ids:
        sel = json.loads(args.ids.read_text())
        sel = sel["routed"] if isinstance(sel, dict) else sel
        ids = [i for i in ids if i in set(sel)]
    _, std = load_action_stats()
    shift = np.zeros(len(std), np.float32)
    shift[args.joints] = args.sigma * std[args.joints]

    for sub in ("images", "gt_videos", "actions"):
        (args.out / sub).mkdir(parents=True, exist_ok=True)
    for i in ids:
        shutil.copy2(args.src / "images" / f"{i}.png", args.out / "images" / f"{i}.png")
        shutil.copy2(args.src / "gt_videos" / f"{i}.mp4", args.out / "gt_videos" / f"{i}.mp4")
        a = np.load(args.src / "actions" / f"{i}.npy")
        np.save(args.out / "actions" / f"{i}.npy", (a + shift).astype(a.dtype))
    windows = pd.read_csv(args.src / "windows.csv")
    windows[windows.sample_id.isin(ids)].to_csv(args.out / "windows.csv", index=False)
    if (args.src / "gt_features.pt").exists():
        shutil.copy2(args.src / "gt_features.pt", args.out / "gt_features.pt")
    meta = {"src": str(args.src), "n": len(ids), "joints": args.joints, "sigma": args.sigma,
            "shift_raw": shift.tolist(), "ids": ids}
    (args.out / "offset.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: meta[k] for k in ("n", "joints", "sigma", "shift_raw")}))


if __name__ == "__main__":
    main()

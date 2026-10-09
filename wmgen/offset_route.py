"""Route joints whose readings fall outside the training robots' calibration envelope.

Eval scene1 robots read shoulder_lift / elbow_flex about -2.6 / -2.2 train-sigma, beyond every training
dataset's mean (docs/DATA_ANALYSIS.md section 3), which looks like a different joint zero. For such joints the
generator's absolute-pose channel is replaced by the relative one (wmgen.actions), so the clip is drawn from
its motion only. Other joints and in-envelope clips are untouched.

Envelope: per joint, min/max over the 116 training datasets of the dataset-mean z (configs/splits/holdout_v1.json,
C:/Dacon/WM_Shared/data_index/datasets.csv). A joint routes when its 16-step mean z lies more than `delta`
outside. delta is the smallest value in DELTAS that routes at most 5% of the clean holdout_v1 windows.
Only training data and the holdout (training-data uploaders) set these numbers; no eval data does.

python -m wmgen.offset_route build
python -m wmgen.offset_route list --root <eval or holdout root> --out route.json
python -m wmgen.offset_route copy-unrouted --root <eval root> --src <raw videos> --dst <new raw videos>
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from wmscore.data import load_action_stats

REPO = Path(__file__).resolve().parents[1]
ENVELOPE_PATH = REPO / "configs" / "offset_envelope.json"
DATASETS_CSV = Path(r"C:\Dacon\WM_Shared\data_index\datasets.csv")
HOLDOUT = Path(r"C:\Dacon\WM_Shared\holdout_v1")
DELTAS = (0.0, 0.25, 0.5, 0.75, 1.0)
MAX_HOLDOUT_FRACTION = 0.05


def load_envelope(path: Path = ENVELOPE_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def mean_z(actions: np.ndarray) -> np.ndarray:
    """(16, 6) raw readings -> (6,) mean z-score over the clip."""
    mean, std = load_action_stats()
    return ((np.asarray(actions, np.float64) - mean) / std).mean(axis=0)


def routed_joints(actions: np.ndarray, env: dict) -> np.ndarray:
    """(6,) bool: joints whose clip-mean z lies more than delta outside the training envelope."""
    z = mean_z(actions)
    lo, hi, d = np.asarray(env["lo"]), np.asarray(env["hi"]), float(env["delta"])
    return (z < lo - d) | (z > hi + d)


def build() -> dict:
    import pandas as pd

    split = json.loads((REPO / "configs" / "splits" / "holdout_v1.json").read_text(encoding="utf-8"))
    ds = pd.read_csv(DATASETS_CSV)
    ds = ds[(ds.user + "/" + ds.dataset).isin(split["train_datasets"])]
    assert len(ds) == len(split["train_datasets"]), "datasets.csv does not cover every training dataset"
    mean, std = load_action_stats()
    z = (np.array([[float(x) for x in s.split()] for s in ds.act_mean]) - mean) / std
    env = {"lo": z.min(0).round(4).tolist(), "hi": z.max(0).round(4).tolist()}
    hz = np.stack([mean_z(np.load(p)) for p in sorted((HOLDOUT / "actions").glob("*.npy"))])
    for d in DELTAS:
        routed = ((hz < np.array(env["lo"]) - d) | (hz > np.array(env["hi"]) + d)).any(1)
        if routed.mean() <= MAX_HOLDOUT_FRACTION:
            break
    ids = [p.stem for p in sorted((HOLDOUT / "actions").glob("*.npy"))]
    env.update({"delta": d, "n_train_datasets": len(ds), "holdout_routed": [i for i, r in zip(ids, routed) if r],
                "holdout_windows": len(ids), "max_holdout_fraction": MAX_HOLDOUT_FRACTION})
    ENVELOPE_PATH.write_text(json.dumps(env, indent=1), encoding="utf-8")
    return env


def route_table(root: Path, env: dict) -> dict:
    out = {}
    for p in sorted((root / "actions").glob("*.npy")):
        joints = routed_joints(np.load(p), env)
        if joints.any():
            out[p.stem] = [int(j) for j in np.flatnonzero(joints)]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    p = sub.add_parser("list")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("copy-unrouted")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--dst", type=Path, required=True)
    args = ap.parse_args()

    if args.cmd == "build":
        env = build()
        print(json.dumps({k: env[k] for k in ("lo", "hi", "delta", "holdout_routed")}))
        return
    env = load_envelope()
    table = route_table(args.root, env)
    if args.cmd == "list":
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"envelope": str(ENVELOPE_PATH), "n_routed": len(table), "routed": table}, indent=1))
        print(f"{len(table)} routed clips -> {args.out}")
    else:
        args.dst.mkdir(parents=True, exist_ok=True)
        n = 0
        for p in sorted((args.root / "actions").glob("*.npy")):
            if p.stem not in table:
                shutil.copy2(args.src / f"{p.stem}.mp4", args.dst / f"{p.stem}.mp4")
                n += 1
        print(f"copied {n} unrouted clips, {len(table)} routed left to generate")


if __name__ == "__main__":
    main()

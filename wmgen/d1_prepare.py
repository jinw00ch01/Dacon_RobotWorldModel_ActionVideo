"""Diagnostic D1 (CPU): per-window joint-calibration shifts for holdout_v1_sub64, all six joints, from training data only.

Question (docs/STATUS.md 15): do the models trained past 16k (v2_long2 22k, v4_aug soup) lose more than the 16k model when
a robot reads its joints with another zero? That would explain why both beat 16k on the clean holdout and lost on the board.

Shift of window w (raw units) = act_mean[k] - act_mean[own(w)], where own(w) is w's holdout dataset and k is a training
dataset drawn uniformly (seed 0) from the quarter of the 116 training datasets (configs/splits/holdout_v1.json) whose mean
pose is farthest (z-norm) from own(w): the window reads as if its robot had robot k's typical pose, i.e. the v4_aug
augmentation (calibration_offsets) at its largest scale 1.0 and toward its far end. A uniform k gives a mean |shift| of
only about 0.5 sigma per joint, too small to tell the models apart; the far quarter gives about 0.85. Means are over all
frames of the dataset (C:/Dacon/WM_Shared/data_index/datasets.csv). No eval number enters.

python -m wmgen.d1_prepare            # writes configs/d1_shifts.json
python -m wmgen.make_sim_holdout --src C:/Dacon/WM_Shared/holdout_v1_sub64 --out C:/Dacon/WM_Shared/holdout_v1_sub64_d1 \
    --shifts configs/d1_shifts.json   (wmgen.d1_run does this step)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from wmgen.offset_route import load_envelope, routed_joints
from wmscore.data import load_action_stats, load_split

REPO = Path(__file__).resolve().parents[1]
SUB64 = Path(r"C:\Dacon\WM_Shared\holdout_v1_sub64")
DATASETS_CSV = Path(r"C:\Dacon\WM_Shared\data_index\datasets.csv")
OUT = REPO / "configs" / "d1_shifts.json"
DETAIL = REPO / "configs" / "d1_shifts_detail.json"
SEED = 0
SCALE = 1.0
FAR_QUANTILE = 0.75


def main() -> None:
    split = load_split("holdout_v1")
    ds = pd.read_csv(DATASETS_CSV)
    ds["key"] = ds.user + "/" + ds.dataset
    act_mean = {k: np.array([float(x) for x in s.split()]) for k, s in zip(ds.key, ds.act_mean)}
    train = list(split["train_datasets"])
    assert all(k in act_mean for k in train), "datasets.csv misses a training dataset"
    windows = pd.read_csv(SUB64 / "windows.csv").sort_values("sample_id")
    rng = np.random.default_rng(SEED)
    mean, std = load_action_stats()
    env = load_envelope()
    train_z = np.stack([(act_mean[k] - mean) / std for k in train])
    shifts, rows = {}, []
    for w in windows.itertuples():
        own = f"{w.user}/{w.dataset}"
        assert own not in train, f"{w.sample_id}: own dataset {own} is a training dataset"
        dist = np.linalg.norm(train_z - (act_mean[own] - mean) / std, axis=1)
        far = np.flatnonzero(dist >= np.quantile(dist, FAR_QUANTILE))
        k = train[int(far[rng.integers(len(far))])]
        shift = SCALE * (act_mean[k] - act_mean[own])
        shifts[w.sample_id] = shift.round(4).tolist()
        a = np.load(SUB64 / "actions" / f"{w.sample_id}.npy")
        rows.append({"sample_id": w.sample_id, "own": own, "other": k, "shift_z": (shift / std).round(3).tolist(),
                     "routed_sub5": [int(j) for j in np.flatnonzero(routed_joints(a + shift, env))]})
    z = np.array([r["shift_z"] for r in rows])
    summary = {"n": len(rows), "seed": SEED, "scale": SCALE, "far_quantile": FAR_QUANTILE,
               "mean_abs_shift_z": np.abs(z).mean(0).round(3).tolist(),
               "max_abs_shift_z": np.abs(z).max(0).round(3).tolist(),
               "n_routed_sub5": sum(bool(r["routed_sub5"]) for r in rows)}
    OUT.write_text(json.dumps(shifts, indent=1), encoding="utf-8")  # the {sample_id: shift} form make_sim_holdout reads
    DETAIL.write_text(json.dumps({"summary": summary, "windows": rows}, indent=1), encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()

"""Pre-registered gates for submission 5 (offset routing). Thresholds are fixed here before the GPU run.

R1, mechanism: holdout_v1_sub64 with a simulated -2.5 sigma offset on shoulder_lift and elbow_flex, scored against the
offset targets. AUTO (relative channel on routed joints) vs KEEP, on the windows the rule routes. PASS only if
  - n routed windows >= R1_MIN_N
  - mean dT (idm_v2) <= R1_MAX_DT, and the upper bounds of both its window-level and its uploader-cluster paired
    bootstrap 95% CIs < 0 (windows of one uploader are correlated)
  - mean dT (idm_v1) <= R1_MAX_DT_V1
  - mean d(0.3 DINO + 0.3 R3D) <= 0
  - at most R1_MAX_DINO_BLOWUPS windows with dDINO > DINO_BLOWUP
R2, deployment cost: the real holdout_v1 windows the rule routes (true targets). AUTO vs KEEP (16k as in sub3).
  - sum(dT) / 192 <= R2_MAX_FULL_DT under both IDMs
  - no window with dDINO > DINO_BLOWUP
T = 0.3 DINO + 0.3 R3D + 0.4 Action per window; d = AUTO - KEEP, paired by sample_id.
Reported, never gated: the cost of the relative channel when there is no offset (AUTO under the simulation, scored
against the true targets, vs the sub3 16k videos on the same windows).

python -m wmgen.c2_gate --dir C:/Dacon/WM_Shared/c2_gates
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

R1_MIN_N = 10
R1_MAX_DT = -0.0075
R1_MAX_DT_V1 = -0.004
R1_MAX_DINO_BLOWUPS = 2
DINO_BLOWUP = 0.15
R2_MAX_FULL_DT = 0.002
FULL_WINDOWS = 192
BOOTSTRAP = 10_000
COLS = ["dino", "r3d", "action", "visual", "T"]


def per_window(csv: Path) -> pd.DataFrame:
    d = pd.read_csv(csv).set_index("sample_id")
    d["visual"] = 0.3 * d.dino + 0.3 * d.r3d
    d["T"] = d.visual + 0.4 * d.action
    return d


def paired(a: Path, b: Path, ids: list[str] | None = None) -> pd.DataFrame:
    """a - b per window (both score CSVs of the same windows), with the uploader of each window."""
    da, db = per_window(a), per_window(b)
    common = da.index.intersection(db.index) if ids is None else pd.Index(ids)
    out = da.loc[common, COLS] - db.loc[common, COLS]
    out["user"] = da.loc[common, "user"]
    return out


def bootstrap_ci(x: np.ndarray, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(x, (BOOTSTRAP, len(x))).mean(1)
    return np.percentile(means, [2.5, 97.5]).tolist()


def cluster_ci(x: np.ndarray, groups: np.ndarray, seed: int = 0) -> list[float]:
    """Resample whole uploaders with replacement; each draw is the window-weighted mean."""
    rng = np.random.default_rng(seed)
    names = np.unique(groups)
    sums = np.array([x[groups == g].sum() for g in names])
    counts = np.array([(groups == g).sum() for g in names])
    pick = rng.integers(0, len(names), (BOOTSTRAP, len(names)))
    return np.percentile(sums[pick].sum(1) / counts[pick].sum(1), [2.5, 97.5]).tolist()


def gate(d: Path, sub3_sub64: dict[str, Path]) -> dict:
    routed_sim = sorted(json.loads((d / "sim_routed.json").read_text())["routed"])
    r1v2 = paired(d / "sim_auto_idm_v2.csv", d / "sim_keep_idm_v2.csv", routed_sim)
    r1v1 = paired(d / "sim_auto_idm_v1.csv", d / "sim_keep_idm_v1.csv", routed_sim)
    nan2 = [float("nan"), float("nan")]
    ci = bootstrap_ci(r1v2["T"].to_numpy()) if len(r1v2) else nan2
    cci = cluster_ci(r1v2["T"].to_numpy(), r1v2["user"].to_numpy()) if len(r1v2) else nan2
    r1 = {"n": len(r1v2), "users": r1v2["user"].value_counts().to_dict(), "dT_v2": float(r1v2["T"].mean()),
          "dT_v2_ci95": ci, "dT_v2_cluster_ci95": cci, "dT_v1": float(r1v1["T"].mean()),
          "d_visual": float(r1v2["visual"].mean()), "d_dino": float(r1v2["dino"].mean()), "d_r3d": float(r1v2["r3d"].mean()),
          "d_action_v2": float(r1v2["action"].mean()), "d_action_v1": float(r1v1["action"].mean()),
          "dino_blowups": int((r1v2["dino"] > DINO_BLOWUP).sum())}
    r1["pass"] = bool(r1["n"] >= R1_MIN_N and r1["dT_v2"] <= R1_MAX_DT and ci[1] < 0 and cci[1] < 0
                      and r1["dT_v1"] <= R1_MAX_DT_V1 and r1["d_visual"] <= 0 and r1["dino_blowups"] <= R1_MAX_DINO_BLOWUPS)

    r2v2 = paired(d / "route_auto_idm_v2.csv", d / "route_keep_idm_v2.csv")
    r2v1 = paired(d / "route_auto_idm_v1.csv", d / "route_keep_idm_v1.csv")
    r2 = {"n": len(r2v2), "full_dT_v2": float(r2v2["T"].sum() / FULL_WINDOWS), "full_dT_v1": float(r2v1["T"].sum() / FULL_WINDOWS),
          "per_window_dT_v2": r2v2["T"].round(4).to_dict(), "per_window_dT_v1": r2v1["T"].round(4).to_dict(),
          "per_window_dDINO": r2v2["dino"].round(4).to_dict()}
    r2["pass"] = bool(r2["full_dT_v2"] <= R2_MAX_FULL_DT and r2["full_dT_v1"] <= R2_MAX_FULL_DT
                      and not (r2v2["dino"] > DINO_BLOWUP).any())

    control = {}
    for idm, base in sub3_sub64.items():
        if (d / f"simclean_auto_{idm}.csv").exists() and base.exists():
            c = paired(d / f"simclean_auto_{idm}.csv", base, routed_sim)
            control[idm] = {k: float(c[k].mean()) for k in COLS}
    thresholds = {k: v for k, v in globals().items() if k.isupper() and isinstance(v, (int, float))}
    return {"R1": r1, "R2": r2, "PASS": r1["pass"] and r2["pass"], "no_offset_control": control, "thresholds": thresholds}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    args = ap.parse_args()
    sub3_sub64 = {i: args.dir / f"sub64_16k_t20_{i}.csv" for i in ("idm_v2", "idm_v1")}
    verdict = gate(args.dir, sub3_sub64)
    (args.dir / "verdict.json").write_text(json.dumps(verdict, indent=1))
    print(json.dumps(verdict, indent=1))


if __name__ == "__main__":
    main()

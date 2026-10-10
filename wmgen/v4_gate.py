"""Pre-registered gate for submission 7: the v4_aug soup (calibration-offset augmented training) vs submission 5's model,
both with submission 5's inference pipeline (g3, 30 steps, seed 0, --abs-mode auto, anchor t20). Constants fixed before the GPU run.

G1, clean holdout_v1 (all 192 windows, true targets): d = V4 - 16k per window, T = 0.3 DINO + 0.3 R3D + 0.4 Action.
    mean dT(idm_v2) <= G1_MAX_DT, mean dT(idm_v1) <= G1_MAX_DT_V1, mean d(visual) <= G1_MAX_VISUAL, upper bounds of the
    window and uploader-cluster bootstrap 95% CIs of dT(idm_v2) < 0, at most G1_MAX_BLOWUPS windows with dDINO > DINO_BLOWUP.
G2, offset robustness on holdout_v1_s6_band (64 windows, shoulder/elbow shifted just below the training range, shifted
    targets): mean dT <= G2_MAX_DT under both IDMs (the s6 result showed that ignoring such offsets can cost action score).
PASS = G1 and G2. The 16k baseline is the existing full-192 KEEP scores with the 5 windows the sub5 rule routes replaced by
their AUTO scores (c2_gates/route_auto), i.e. exactly submission 5's pipeline.

python -m wmgen.v4_gate --dir C:/Dacon/WM_Shared/v4_gates
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from wmgen.c2_gate import BOOTSTRAP, COLS, bootstrap_ci, cluster_ci, per_window

G1_MAX_DT = -0.005
G1_MAX_DT_V1 = 0.0
G1_MAX_VISUAL = 0.0
G1_MAX_BLOWUPS = 4
G2_MAX_DT = 0.005
DINO_BLOWUP = 0.15
SIGMA_B = 0.13
BOARD_K = 216
SH = Path(r"C:\Dacon\WM_Shared")
IDMS = ("idm_v2", "idm_v1")


def baseline_full(idm: str) -> pd.DataFrame:
    keep = per_window(SH / "wmscore" / f"full192_v2long_s16000_g3_anchor_t20_{idm}.csv")
    auto = per_window(SH / "c2_gates" / f"route_auto_{idm}.csv")
    keep.loc[auto.index, auto.columns.intersection(keep.columns)] = auto[auto.columns.intersection(keep.columns)]
    keep["visual"] = 0.3 * keep.dino + 0.3 * keep.r3d
    keep["T"] = keep.visual + 0.4 * keep.action
    return keep


def diff(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    ids = a.index.intersection(b.index)
    out = a.loc[ids, COLS] - b.loc[ids, COLS]
    out["user"] = a.loc[ids, "user"]
    return out


def gate(d: Path) -> dict:
    g1 = {idm: diff(per_window(d / f"full_v4_{idm}.csv"), baseline_full(idm)) for idm in IDMS}
    v2, v1 = g1["idm_v2"], g1["idm_v1"]
    ci, cci = bootstrap_ci(v2["T"].to_numpy()), cluster_ci(v2["T"].to_numpy(), v2["user"].to_numpy())
    G1 = {"n": len(v2), "dT_v2": float(v2["T"].mean()), "dT_v2_ci95": ci, "dT_v2_cluster_ci95": cci, "dT_v1": float(v1["T"].mean()),
          "d_visual": float(v2["visual"].mean()), "d_dino": float(v2["dino"].mean()), "d_r3d": float(v2["r3d"].mean()),
          "d_action_v2": float(v2["action"].mean()), "d_action_v1": float(v1["action"].mean()),
          "dino_blowups": int((v2["dino"] > DINO_BLOWUP).sum()),
          "by_user_dT_v2": v2.groupby("user")["T"].mean().round(4).to_dict()}
    G1["pass"] = bool(G1["n"] == 192 and G1["dT_v2"] <= G1_MAX_DT and G1["dT_v1"] <= G1_MAX_DT_V1 and G1["d_visual"] <= G1_MAX_VISUAL
                      and ci[1] < 0 and cci[1] < 0 and G1["dino_blowups"] <= G1_MAX_BLOWUPS)
    G2 = {}
    if all((d / f"band_v4_{idm}.csv").exists() for idm in IDMS):
        g2 = {idm: diff(per_window(d / f"band_v4_{idm}.csv"), per_window(SH / "s6_gates" / f"band_auto_{idm}.csv")) for idm in IDMS}
        G2 = {"n": len(g2["idm_v2"]), **{f"{k}_{idm}": float(g2[idm][k].mean()) for idm in IDMS for k in ("T", "visual", "action")}}
        G2["pass"] = bool(G2["n"] == 64 and G2["T_idm_v2"] <= G2_MAX_DT and G2["T_idm_v1"] <= G2_MAX_DT)
    else:
        G2["pass"] = False
        G2["note"] = "not run (G1 failed first)"
    band = 2 * SIGMA_B * np.sqrt(BOARD_K) / 216
    thresholds = {k: v for k, v in globals().items() if k.isupper() and isinstance(v, (int, float))}
    return {"G1": G1, "G2": G2, "PASS": G1["pass"] and G2["pass"], "board_tie_band": band,
            "projected_dC": {f"ratio_{r}": r * G1["dT_v2"] for r in (2.0, 2.5, 3.5)}, "thresholds": thresholds, "bootstrap": BOOTSTRAP}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    args = ap.parse_args()
    verdict = gate(args.dir)
    (args.dir / "verdict.json").write_text(json.dumps(verdict, indent=1))
    print(json.dumps(verdict, indent=1))


if __name__ == "__main__":
    main()

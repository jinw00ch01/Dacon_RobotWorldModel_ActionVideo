"""Gate for the v5 candidate (Colab v5: 5.4x clips, latent-IDM loss, shift-5 noise levels), fixed before its GPU run.

The clean holdout ranked longer-trained weights wrongly twice (submissions 4 and 7), so here it only screens for breakage;
the board judges the gain. G1 and G2 compare v5 with submission 5's model and pipeline on the same windows.

G1, clean holdout_v1 (192 windows, true targets), v5 - 16k (16k = submission 5's videos: KEEP with the 5 routed windows
    replaced by AUTO): mean dT(idm_v2) <= G1_MAX_DT, mean dT(idm_v1) <= G1_MAX_DT, mean d(visual) <= G1_MAX_VISUAL, at most
    G1_MAX_BLOWUPS windows with dDINO > DINO_BLOWUP.
G2, diagnostic D1's offset set (holdout_v1_sub64_d1: every joint read through another training robot's zero; true
    targets), v5 - 16k keep: mean dT(idm_v2) <= G2_MAX_DT.
PASS = G1 and G2. Also reported: per-uploader dT, the components, the window and uploader-cluster CIs.

python -m wmgen.v5_gate --dir C:/Dacon/WM_Shared/v5_gates
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from wmgen.c2_gate import BOOTSTRAP, bootstrap_ci, cluster_ci, per_window
from wmgen.v4_gate import SH, baseline_full, diff

G1_MAX_DT = 0.005
G1_MAX_VISUAL = 0.005
G1_MAX_BLOWUPS = 4
G2_MAX_DT = 0.005
DINO_BLOWUP = 0.15
IDMS = ("idm_v2", "idm_v1")
D1 = SH / "d1_shift"


def summary(d) -> dict:
    t = d["T"].to_numpy()
    return {"n": len(d), "dT": float(t.mean()), "ci95": bootstrap_ci(t), "cluster_ci95": cluster_ci(t, d["user"].to_numpy()),
            "d_dino": float(d["dino"].mean()), "d_r3d": float(d["r3d"].mean()), "d_action": float(d["action"].mean()),
            "d_visual": float(d["visual"].mean()), "dino_blowups": int((d["dino"] > DINO_BLOWUP).sum()),
            "by_user_dT": d.groupby("user")["T"].mean().round(4).to_dict()}


def gate(d: Path) -> dict:
    g1 = {idm: summary(diff(per_window(d / f"full_v5_{idm}.csv"), baseline_full(idm))) for idm in IDMS}
    G1 = {**g1, "pass": bool(g1["idm_v2"]["n"] == 192 and g1["idm_v2"]["dT"] <= G1_MAX_DT and g1["idm_v1"]["dT"] <= G1_MAX_DT
                            and g1["idm_v2"]["d_visual"] <= G1_MAX_VISUAL and g1["idm_v2"]["dino_blowups"] <= G1_MAX_BLOWUPS)}
    G2 = {"pass": False, "note": "not run"}
    if all((d / f"shift_v5_{idm}.csv").exists() for idm in IDMS):
        g2 = {idm: summary(diff(per_window(d / f"shift_v5_{idm}.csv"), per_window(D1 / f"shift_16k_{idm}.csv"))) for idm in IDMS}
        G2 = {**g2, "pass": bool(g2["idm_v2"]["n"] == 64 and g2["idm_v2"]["dT"] <= G2_MAX_DT)}
    thresholds = {k: v for k, v in globals().items() if k.isupper() and isinstance(v, (int, float))}
    return {"G1": G1, "G2": G2, "PASS": G1["pass"] and G2["pass"], "thresholds": thresholds, "bootstrap": BOOTSTRAP,
            "board_tie_band_216": 2 * 0.13 * np.sqrt(216) / 216}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    args = ap.parse_args()
    verdict = gate(args.dir)
    (args.dir / "verdict.json").write_text(json.dumps(verdict, indent=1))
    print(json.dumps(verdict, indent=1))


if __name__ == "__main__":
    main()

"""Diagnostic D1 reading (wmgen.d1_run). Constants fixed before the GPU run (docs/STATUS.md 15). Diagnosis only.

Per window w and model m in {22k, v4}, with T = 0.3 DINO + 0.3 R3D + 0.4 Action:
  shift diff  S_m(w) = T_m(w) - T_16k(w) on the shifted-input videos (targets: the true actions, i.e. an extractor that
                       reads each robot in its own zero, which is what the board's static-video Action score of 0.4287 vs
                       0.34 for an exact a0 readout suggests)
  clean diff  C_m(w) = the same on the existing clean videos
  erosion     E_m(w) = S_m(w) - C_m(w): how much of m's clean advantage the reading offset takes away
Reading (idm_v2 decides, idm_v1 must agree in sign):
  "offset_erodes" for m  if mean E_m >= E_MIN, the uploader-cluster bootstrap 95% CI of E_m lies above 0 and
                          mean E_m(idm_v1) > 0
  "offset_flips" for m   if offset_erodes and mean S_m > 0 (m worse than 16k once readings are offset)
Both for 22k and v4 -> the post-16k losses on the board fit "offset sensitivity grew with training": the next model must
pass this test against 16k besides the clean holdout. Neither -> offsets within the training spread do not explain them.
Also reported: the visual part alone (no target convention), the shifted-target Action (an extractor that reads the
training zero), each model's own offset penalty T_shift - T_clean, and the windows whose clean v4 video was routed (sub5
rule), which are left out of every clean comparison.

python -m wmgen.d1_report --dir C:/Dacon/WM_Shared/d1_shift
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from wmgen.c2_gate import BOOTSTRAP, bootstrap_ci, cluster_ci
from wmgen.offset_route import load_envelope, routed_joints
from wmscore.data import load_action_stats

E_MIN = 0.010
SH = Path(r"C:\Dacon\WM_Shared")
SUB64, SIM = SH / "holdout_v1_sub64", SH / "holdout_v1_sub64_d1"
IDMS = ("idm_v2", "idm_v1")
MODELS = ("16k", "22k", "v4")
COLS = ["dino", "r3d", "action", "visual", "T"]


def load(d: Path, name: str, idm: str, shifted_targets: bool = False) -> pd.DataFrame:
    df = pd.read_csv(d / f"{name}_{idm}.csv").set_index("sample_id")
    if shifted_targets:
        mean, std = load_action_stats()
        pred = np.load(d / f"{name}_{idm}.idm_pred.npz")
        df["action"] = [float(np.abs(pred[i] - (np.load(SIM / "actions" / f"{i}.npy") - mean) / std).mean()) for i in df.index]
    df["visual"] = 0.3 * df.dino + 0.3 * df.r3d
    df["T"] = df.visual + 0.4 * df.action
    return df


def summary(x: pd.Series, users: pd.Series) -> dict:
    v = x.to_numpy()
    return {"mean": float(v.mean()), "ci95": bootstrap_ci(v), "cluster_ci95": cluster_ci(v, users.loc[x.index].to_numpy())}


def report(d: Path) -> dict:
    env = load_envelope()
    clean_routed = sorted(p.stem for p in (SUB64 / "actions").glob("*.npy") if routed_joints(np.load(p), env).any())
    out = {"clean_v4_routed_excluded": clean_routed, "models": {}, "vs_16k": {}}
    tables = {}
    for idm in IDMS:
        for m in MODELS:
            tables[(m, idm, "shift")] = load(d, f"shift_{m}", idm)
            tables[(m, idm, "shift_unaware")] = load(d, f"shift_{m}", idm, shifted_targets=True)
            tables[(m, idm, "clean")] = load(d, f"clean_{m}", idm)
    users = tables[("16k", "idm_v2", "shift")]["user"]
    keep = users.index.difference(clean_routed)
    for m in MODELS:
        out["models"][m] = {idm: {"T_shift": float(tables[(m, idm, "shift")]["T"].mean()),
                                  "T_clean": float(tables[(m, idm, "clean")].loc[keep, "T"].mean()),
                                  "penalty": float((tables[(m, idm, "shift")]["T"] - tables[(m, idm, "clean")]["T"]).loc[keep].mean()),
                                  "action_shift": float(tables[(m, idm, "shift")]["action"].mean()),
                                  "action_shift_unaware": float(tables[(m, idm, "shift_unaware")]["action"].mean()),
                                  "dino_shift": float(tables[(m, idm, "shift")]["dino"].mean()),
                                  "dino_clean": float(tables[(m, idm, "clean")].loc[keep, "dino"].mean())} for idm in IDMS}
    reading = {}
    for m in ("22k", "v4"):
        r = {}
        for idm in IDMS:
            s, c, su = (tables[(m, idm, k)][COLS] - tables[("16k", idm, k)][COLS] for k in ("shift", "clean", "shift_unaware"))
            e = (s["T"] - c["T"]).loc[keep]
            r[idm] = {"S": summary(s["T"], users), "C": float(c["T"].loc[keep].mean()), "E": summary(e, users),
                      "S_visual": float(s["visual"].mean()), "E_visual": summary((s["visual"] - c["visual"]).loc[keep], users),
                      "S_action": float(s["action"].mean()), "S_unaware": float(su["T"].mean()),
                      "S_dino": float(s["dino"].mean()), "C_dino": float(c["dino"].loc[keep].mean()),
                      "by_user_E": e.groupby(users.loc[keep]).mean().round(4).to_dict()}
        erodes = bool(r["idm_v2"]["E"]["mean"] >= E_MIN and r["idm_v2"]["E"]["cluster_ci95"][0] > 0
                      and r["idm_v1"]["E"]["mean"] > 0)
        r["offset_erodes"] = erodes
        r["offset_flips"] = bool(erodes and r["idm_v2"]["S"]["mean"] > 0)
        out["vs_16k"][m] = r
        reading[m] = "flips" if r["offset_flips"] else "erodes" if erodes else "no"
    both = all(out["vs_16k"][m]["offset_erodes"] for m in ("22k", "v4"))
    out["reading"] = {**reading, "conclusion": "offset sensitivity explains the post-16k board losses" if both else
                      "offsets within the training spread do not explain both post-16k board losses" if not any(
                          out["vs_16k"][m]["offset_erodes"] for m in ("22k", "v4")) else "mixed: see vs_16k"}
    out["thresholds"] = {"E_MIN": E_MIN, "BOOTSTRAP": BOOTSTRAP}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    args = ap.parse_args()
    verdict = report(args.dir)
    (args.dir / "verdict.json").write_text(json.dumps(verdict, indent=1))
    print(json.dumps(verdict, indent=1))


if __name__ == "__main__":
    main()

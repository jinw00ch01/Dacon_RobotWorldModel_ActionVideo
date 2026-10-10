"""Pre-registered gate for submission 6 (shoulder_lift lower band, docs/STATUS.md item 9). Constants fixed before the GPU run.

Arms on holdout_v1_s6_band (64 windows from wmgen.s6_prepare, inputs and targets shifted): AUTO (sub5 rule, routes nothing
here) and LIFTLO (s6 rule, routes shoulder_lift on all 64). d = LIFTLO - AUTO per window, T = 0.3 DINO + 0.3 R3D + 0.4 Action.
R1 (shifted targets, all 64): mean dT(idm_v2) <= R1_MAX_DT, mean dT(idm_v1) <= R1_MAX_DT_V1, mean d(visual) <= R1_MAX_VISUAL,
    upper bounds of the window and uploader-cluster bootstrap 95% CIs of dT(idm_v2) < 0, at most R1_MAX_BLOWUPS windows with
    dDINO > DINO_BLOWUP, and in each stratum (P1, P2) mean dT(idm_v2) <= STRATUM_MAX_DT.
R2 (cost when the reading is right, true targets):
    P1 LIFTLO videos (their input features equal the clean ones) vs the 16k KEEP videos of the same windows:
    mean dT(idm_v2) <= R2_MAX_DT, mean dT(idm_v1) <= R2_MAX_DT_V1, mean d(visual) <= R2_MAX_VISUAL, dDINO > DINO_BLOWUP on
    at most R2_MAX_BLOWUPS windows; hold_000003 (the only clean holdout_v1 window the rule changes) LIFTLO vs 16k:
    dT / 192 <= H003_MAX_FULL_DT under both IDMs and dDINO <= DINO_BLOWUP.
PASS = R1 and R2. Diagnostics are reported, never gated.

python -m wmgen.s6_gate --dir C:/Dacon/WM_Shared/s6_gates
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from wmgen.c2_gate import BOOTSTRAP, COLS, bootstrap_ci, cluster_ci, paired, per_window

R1_N = 64
R1_MAX_DT = -0.012
R1_MAX_DT_V1 = -0.006
R1_MAX_VISUAL = -0.006
R1_MAX_BLOWUPS = 2
STRATUM_MAX_DT = 0.005
R2_MAX_DT = 0.005
R2_MAX_DT_V1 = 0.005
R2_MAX_VISUAL = 0.003
R2_MAX_BLOWUPS = 1
H003_MAX_FULL_DT = 0.002
DINO_BLOWUP = 0.15
FULL_WINDOWS = 192
SIGMA_B = 0.13
BOARD_K = 33
RATIO = 2.0
SH = Path(r"C:\Dacon\WM_Shared")
KEEP_FULL = {i: SH / "wmscore" / f"full192_v2long_s16000_g3_anchor_t20_{i}.csv" for i in ("idm_v2", "idm_v1")}
KEEP_SUB64 = {i: SH / "c2_gates" / f"sub64_16k_t20_{i}.csv" for i in ("idm_v2", "idm_v1")}
H003 = "hold_000003"


def gate(d: Path, design: dict) -> dict:
    strata = {i: w["stratum"] for i, w in design["windows"].items()}
    ids = sorted(strata)
    r1 = {}
    for idm in ("idm_v2", "idm_v1"):
        r1[idm] = paired(d / f"band_liftlo_{idm}.csv", d / f"band_auto_{idm}.csv", ids)
    v2, v1 = r1["idm_v2"], r1["idm_v1"]
    ci, cci = bootstrap_ci(v2["T"].to_numpy()), cluster_ci(v2["T"].to_numpy(), v2["user"].to_numpy())
    by_stratum = {s: {k: float(v2.loc[[i for i in ids if strata[i] == s], k].mean()) for k in COLS} for s in ("P1", "P2")}
    R1 = {"n": len(v2), "dT_v2": float(v2["T"].mean()), "dT_v2_ci95": ci, "dT_v2_cluster_ci95": cci,
          "dT_v1": float(v1["T"].mean()), "d_visual": float(v2["visual"].mean()), "d_dino": float(v2["dino"].mean()),
          "d_r3d": float(v2["r3d"].mean()), "d_action_v2": float(v2["action"].mean()), "d_action_v1": float(v1["action"].mean()),
          "dino_blowups": int((v2["dino"] > DINO_BLOWUP).sum()), "by_stratum_v2": by_stratum}
    R1["pass"] = bool(R1["n"] == R1_N and R1["dT_v2"] <= R1_MAX_DT and R1["dT_v1"] <= R1_MAX_DT_V1
                      and R1["d_visual"] <= R1_MAX_VISUAL and ci[1] < 0 and cci[1] < 0
                      and R1["dino_blowups"] <= R1_MAX_BLOWUPS
                      and all(by_stratum[s]["T"] <= STRATUM_MAX_DT for s in by_stratum))

    p1 = sorted(i for i in ids if strata[i] == "P1")
    r2 = {idm: paired(d / f"p1clean_liftlo_{idm}.csv", KEEP_FULL[idm], p1) for idm in ("idm_v2", "idm_v1")}
    h = {idm: paired(d / f"h003_liftlo_{idm}.csv", KEEP_SUB64[idm], [H003]) for idm in ("idm_v2", "idm_v1")}
    R2 = {"n_p1": len(r2["idm_v2"]), "dT_v2": float(r2["idm_v2"]["T"].mean()), "dT_v1": float(r2["idm_v1"]["T"].mean()),
          "d_visual": float(r2["idm_v2"]["visual"].mean()), "dino_blowups": int((r2["idm_v2"]["dino"] > DINO_BLOWUP).sum()),
          "h003_dT_v2": float(h["idm_v2"]["T"].iloc[0]), "h003_dT_v1": float(h["idm_v1"]["T"].iloc[0]),
          "h003_dDINO": float(h["idm_v2"]["dino"].iloc[0])}
    R2["pass"] = bool(R2["dT_v2"] <= R2_MAX_DT and R2["dT_v1"] <= R2_MAX_DT_V1 and R2["d_visual"] <= R2_MAX_VISUAL
                      and R2["dino_blowups"] <= R2_MAX_BLOWUPS
                      and R2["h003_dT_v2"] / FULL_WINDOWS <= H003_MAX_FULL_DT and R2["h003_dT_v1"] / FULL_WINDOWS <= H003_MAX_FULL_DT
                      and R2["h003_dDINO"] <= DINO_BLOWUP)

    # diagnostics (never gated): lift-shift terciles, AUTO harm vs 16k on the shifted windows, projected board change
    shift = {i: w["z_target"][1] - w["z_clean"][1] for i, w in design["windows"].items()}
    terc = np.quantile(list(shift.values()), [1 / 3, 2 / 3])
    tercile = {i: int(np.searchsorted(terc, s)) for i, s in shift.items()}  # 0 = largest downward shift
    by_tercile = {t: float(v2.loc[[i for i in ids if tercile[i] == t], "T"].mean()) for t in (0, 1, 2)}
    auto_harm = {}
    for idm in ("idm_v2",):
        a = per_window(d / f"band_auto_{idm}.csv").loc[ids]
        k = per_window(KEEP_FULL[idm]).loc[ids]
        auto_harm = {"d_dino_auto_vs_16k": float((a.dino - k.dino).mean()), "d_visual_auto_vs_16k": float((a.visual - k.visual).mean())}
    projected = {f"ratio_{r}": BOARD_K / 216 * r * R1["dT_v2"] for r in (1.7, RATIO, 3.0)}
    band = 2 * SIGMA_B * np.sqrt(BOARD_K) / 216
    thresholds = {k: v for k, v in globals().items() if k.isupper() and isinstance(v, (int, float))}
    return {"R1": R1, "R2": R2, "PASS": R1["pass"] and R2["pass"],
            "diagnostics": {"dT_v2_by_lift_shift_tercile_most_to_least_negative": by_tercile, **auto_harm,
                            "projected_dC": projected, "board_tie_band": band},
            "thresholds": thresholds, "bootstrap": BOOTSTRAP}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--design", type=Path, default=Path(__file__).resolve().parents[1] / "configs" / "s6_band_design.json")
    args = ap.parse_args()
    verdict = gate(args.dir, json.loads(args.design.read_text(encoding="utf-8")))
    (args.dir / "verdict.json").write_text(json.dumps(verdict, indent=1))
    print(json.dumps(verdict, indent=1))


if __name__ == "__main__":
    main()

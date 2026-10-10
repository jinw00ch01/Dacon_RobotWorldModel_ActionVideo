"""CPU preparation for the submission-6 gate (wmgen.s6_gate): a fresh shoulder_lift band test built from training data only.

Windows: holdout_v1 minus sub64 (sub64 produced the hypothesis), minus windows the sub5 rule already routes on clean inputs.
N windows are drawn per uploader by largest remainder (seed SEED). In a seed-SEED order, the k-th window gets its
shoulder_lift clip-mean z moved to lo_1 - BAND*(k+0.5)/N (inside the band the sub5 rule leaves alone and the s6 rule
routes). Odd k (stratum P2) also get elbow_flex moved to lo_2 - BAND*(m+0.5)/(N/2) (seed ELBOW_SEED order), the
pattern of eval clips with both joints low but neither beyond the sub5 margin. Inputs and targets both carry the shift.
Every constant comes from the training envelope (configs/offset_envelope.json) and the delta grid; none from eval.

python -m wmgen.s6_prepare
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from wmgen.actions import action_features
from wmgen.offset_route import ENVELOPES, REPO, load_envelope, mean_z, route_table, routed_joints
from wmscore.data import load_action_stats

SEED, ELBOW_SEED, N = 0, 1, 64
SH = Path(r"C:\Dacon\WM_Shared")
FULL, SUB64 = SH / "holdout_v1", SH / "holdout_v1_sub64"
BAND_ROOT, P1_ROOT, H003_ROOT = SH / "holdout_v1_s6_band", SH / "holdout_v1_s6_p1clean", SH / "holdout_v1_s6_h003"
DESIGN = REPO / "configs" / "s6_band_design.json"
H003 = "hold_000003"
CREATE_NO_WINDOW = 0x08000000


def select() -> tuple[list[str], dict]:
    sub5 = load_envelope(ENVELOPES["sub5"])
    BAND = float(sub5["delta"])
    windows = pd.read_csv(FULL / "windows.csv")
    sub64 = {p.stem for p in (SUB64 / "images").glob("*.png")}
    exclude = set(route_table(FULL, sub5))
    elig = windows[~windows.sample_id.isin(sub64 | exclude)].sort_values("sample_id")
    counts = elig.groupby("user").size().sort_index()
    quota = counts * N / len(elig)
    k = np.floor(quota).astype(int)
    for u in (quota - k).sort_values(ascending=False, kind="stable").index[: N - k.sum()]:
        k[u] += 1
    rng = np.random.default_rng(SEED)
    chosen = []
    for u in counts.index:
        ids = sorted(elig[elig.user == u].sample_id)
        chosen += [ids[i] for i in rng.permutation(len(ids))[: k[u]]]
    order = [chosen[i] for i in np.random.default_rng(SEED).permutation(len(chosen))]
    meta = {"band": BAND, "eligible": len(elig), "excluded_sub5_routed": sorted(exclude), "per_user": k.to_dict()}
    return order, meta


def design() -> dict:
    order, meta = select()
    sub5 = load_envelope(ENVELOPES["sub5"])
    lo, band = np.asarray(sub5["lo"]), meta["band"]
    _, std = load_action_stats()
    p2 = order[1::2]
    elbow_rank = {i: m for m, i in enumerate(p2[j] for j in np.random.default_rng(ELBOW_SEED).permutation(len(p2)))}
    windows = pd.read_csv(FULL / "windows.csv").set_index("sample_id")
    rows, shifts = {}, {}
    for k, i in enumerate(order):
        z = mean_z(np.load(FULL / "actions" / f"{i}.npy"))
        target = z.copy()
        target[1] = lo[1] - band * (k + 0.5) / len(order)
        stratum = "P1"
        if i in elbow_rank:
            stratum = "P2"
            target[2] = lo[2] - band * (elbow_rank[i] + 0.5) / len(p2)
        shift = ((target - z) * std).astype(np.float32)
        shifts[i] = shift.tolist()
        rows[i] = {"k": k, "user": windows.loc[i, "user"], "stratum": stratum, "z_clean": z.round(4).tolist(),
                   "z_target": target.round(4).tolist(), "shift_raw": shift.tolist()}
    return {"seed": SEED, "elbow_seed": ELBOW_SEED, "n": len(order), **meta, "windows": rows, "shifts": shifts}


def build_root(out: Path, ids_or_shifts: dict | list, kind: str) -> None:
    tmp = out.parent / f"{out.name}.input.json"
    tmp.write_text(json.dumps(ids_or_shifts))
    args = ["--shifts", str(tmp)] if kind == "shifts" else ["--ids", str(tmp)]
    subprocess.run([sys.executable, "-m", "wmgen.make_sim_holdout", "--src", str(FULL), "--out", str(out), *args],
                   check=True, creationflags=CREATE_NO_WINDOW, cwd=REPO)


def check(d: dict) -> None:
    sub5, s6 = load_envelope(ENVELOPES["sub5"]), load_envelope(ENVELOPES["s6"])
    for i, w in d["windows"].items():
        a = np.load(BAND_ROOT / "actions" / f"{i}.npy")
        assert not routed_joints(a, sub5).any(), f"sub5 rule routes {i}"
        assert np.flatnonzero(routed_joints(a, s6)).tolist() == [1], f"s6 rule does not route exactly lift on {i}"
        if w["stratum"] == "P1":
            clean = np.load(FULL / "actions" / f"{i}.npy")
            mask = routed_joints(a, s6)
            f_shift = action_features(a, mask).to(torch.bfloat16)
            f_clean = action_features(clean, mask).to(torch.bfloat16)
            assert torch.equal(f_shift, f_clean), f"P1 features differ from clean on {i}"
    h = np.load(H003_ROOT / "actions" / f"{H003}.npy")
    assert not routed_joints(h, sub5).any() and np.flatnonzero(routed_joints(h, s6)).tolist() == [1]


def main() -> None:
    d = design()
    DESIGN.write_text(json.dumps(d, indent=1), encoding="utf-8")
    build_root(BAND_ROOT, d["shifts"], "shifts")
    build_root(P1_ROOT, [i for i, w in d["windows"].items() if w["stratum"] == "P1"], "ids")
    build_root(H003_ROOT, [H003], "ids")
    check(d)
    strata = pd.Series({i: w["stratum"] for i, w in d["windows"].items()}).value_counts().to_dict()
    print(json.dumps({"n": d["n"], "eligible": d["eligible"], "per_user": d["per_user"], "strata": strata,
                      "lift_shift_sigma_range": [round(min(w["z_target"][1] - w["z_clean"][1] for w in d["windows"].values()), 2),
                                                 round(max(w["z_target"][1] - w["z_clean"][1] for w in d["windows"].values()), 2)]}))


if __name__ == "__main__":
    main()

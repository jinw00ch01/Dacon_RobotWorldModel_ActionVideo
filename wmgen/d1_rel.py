"""Diagnostic D1, extra arm (reported, not part of the pre-registered reading): the 16k model with --abs-mode rel.

With every joint relative the generator input is the same with or without a reading offset, so one run on clean
holdout_v1_sub64 gives both its clean score and its offset score. Set against wmgen.d1_run's 16k keep scores it prices
dropping the absolute channel at inference: what it costs on clean robots and what it saves on offset ones.

python -m wmgen.d1_rel
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from wmgen.c2_run import GUIDANCE, IDMS, THRESHOLD, run
from wmgen.d1_run import OUT, SUB64, n_mp4


def table(name: str, idm: str) -> pd.DataFrame:
    df = pd.read_csv(OUT / f"{name}_{idm}.csv").set_index("sample_id")
    df["visual"] = 0.3 * df.dino + 0.3 * df.r3d
    df["T"] = df.visual + 0.4 * df.action
    return df


def main() -> None:
    raw, anchored = OUT / "pred" / "16k_rel" / "videos", OUT / "pred" / "16k_rel_anchor_t20" / "videos"
    n = n_mp4(SUB64 / "gt_videos")
    if n_mp4(raw) < n:
        run("wmgen.gen_cosmos_ac", "--adapter", str(Path(r"C:\Dacon\WM_Shared\cosmos_ac\runs\v2_long\adapter_016000.pt")),
            "--eval-root", str(SUB64), "--out", str(raw), "--guidance", GUIDANCE, "--abs-mode", "rel")
    if n_mp4(anchored) < n:
        run("wmgen.apply_anchor", "--pred", str(raw), "--eval-root", str(SUB64), "--out", str(anchored), "--threshold", THRESHOLD)
    for idm_name, idm in IDMS.items():
        dst = OUT / f"rel_16k_{idm_name}.json"
        if not dst.exists():
            run("wmscore.score", "--pred", str(anchored), "--holdout", str(SUB64), "--idm", str(idm), "--out", str(dst), "--save-pred")
    out = {}
    for idm in IDMS:
        rel = table("rel_16k", idm)
        for ref in ("clean_16k", "shift_16k"):
            if (OUT / f"{ref}_{idm}.csv").exists():
                d = rel[["dino", "r3d", "action", "visual", "T"]] - table(ref, idm)[["dino", "r3d", "action", "visual", "T"]]
                out[f"rel_minus_{ref}_{idm}"] = d.mean().round(4).to_dict()
    (OUT / "rel_arm.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    main()

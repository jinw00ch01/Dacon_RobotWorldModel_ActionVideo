"""GPU job: compare inference settings of one adapter on a holdout set (generate, anchor t20, score with both IDMs).

Each setting is name=extra gen_cosmos_ac flags, e.g. "ag06=--guidance 0 --auto-guidance 0.6". Videos go to
<holdout>/pred/<tag>_<name>/videos, scores to <out>/<name>_<idm>.json; finished steps are skipped. Holdout data only.

python -m wmgen.inf_sweep --adapter C:/Dacon/WM_Shared/cosmos_ac/runs/v2_long/adapter_016000.pt --tag 16k \
    --setting "ag06=--guidance 0 --auto-guidance 0.6" --setting "g3hold=--guidance 3 --null hold"
"""
from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

import pandas as pd

from wmgen.c2_run import IDMS, THRESHOLD, run

SH = Path(r"C:\Dacon\WM_Shared")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--tag", required=True, help="short adapter name for folder names")
    ap.add_argument("--holdout", type=Path, default=SH / "holdout_v1_sub64")
    ap.add_argument("--out", type=Path, default=SH / "inf_sweep")
    ap.add_argument("--setting", action="append", required=True, help="name=gen_cosmos_ac flags")
    args = ap.parse_args()
    out = args.out / args.tag
    out.mkdir(parents=True, exist_ok=True)
    n = len(list((args.holdout / "images").glob("*.png")))
    rows = []
    for spec in args.setting:
        name, flags = spec.split("=", 1)
        raw = args.holdout / "pred" / f"{args.tag}_{name}" / "videos"
        anchored = args.holdout / "pred" / f"{args.tag}_{name}_anchor_t{THRESHOLD}" / "videos"
        if len(list(raw.glob("*.mp4"))) < n:
            run("wmgen.gen_cosmos_ac", "--adapter", str(args.adapter), "--eval-root", str(args.holdout), "--out", str(raw),
                *shlex.split(flags))
        if len(list(anchored.glob("*.mp4"))) < n:
            run("wmgen.apply_anchor", "--pred", str(raw), "--eval-root", str(args.holdout), "--out", str(anchored),
                "--threshold", THRESHOLD)
        row = {"setting": name, "flags": flags}
        for idm_name, idm in IDMS.items():
            dst = out / f"{name}_{idm_name}.json"
            if not dst.exists():
                run("wmscore.score", "--pred", str(anchored), "--holdout", str(args.holdout), "--idm", str(idm),
                    "--out", str(dst))
            s = json.loads(dst.read_text())
            row.update({f"T_{idm_name}": round(s["score"], 4), f"action_{idm_name}": round(s["action"], 4)})
            row.update({"dino": round(s["dino"], 4), "r3d": round(s["r3d"], 4)})
        rows.append(row)
        print(json.dumps(row), flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(out / "summary.csv", index=False)
    print(table.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

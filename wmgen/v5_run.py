"""GPU job for the v5 candidate: gate (wmgen.v5_gate) on the clean holdout and the D1 offset set, then, only on PASS, the
eval build (216 clips, anchor t20, submission kit CSV). Resumable: finished steps are skipped.

The generation flags are fixed with the gate before the run (docs/STATUS.md); a model trained with --rel-only switches
itself to relative readings in wmgen.gen_cosmos_ac.

python -m wmgen.v5_run --soup C:/Dacon/WM_Shared/cosmos_ac/soups/soup_mean_v5_30500-32000.pt --gen-flags "--guidance 3 --abs-mode auto"
"""
from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

from wmgen.c2_run import IDMS, THRESHOLD, run

SH = Path(r"C:\Dacon\WM_Shared")
FULL, SUB64, SIM = SH / "holdout_v1", SH / "holdout_v1_sub64", SH / "holdout_v1_sub64_d1"
EVAL = Path(r"C:\Dacon\RobotWorldModel_ActionVideo\open\data\eval")


def n_mp4(d: Path) -> int:
    return len(list(d.glob("*.mp4"))) if d.exists() else 0


def gen_anchor(adapter: Path, root: Path, out: Path, flags: list[str]) -> Path:
    n = len(list((root / "images").glob("*.png")))
    raw, anchored = out / "raw" / "videos", out / f"anchor_t{THRESHOLD}" / "videos"
    if n_mp4(raw) < n:
        run("wmgen.gen_cosmos_ac", "--adapter", str(adapter), "--eval-root", str(root), "--out", str(raw), *flags)
    if n_mp4(anchored) < n:
        run("wmgen.apply_anchor", "--pred", str(raw), "--eval-root", str(root), "--out", str(anchored), "--threshold", THRESHOLD)
    return anchored


def score(pred: Path, targets: Path, dst_dir: Path, name: str) -> None:
    for idm_name, idm in IDMS.items():
        dst = dst_dir / f"{name}_{idm_name}.json"
        if not dst.exists():
            run("wmscore.score", "--pred", str(pred), "--holdout", str(targets), "--idm", str(idm), "--out", str(dst))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--soup", type=Path, required=True)
    ap.add_argument("--gen-flags", required=True, help="wmgen.gen_cosmos_ac flags of the candidate pipeline")
    ap.add_argument("--out", type=Path, default=SH / "v5_gates")
    ap.add_argument("--submission", type=Path, default=SH / "submissions" / "sub8")
    args = ap.parse_args()
    assert args.soup.exists(), args.soup
    flags = shlex.split(args.gen_flags)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "run_args.json").write_text(json.dumps({"soup": str(args.soup), "gen_flags": args.gen_flags}, indent=1))
    score(gen_anchor(args.soup, FULL, args.out / "full", flags), FULL, args.out, "full_v5")
    # offset set: same flags with the absolute channel kept, as for the 16k keep reference
    keep = [f for f in flags]
    if "--abs-mode" in keep:
        i = keep.index("--abs-mode")
        keep[i + 1] = "keep"
    score(gen_anchor(args.soup, SIM, args.out / "shift", keep), SUB64, args.out, "shift_v5")
    run("wmgen.v5_gate", "--dir", str(args.out))
    if not json.loads((args.out / "verdict.json").read_text())["PASS"]:
        print("v5 gate FAILED: no eval build", flush=True)
        return
    videos = gen_anchor(args.soup, EVAL, args.submission, flags)
    csv = args.submission / f"submission_v5_soup_{'_'.join(f.lstrip('-') for f in flags)}_anchor_t{THRESHOLD}.csv"
    if not csv.exists():
        run("wmgen.make_kit_csv", "--videos", str(videos), "--out", str(csv))
    print("submission:", csv, flush=True)


if __name__ == "__main__":
    main()

"""Pick the best adapter by our own holdout score, then build its eval submission end to end.

Selection uses only wmscore results on the training-data holdout (never the submission kit, rule 7).
Steps: generate the 216 eval clips -> background anchor -> official submission kit CSV.

python -m wmgen.submit_best --out C:/Dacon/WM_Shared/submissions/sub3 --guidance 3 --threshold 20 \
    --candidate v2long_s12000=C:/.../adapter_012000.pt=C:/.../sub64_v2long_s12000_g3_anchor_t20_idm_v2.json ...
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

CREATE_NO_WINDOW = 0x08000000
EVAL_ROOT = Path(r"C:\Dacon\RobotWorldModel_ActionVideo\open\data\eval")


def run(cmd: list[str]) -> None:
    print("running:", " ".join(cmd), flush=True)
    code = subprocess.call(cmd, creationflags=CREATE_NO_WINDOW, stdout=sys.stdout, stderr=sys.stderr)
    if code:
        raise SystemExit(f"step failed with exit code {code}: {cmd[2]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--candidate", action="append", required=True, help="name=adapter_path=score_json")
    ap.add_argument("--guidance", type=float, default=3.0)
    ap.add_argument("--threshold", type=float, default=20.0)
    ap.add_argument("--eval-root", type=Path, default=EVAL_ROOT)
    ap.add_argument("--abs-mode", choices=["keep", "auto", "liftlo", "rel"], default="keep")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    scored = []
    for spec in args.candidate:
        name, adapter, score_json = spec.split("=", 2)
        path = Path(score_json)
        if path.exists():
            scored.append((json.loads(path.read_text())["score"], name, adapter, score_json))
        else:
            print(f"no holdout score for {name}, skipped", flush=True)
    if not scored:
        raise SystemExit("no scored candidates")
    scored.sort()
    score, name, adapter, score_json = scored[0]
    args.out.mkdir(parents=True, exist_ok=True)
    choice = {"chosen": name, "adapter": adapter, "holdout_score": score, "score_json": score_json,
              "guidance": args.guidance, "anchor_threshold": args.threshold, "abs_mode": args.abs_mode, "seed": args.seed,
              "ranking": [{"name": n, "holdout_score": s} for s, n, _, _ in scored]}
    (args.out / "choice.json").write_text(json.dumps(choice, indent=1))
    print(json.dumps(choice), flush=True)

    py = sys.executable
    raw, anchored = args.out / "raw" / "videos", args.out / f"anchor_t{args.threshold:g}" / "videos"
    run([py, "-m", "wmgen.gen_cosmos_ac", "--adapter", adapter, "--eval-root", str(args.eval_root),
         "--out", str(raw), "--guidance", str(args.guidance), "--abs-mode", args.abs_mode, "--seed", str(args.seed)])
    run([py, "-m", "wmgen.apply_anchor", "--pred", str(raw), "--eval-root", str(args.eval_root),
         "--out", str(anchored), "--threshold", str(args.threshold)])
    extra = (f"_abs{args.abs_mode}" if args.abs_mode != "keep" else "") + (f"_seed{args.seed}" if args.seed else "")
    csv = args.out / f"submission_{name}_g{args.guidance:g}{extra}_anchor_t{args.threshold:g}.csv"
    run([py, "-m", "wmgen.make_kit_csv", "--videos", str(anchored), "--out", str(csv)])
    print("submission:", csv, flush=True)


if __name__ == "__main__":
    main()

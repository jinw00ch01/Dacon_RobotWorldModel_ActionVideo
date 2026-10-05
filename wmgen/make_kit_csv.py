"""Run the untouched submission kit in .venv-kit on a finished video folder (rule 7: CSV only).

python -m wmgen.make_kit_csv --videos C:/Dacon/WM_Shared/s0_first_frame/videos --out C:/Dacon/WM_Shared/s0_first_frame/submission.csv
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

MAIN = Path(r"C:\Dacon\RobotWorldModel_ActionVideo")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--kit-python", type=Path, default=MAIN / ".venv-kit" / "Scripts" / "python.exe")
    ap.add_argument("--open-root", type=Path, default=MAIN / "open")
    args = ap.parse_args()

    kit = args.open_root / "submission_kit"
    cmd = [
        str(args.kit_python), "make_submission_csv.py",
        "--prediction-root", str(args.videos.resolve()),
        "--output-csv", str(args.out.resolve()),
    ]
    print(" ".join(cmd), flush=True)
    sys.exit(subprocess.call(cmd, cwd=kit))


if __name__ == "__main__":
    main()

"""GPU job for the submission-5 gates (wmgen.c2_gate): generate, anchor, score, judge. Resumable: finished steps are skipped.

Needs the CPU preparation first:
  python -m wmgen.offset_route build
  python -m wmgen.make_sim_holdout --src C:/Dacon/WM_Shared/holdout_v1_sub64 --out C:/Dacon/WM_Shared/holdout_v1_sub64_simoff25 --joints 1 2 --sigma -2.5
  python -m wmgen.offset_route list --root C:/Dacon/WM_Shared/holdout_v1 --out C:/Dacon/WM_Shared/c2_gates/holdout_routed.json
  python -m wmgen.make_sim_holdout --src C:/Dacon/WM_Shared/holdout_v1 --out C:/Dacon/WM_Shared/holdout_v1_route5 --ids C:/Dacon/WM_Shared/c2_gates/holdout_routed.json
and the 16k raw videos for all 192 holdout windows in holdout_v1/pred/v2long_s16000_g3/videos.

python -m wmgen.c2_run
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from wmgen.offset_route import load_envelope, route_table
from wmgen.video_io import read_mp4

CREATE_NO_WINDOW = 0x08000000
SH = Path(r"C:\Dacon\WM_Shared")
ADAPTER = SH / "cosmos_ac" / "runs" / "v2_long" / "adapter_016000.pt"
SUB64, FULL = SH / "holdout_v1_sub64", SH / "holdout_v1"
SIM, ROUTE = SH / "holdout_v1_sub64_simoff25", SH / "holdout_v1_route5"
OUT = SH / "c2_gates"
FULL16 = FULL / "pred" / "v2long_s16000_g3"
IDMS = {"idm_v2": SH / "idm" / "idm_v2.pt", "idm_v1": SH / "idm" / "idm_v1.pt"}
GUIDANCE, THRESHOLD = "3", "20"


def run(*args: str) -> None:
    cmd = [sys.executable, "-m", *args]
    print("running:", " ".join(cmd), flush=True)
    if subprocess.call(cmd, creationflags=CREATE_NO_WINDOW, stdout=sys.stdout, stderr=sys.stderr):
        raise SystemExit(f"failed: {' '.join(args[:2])}")


def gen(root: Path, out: Path, mode: str, limit: int = 0) -> None:
    n = len(list((root / "images").glob("*.png")))
    if out.exists() and len(list(out.glob("*.mp4"))) >= (limit or n):
        return
    run("wmgen.gen_cosmos_ac", "--adapter", str(ADAPTER), "--eval-root", str(root), "--out", str(out),
        "--guidance", GUIDANCE, "--abs-mode", mode, *(["--limit", str(limit)] if limit else []))


def anchor(root: Path, raw: Path, out: Path) -> None:
    n = len(list((root / "images").glob("*.png")))
    if out.exists() and len(list(out.glob("*.mp4"))) >= n:
        return
    run("wmgen.apply_anchor", "--pred", str(raw), "--eval-root", str(root), "--out", str(out), "--threshold", THRESHOLD)


def score(pred: Path, holdout: Path, out: Path) -> None:
    for name, idm in IDMS.items():
        dst = out.parent / f"{out.name}_{name}.json"
        if not dst.exists():
            run("wmscore.score", "--pred", str(pred), "--holdout", str(holdout), "--idm", str(idm), "--out", str(dst))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(((a.astype(np.float64) - b) ** 2).mean())
    return 99.0 if mse == 0 else 10 * np.log10(255 ** 2 / mse)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for p in (SIM / "actions", ROUTE / "actions", FULL16 / "videos"):
        assert p.exists(), f"missing preparation: {p}"
    assert len(list((FULL16 / "videos").glob("*.mp4"))) == 192, "16k full-holdout generation not finished"

    # determinism probe (diagnostic only): regenerate 4 sub64 windows with the sub3 settings
    probe = OUT / "probe" / "videos"
    gen(SUB64, probe, "keep", limit=4)
    ref = SUB64 / "pred" / "v2long_s16000_g3" / "videos"
    (OUT / "probe.json").write_text(json.dumps(
        {p.stem: psnr(read_mp4(p), read_mp4(ref / p.name)) for p in sorted(probe.glob("*.mp4"))}, indent=1))

    # R1: simulated offset, KEEP vs AUTO; AUTO only differs on routed windows, so reuse KEEP elsewhere
    keep_raw, auto_raw = SIM / "pred" / "keep" / "videos", SIM / "pred" / "auto" / "videos"
    gen(SIM, keep_raw, "keep")
    routed = route_table(SIM, load_envelope())
    (OUT / "sim_routed.json").write_text(json.dumps({"n": len(routed), "routed": routed}, indent=1))
    auto_raw.mkdir(parents=True, exist_ok=True)
    for p in keep_raw.glob("*.mp4"):
        if p.stem not in routed and not (auto_raw / p.name).exists():
            shutil.copy2(p, auto_raw / p.name)
    gen(SIM, auto_raw, "auto")

    # R2: real holdout windows the rule routes
    route_raw = ROUTE / "pred" / "auto" / "videos"
    gen(ROUTE, route_raw, "auto")

    full16_anchor = FULL / "pred" / "v2long_s16000_g3_anchor_t20" / "videos"
    anchor(SIM, keep_raw, SIM / "pred" / "keep_anchor_t20" / "videos")
    anchor(SIM, auto_raw, SIM / "pred" / "auto_anchor_t20" / "videos")
    anchor(ROUTE, route_raw, ROUTE / "pred" / "auto_anchor_t20" / "videos")
    anchor(FULL, FULL16 / "videos", full16_anchor)

    score(SIM / "pred" / "keep_anchor_t20" / "videos", SIM, OUT / "sim_keep")
    score(SIM / "pred" / "auto_anchor_t20" / "videos", SIM, OUT / "sim_auto")
    score(ROUTE / "pred" / "auto_anchor_t20" / "videos", ROUTE, OUT / "route_auto")
    score(full16_anchor, ROUTE, OUT / "route_keep")
    # diagnostics: the offset runs scored against the true (un-offset) targets, and the 16k baseline on all 192 windows
    score(SIM / "pred" / "keep_anchor_t20" / "videos", SUB64, OUT / "simclean_keep")
    score(SIM / "pred" / "auto_anchor_t20" / "videos", SUB64, OUT / "simclean_auto")
    score(SUB64 / "pred" / "v2long_s16000_g3_anchor_t20" / "videos", SUB64, OUT / "sub64_16k_t20")
    score(full16_anchor, FULL, SH / "wmscore" / "full192_v2long_s16000_g3_anchor_t20")

    run("wmgen.c2_gate", "--dir", str(OUT))


if __name__ == "__main__":
    main()

"""GPU job for submission 6: gate (wmgen.s6_gate), then, only on PASS, the eval build. Resumable: finished steps are skipped.

Needs `python -m wmgen.s6_prepare` first (configs/s6_band_design.json committed, WM_Shared roots built).
Every output uses new s6 names so nothing from the sub5 era can be reused by the skip logic.

python -m wmgen.s6_run
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


from wmgen.c2_run import ADAPTER, CREATE_NO_WINDOW, GUIDANCE, IDMS, THRESHOLD, psnr, run
from wmgen.offset_route import REPO
from wmgen.s6_prepare import BAND_ROOT, DESIGN, H003_ROOT, P1_ROOT
from wmgen.video_io import read_mp4

SH = Path(r"C:\Dacon\WM_Shared")
OUT = SH / "s6_gates"
EVAL = Path(r"C:\Dacon\RobotWorldModel_ActionVideo\open\data\eval")
SUB5, SUB6 = SH / "submissions" / "sub5", SH / "submissions" / "sub6"
SUB64_SCORE = SH / "wmscore" / "sub64_v2long_s16000_g3_anchor_t20_idm_v2.json"
EXPECTED_CHANGED = 33


def n_mp4(d: Path) -> int:
    return len(list(d.glob("*.mp4"))) if d.exists() else 0


def gen(root: Path, out: Path, mode: str, limit: int = 0) -> None:
    n = limit or len(list((root / "images").glob("*.png")))
    if n_mp4(out) >= n:
        return
    run("wmgen.gen_cosmos_ac", "--adapter", str(ADAPTER), "--eval-root", str(root), "--out", str(out),
        "--guidance", GUIDANCE, "--abs-mode", mode, *(["--limit", str(limit)] if limit else []))


def anchor(root: Path, raw: Path, out: Path) -> None:
    if n_mp4(out) >= len(list((root / "images").glob("*.png"))):
        return
    run("wmgen.apply_anchor", "--pred", str(raw), "--eval-root", str(root), "--out", str(out), "--threshold", THRESHOLD)


def score(pred: Path, holdout: Path, name: str) -> None:
    for idm_name, idm in IDMS.items():
        dst = OUT / f"{name}_{idm_name}.json"
        if not dst.exists():
            run("wmscore.score", "--pred", str(pred), "--holdout", str(holdout), "--idm", str(idm), "--out", str(dst))


def gate_stage() -> bool:
    OUT.mkdir(parents=True, exist_ok=True)
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(DESIGN.relative_to(REPO))], cwd=REPO,
                             capture_output=True, stdin=subprocess.DEVNULL, creationflags=CREATE_NO_WINDOW)
    assert tracked.returncode == 0, "configs/s6_band_design.json must be committed before the gate run"
    for p in (BAND_ROOT, P1_ROOT, H003_ROOT):
        assert (p / "actions").exists(), f"missing preparation: {p}"

    auto_raw, lift_raw = BAND_ROOT / "pred" / "s6_auto" / "videos", BAND_ROOT / "pred" / "s6_liftlo" / "videos"
    h_raw = H003_ROOT / "pred" / "s6_liftlo" / "videos"
    gen(BAND_ROOT, auto_raw, "auto")
    routed = json.loads((auto_raw.parent / "gen_stats.json").read_text()).get("routed", {})
    assert not routed, f"the sub5 rule routed band windows: {list(routed)[:3]}"
    gen(BAND_ROOT, lift_raw, "liftlo")
    gen(H003_ROOT, h_raw, "liftlo")

    auto_a = BAND_ROOT / "pred" / "s6_auto_anchor_t20" / "videos"
    lift_a = BAND_ROOT / "pred" / "s6_liftlo_anchor_t20" / "videos"
    h_a = H003_ROOT / "pred" / "s6_liftlo_anchor_t20" / "videos"
    anchor(BAND_ROOT, auto_raw, auto_a)
    anchor(BAND_ROOT, lift_raw, lift_a)
    anchor(H003_ROOT, h_raw, h_a)

    score(auto_a, BAND_ROOT, "band_auto")
    score(lift_a, BAND_ROOT, "band_liftlo")
    score(lift_a, P1_ROOT, "p1clean_liftlo")  # P1 inputs to LIFTLO equal the clean ones; the scorer reads only P1 ids
    score(h_a, H003_ROOT, "h003_liftlo")
    run("wmgen.s6_gate", "--dir", str(OUT))
    return bool(json.loads((OUT / "verdict.json").read_text())["PASS"])


def eval_stage() -> None:
    probe = SUB6 / "probe" / "videos"
    gen(EVAL, probe, "liftlo", limit=4)  # scene0 clips, unchanged by the s6 rule: must equal sub5 exactly
    vals = {p.stem: psnr(read_mp4(p), read_mp4(SUB5 / "raw" / "videos" / p.name)) for p in sorted(probe.glob("*.mp4"))}
    (SUB6 / "probe.json").write_text(json.dumps(vals, indent=1))
    assert vals and min(vals.values()) >= 99.0, f"determinism probe failed: {vals}"

    raw = SUB6 / "raw" / "videos"
    if not (SUB6 / "raw" / "changed.json").exists():
        run("wmgen.offset_route", "copy-unchanged", "--envelope", "s6", "--root", str(EVAL), "--src", str(SUB5 / "raw" / "videos"),
            "--dst", str(raw), "--base-route", str(SUB5 / "route.json"))
    changed = json.loads((SUB6 / "raw" / "changed.json").read_text())
    assert changed["n"] == EXPECTED_CHANGED, f"expected {EXPECTED_CHANGED} changed clips, got {changed['n']}"
    assert not [i for i in changed["changed"] if int(i.split("_")[1]) <= 153], "a scene0 clip changed"
    run("wmgen.offset_route", "list", "--envelope", "s6", "--root", str(EVAL), "--out", str(SUB6 / "route.json"))
    run("wmgen.submit_best", "--out", str(SUB6), "--abs-mode", "liftlo", "--guidance", GUIDANCE, "--threshold", THRESHOLD,
        "--candidate", f"v2long_s16000={ADAPTER}={SUB64_SCORE}")


def main() -> None:
    if not gate_stage():
        print("s6 gate FAILED: no eval build, submission 5 stays the base", flush=True)
        return
    eval_stage()


if __name__ == "__main__":
    main()

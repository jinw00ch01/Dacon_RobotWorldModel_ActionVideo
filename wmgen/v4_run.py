"""GPU job for submission 7: v4_aug soup on the clean holdout (G1), then the s6 band (G2), then, only on PASS, the eval build
with submission 5's pipeline. Resumable: finished steps are skipped.

python -m wmgen.v4_run
"""
from __future__ import annotations

import json
from pathlib import Path

from wmgen.c2_run import GUIDANCE, IDMS, THRESHOLD, run
from wmgen.s6_prepare import BAND_ROOT

SH = Path(r"C:\Dacon\WM_Shared")
SOUP = SH / "cosmos_ac" / "soups" / "soup_mean_v4aug_22500-24000.pt"
FULL = SH / "holdout_v1"
OUT = SH / "v4_gates"
SUB7 = SH / "submissions" / "sub7"


def n_mp4(d: Path) -> int:
    return len(list(d.glob("*.mp4"))) if d.exists() else 0


def gen_anchor_score(root: Path, name: str) -> None:
    raw = root / "pred" / "v4soup_auto" / "videos"
    anchored = root / "pred" / "v4soup_auto_anchor_t20" / "videos"
    n = len(list((root / "images").glob("*.png")))
    if n_mp4(raw) < n:
        run("wmgen.gen_cosmos_ac", "--adapter", str(SOUP), "--eval-root", str(root), "--out", str(raw),
            "--guidance", GUIDANCE, "--abs-mode", "auto")
    if n_mp4(anchored) < n:
        run("wmgen.apply_anchor", "--pred", str(raw), "--eval-root", str(root), "--out", str(anchored), "--threshold", THRESHOLD)
    for idm_name, idm in IDMS.items():
        dst = OUT / f"{name}_{idm_name}.json"
        if not dst.exists():
            run("wmscore.score", "--pred", str(anchored), "--holdout", str(root), "--idm", str(idm), "--out", str(dst))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    assert SOUP.exists(), SOUP
    gen_anchor_score(FULL, "full_v4")
    run("wmgen.v4_gate", "--dir", str(OUT))
    if not json.loads((OUT / "verdict.json").read_text())["G1"]["pass"]:
        print("v4 gate G1 FAILED: no band run, no eval build", flush=True)
        return
    gen_anchor_score(BAND_ROOT, "band_v4")
    run("wmgen.v4_gate", "--dir", str(OUT))
    if not json.loads((OUT / "verdict.json").read_text())["PASS"]:
        print("v4 gate FAILED: no eval build, submission 5 stays the base", flush=True)
        return
    run("wmgen.submit_best", "--out", str(SUB7), "--abs-mode", "auto", "--guidance", GUIDANCE, "--threshold", THRESHOLD,
        "--candidate", f"v4aug_soup={SOUP}={OUT / 'full_v4_idm_v2.json'}")


if __name__ == "__main__":
    main()

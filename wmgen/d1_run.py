"""GPU job for diagnostic D1 (wmgen.d1_prepare, wmgen.d1_report): the 16k, 22k and v4_aug-soup models on holdout_v1_sub64
with every window's joints read through another training robot's zero, plus the same models' clean sub64 videos rescored.

All three use one inference setting (g3, 30 steps, seed 0, anchor t20) and --abs-mode keep, so only the weights differ.
Scores are taken against the true (unshifted) targets and the IDM predictions are saved, so wmgen.d1_report can also score
against the shifted targets. Resumable: finished steps are skipped. Diagnosis only: it selects no submission.

python -m wmgen.d1_run
"""
from __future__ import annotations

import json
from pathlib import Path

from wmgen.c2_run import GUIDANCE, IDMS, THRESHOLD, run

SH = Path(r"C:\Dacon\WM_Shared")
REPO = Path(__file__).resolve().parents[1]
SUB64 = SH / "holdout_v1_sub64"
SIM = SH / "holdout_v1_sub64_d1"
OUT = SH / "d1_shift"
MODELS = {  # name: (weights, existing clean sub64 videos with the same inference setting)
    "16k": (SH / "cosmos_ac" / "runs" / "v2_long" / "adapter_016000.pt", SUB64 / "pred" / "v2long_s16000_g3_anchor_t20"),
    "v4": (SH / "cosmos_ac" / "soups" / "soup_mean_v4aug_22500-24000.pt", SH / "holdout_v1" / "pred" / "v4soup_auto_anchor_t20"),
    "22k": (SH / "cosmos_ac" / "runs" / "v2_long2" / "adapter_022000.pt", SUB64 / "pred" / "v2long2_s22000_g3_anchor_t20"),
}


def n_mp4(d: Path) -> int:
    return len(list(d.glob("*.mp4"))) if d.exists() else 0


def score(pred: Path, name: str) -> None:
    for idm_name, idm in IDMS.items():
        dst = OUT / f"{name}_{idm_name}.json"
        if not dst.exists():
            run("wmscore.score", "--pred", str(pred), "--holdout", str(SUB64), "--idm", str(idm), "--out", str(dst), "--save-pred")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    n = n_mp4(SUB64 / "gt_videos")
    if n_mp4(SIM / "gt_videos") < n:
        run("wmgen.make_sim_holdout", "--src", str(SUB64), "--out", str(SIM), "--shifts", str(REPO / "configs" / "d1_shifts.json"))
    for name, (weights, clean) in MODELS.items():
        assert weights.exists(), weights
        raw, anchored = OUT / "pred" / f"{name}_keep" / "videos", OUT / "pred" / f"{name}_keep_anchor_t20" / "videos"
        if n_mp4(raw) < n:
            run("wmgen.gen_cosmos_ac", "--adapter", str(weights), "--eval-root", str(SIM), "--out", str(raw),
                "--guidance", GUIDANCE, "--abs-mode", "keep")
        if n_mp4(anchored) < n:
            run("wmgen.apply_anchor", "--pred", str(raw), "--eval-root", str(SIM), "--out", str(anchored), "--threshold", THRESHOLD)
        score(anchored, f"shift_{name}")
        score(clean / "videos", f"clean_{name}")
    run("wmgen.d1_report", "--dir", str(OUT))
    print(json.loads((OUT / "verdict.json").read_text())["reading"], flush=True)


if __name__ == "__main__":
    main()

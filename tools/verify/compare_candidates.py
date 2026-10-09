"""Compare two holdout candidates the way docs/VERIFY_SUB4_ANALYSIS.md proposes.

Reads per-sample CSVs for both IDMs (verify_scores/<prefix>_<name>_idm_v1_cpu.csv and _idm_v2_cpu.csv; falls back to
wmscore/<prefix>_<name>_idm_v2.csv) and reports paired differences (A - B, lower is better) with clip-level and
uploader-cluster bootstrap intervals, plus eval-like windows only. Verdict rule:
  better: DINO not significantly worse, and either both IDMs' totals have uploader CI below 0, or DINO is
          significantly better while no IDM's total is significantly worse and any significant action loss is
          smaller than the DINO gain (the board moves with DINO far more than with our action term).
  worse:  the mirror image.  Otherwise: tie (keep the current submission).

  python tools/verify/compare_candidates.py --a v2long2_s24000_g3_anchor_t20 --b v2long_s16000_g3_anchor_t20
"""
import argparse, json, os

import numpy as np, pandas as pd

SHARED = r"C:\Dacon\WM_Shared"
EVAL_LIKE = {"scene0-like": ["bensprenger", "frk2"], "scene1-like": ["DorayakiLin", "sixpigs1"]}
COMPS = ["score", "dino", "r3d", "action"]


def sources(name, idm, prefix):
    return [os.path.join(SHARED, "wmscore", f"{prefix}_{name}_{idm}.csv"),
            os.path.join(SHARED, "verify_scores", f"{prefix}_{name}_{idm}_cpu.csv")]


def load_pair(a, b, idm, prefix):
    """Both candidates from the same scorer code path (the paths differ by ~0.001 in DINO)."""
    for pa, pb in zip(sources(a, idm, prefix), sources(b, idm, prefix)):
        if os.path.exists(pa) and os.path.exists(pb):
            out = []
            for p in (pa, pb):
                d = pd.read_csv(p).set_index("sample_id")[["dino", "r3d", "action"]]
                d["score"] = 0.3 * d.dino + 0.3 * d.r3d + 0.4 * d.action
                out.append(d)
            return out[0], out[1], os.path.dirname(pa)
    return None, None, None


def boot(diff, users, rng, n):
    v = diff.values
    clip = v[rng.integers(0, len(v), (n, len(v)))].mean(1)
    groups = [v[(users == u).values] for u in users.unique()]
    clus = np.array([np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]).mean() for _ in range(n)])
    return v.mean(), np.percentile(clip, [2.5, 97.5]), np.percentile(clus, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--prefix", default="sub64")
    ap.add_argument("--windows", default=os.path.join(SHARED, "holdout_v1_sub64", "windows.csv"))
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--ids", default=None, help="json file with the sample ids to compare (list, or dict with an id list)")
    args = ap.parse_args()
    users = pd.read_csv(args.windows).set_index("sample_id")["user"]
    rng = np.random.default_rng(0)
    keep = None
    if args.ids:
        j = json.load(open(args.ids, encoding="utf-8"))
        if isinstance(j, dict):  # e.g. {"n": 44, "routed": {"hold_000000": [1, 2], ...}}
            j = next(v for v in j.values() if isinstance(v, (list, dict)) and v)
            j = list(j) if isinstance(j, dict) else j
        keep = set(x if isinstance(x, str) else x.get("sample_id") for x in j)
    res = {}
    for idm in ("idm_v1", "idm_v2"):
        a, b, src = load_pair(args.a, args.b, idm, args.prefix)
        if a is None:
            print(f"{idm}: no scorer output with both candidates")
            continue
        ids = a.index.intersection(b.index)
        if keep is not None:
            ids = ids[ids.isin(keep)]
        u = users.loc[ids]
        print(f"\n{idm} [{os.path.basename(src)}]  A={args.a}  B={args.b}  n={len(ids)}   (A - B, lower is better)")
        res[idm] = {}
        for c in COMPS:
            m, ci, cl = boot(a.loc[ids, c] - b.loc[ids, c], u, rng, args.n)
            res[idm][c] = (m, cl)
            print(f"  {c:6s} {m:+.4f}  clip [{ci[0]:+.4f},{ci[1]:+.4f}]  uploader [{cl[0]:+.4f},{cl[1]:+.4f}]")
        for k, us in EVAL_LIKE.items():
            m = u.isin(us)
            d = (a.loc[ids][m] - b.loc[ids][m]).mean()
            print(f"  {k:12s} n={m.sum():2d} " + " ".join(f"{c} {d[c]:+.4f}" for c in COMPS))
    if len(res) < 2:
        print("\nverdict: need both IDMs")
        return
    sig_lo = lambda idm, c: res[idm][c][1][1] < 0  # significantly better (uploader CI below 0)
    sig_hi = lambda idm, c: res[idm][c][1][0] > 0  # significantly worse
    idms = list(res)

    def wins(lo, hi):
        if any(hi(i, "dino") for i in idms):
            return False
        if all(lo(i, "score") for i in idms):
            return True
        # DINO gain counts when no total is significantly worse and the action loss is smaller than the DINO gain
        return (all(lo(i, "dino") for i in idms) and not any(hi(i, "score") for i in idms)
                and all(abs(res[i]["action"][0]) < abs(res[i]["dino"][0]) or not hi(i, "action") for i in idms))

    verdict = "A better" if wins(sig_lo, sig_hi) else "B better" if wins(sig_hi, sig_lo) else "tie (keep current)"
    print(f"\nverdict: {verdict}")


if __name__ == "__main__":
    main()

"""Does a generated video move the arm the way the actions say? Per-joint check with an IDM.

For each sample: IDM(video) -> 16x6 z-actions; compare the change from frame 0 (a_t - a_0) with the target's
change. Reports per-joint correlation of the deltas over all samples and timesteps, the share of samples whose
end-point delta has the same sign as the target's, and the delta magnitude ratio. A joint with negative
correlation suggests a sign or ordering problem in the action conditioning.

  python tools/verify/action_direction.py --pred <dir of mp4> [--pred ...] --holdout C:/Dacon/WM_Shared/holdout_v1_sub64
"""
import argparse, glob, json, os, sys

import numpy as np, torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from tools.verify.feature_scores import IDM_H, IDM_W, letterbox, read_video  # noqa: E402

JOINTS = ["pan", "lift", "elbow", "wrist_flex", "wrist_roll", "gripper"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", action="append", required=True)
    ap.add_argument("--holdout", default=r"C:\Dacon\WM_Shared\holdout_v1_sub64")
    ap.add_argument("--idm", default=r"C:\Dacon\WM_Shared\idm\idm_v2.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    from wmscore.idm import load_idm
    idm = load_idm(args.idm, args.device)
    repo = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    st = json.load(open(os.path.join(repo, "open", "data", "train", "so100_action_statistics.json"), encoding="utf-8"))
    mu, sd = np.array(st["mean"]), np.array(st["std"])
    ids = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(args.holdout, "images", "*.png")))
    tgt = np.stack([(np.load(os.path.join(args.holdout, "actions", i + ".npy")) - mu) / sd for i in ids])
    dt = tgt - tgt[:, :1]
    report = {}
    for pd_ in args.pred:
        est = []
        for i in ids:
            fr = read_video(os.path.join(pd_, i + ".mp4"))
            x = np.stack([letterbox(f, IDM_H, IDM_W) for f in fr]).astype(np.float32) / 255
            with torch.no_grad():
                est.append(idm(torch.from_numpy(x).permute(0, 3, 1, 2)[None].to(args.device))[0].cpu().numpy())
        est = np.stack(est)
        de = est - est[:, :1]
        r = {}
        for j, name in enumerate(JOINTS):
            a, b = de[:, 1:, j].ravel(), dt[:, 1:, j].ravel()
            big = np.abs(dt[:, -1, j]) > 0.1
            r[name] = dict(corr=round(float(np.corrcoef(a, b)[0, 1]), 3),
                           end_sign_agree=round(float((np.sign(de[big, -1, j]) == np.sign(dt[big, -1, j])).mean()), 3) if big.any() else None,
                           n_moving=int(big.sum()),
                           mag_ratio=round(float(np.median(np.abs(de[:, -1, j])) / (np.median(np.abs(dt[:, -1, j])) + 1e-9)), 3))
        r["mae"] = round(float(np.abs(est - tgt).mean()), 4)
        report[pd_] = r
        print(pd_, json.dumps(r))
    if args.out:
        json.dump(report, open(args.out, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()

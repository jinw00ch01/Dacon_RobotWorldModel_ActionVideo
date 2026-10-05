"""Make the dataset(uploader)-level holdout split and a fixed list of validation windows.

Inputs: data index from build_index.py (--index). Outputs:
  configs/splits/holdout_v1.json            train/val dataset lists + rationale
  configs/splits/holdout_v1_val_windows.csv fixed 16-frame validation windows (seeded)
  <index>/holdout_v1_val_grid.jpg           first frames of the val datasets
"""
import argparse, json, os

import cv2, numpy as np, polars as pl

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
ROOT = os.path.join(REPO, "open", "data", "train")
WIN = 16

# Whole uploaders go to val so no environment leaks into train. Chosen by eye from train_grid.jpg against the two
# eval scene types (eval images were only looked at; nothing numeric from eval is used).
VAL_USERS = {
    "bensprenger": "gray floor, top-down camera, two arms (green/blue) - closest to eval scene 0",
    "frk2": "top-down camera over floor, black arm",
    "DorayakiLin": "wooden desk, black arm - like eval scene 1",
    "sixpigs1": "wooden desk with gray mat, black arm - like eval scene 1",
    "shreyasgite": "red arm, different robot color",
    "aimihat": "orange arm on wooden table, different robot color",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=r"C:\Dacon\WM_Shared\data_index")
    ap.add_argument("--per-dataset", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-motion", type=float, default=0.05, help="max_t mean|z_t-z_0| a window must reach")
    args = ap.parse_args()
    ds = pl.read_csv(os.path.join(args.index, "datasets.csv"))
    missing = set(VAL_USERS) - set(ds["user"].to_list())
    assert not missing, missing
    val = ds.filter(pl.col("user").is_in(list(VAL_USERS)))
    train = ds.filter(~pl.col("user").is_in(list(VAL_USERS)))
    stats = json.load(open(os.path.join(ROOT, "so100_action_statistics.json"), encoding="utf-8"))
    mu, sd = np.array(stats["mean"]), np.array(stats["std"])

    rng = np.random.default_rng(args.seed)
    eps = pl.read_parquet(os.path.join(args.index, "episodes.parquet"))
    wins = []
    for r in val.iter_rows(named=True):
        e = eps.filter((pl.col("user") == r["user"]) & (pl.col("dataset") == r["dataset"]) & (pl.col("rows") >= WIN))
        cands = []
        for er in e.iter_rows(named=True):
            a = np.stack(pl.read_parquet(os.path.join(ROOT, er["parquet"]), columns=["action"])["action"].to_numpy())
            z = (a - mu) / sd
            for s in range(0, len(z) - WIN + 1, 4):
                m = np.abs(z[s:s + WIN] - z[s]).mean(1).max()
                if m >= args.min_motion:
                    cands.append((er["episode_index"], s, float(m), er["parquet"], er["video"]))
        pick = rng.choice(len(cands), size=min(args.per_dataset, len(cands)), replace=False)
        for i in sorted(pick, key=lambda i: cands[i][:2]):
            ep, s, m, pq, vid = cands[i]
            wins.append(dict(user=r["user"], dataset=r["dataset"], episode_index=ep, start_frame=s, num_frames=WIN,
                             max_motion_z=round(m, 4), parquet=pq, video=vid))

    out_dir = os.path.join(REPO, "configs", "splits")
    os.makedirs(out_dir, exist_ok=True)
    frac = val["frames"].sum() / ds["frames"].sum()
    manifest = dict(
        name="holdout_v1", unit="uploader (all datasets of a val uploader are held out)", seed=args.seed,
        rule="Train only on train datasets. Val is for model/checkpoint selection with our own scorer (wmscore).",
        criteria="Picked by eye from first frames: environments like the two eval scene types (gray top-down floor, "
                 "wooden desk + black arm) plus robot colors other than the common white/orange. Eval images were "
                 "only viewed, no numeric eval signal was used.",
        val_users=VAL_USERS,
        val_datasets=[f"{u}/{d}" for u, d in val.select("user", "dataset").iter_rows()],
        train_datasets=[f"{u}/{d}" for u, d in train.select("user", "dataset").iter_rows()],
        counts=dict(val_datasets=val.height, train_datasets=train.height, val_users=len(VAL_USERS),
                    train_users=train["user"].n_unique(), val_frames=int(val["frames"].sum()),
                    train_frames=int(train["frames"].sum()), val_frame_frac=round(float(frac), 4),
                    val_episodes=int(val["episodes"].sum()), val_windows=len(wins)),
        val_windows=dict(file="holdout_v1_val_windows.csv", per_dataset=args.per_dataset, stride=4,
                         min_motion_z=args.min_motion, note="frame 0 of a window = conditioning image; actions[s:s+16]"),
        notes=["bensprenger/chess_game_001_blue_stereo shares the chess setup seen in Chojins (train), so it is less unseen than the rest."],
        index=args.index,
    )
    json.dump(manifest, open(os.path.join(out_dir, "holdout_v1.json"), "w", encoding="utf-8"), indent=2)
    pl.DataFrame(wins).write_csv(os.path.join(out_dir, "holdout_v1_val_windows.csv"))

    ims = []
    for r in val.iter_rows(named=True):
        t = cv2.resize(cv2.imread(os.path.join(args.index, "thumbs", f"{r['ds_idx']:03d}.jpg")), (320, 240))
        cv2.rectangle(t, (0, 0), (320, 18), (0, 0, 0), -1)
        cv2.putText(t, f"{r['user']}/{r['dataset']}"[:42], (2, 13), 0, 0.42, (0, 255, 255), 1)
        ims.append(t)
    while len(ims) % 4:
        ims.append(np.zeros_like(ims[0]))
    cv2.imwrite(os.path.join(args.index, "holdout_v1_val_grid.jpg"),
                np.vstack([np.hstack(ims[i:i + 4]) for i in range(0, len(ims), 4)]))
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()

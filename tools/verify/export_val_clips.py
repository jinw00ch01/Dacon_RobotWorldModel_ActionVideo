"""Export the holdout_v1 validation windows as eval-shaped samples.

For each window: <out>/images/val_XXXXXX.png (frame 0, 640x480), actions/val_XXXXXX.npy (16x6 float32),
gt_videos/val_XXXXXX.mp4 (16 frames, 640x480), plus index.csv mapping ids to windows.
"""
import argparse, csv, os

import av, cv2, numpy as np, polars as pl

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
ROOT = os.path.join(REPO, "open", "data", "train")


def read_frames(path, start, n):
    out = []
    with av.open(path) as c:
        for i, f in enumerate(c.decode(video=0)):
            if i >= start:
                out.append(f.to_ndarray(format="rgb24"))
            if len(out) == n:
                break
    return out


def write_mp4(path, frames, crf=10):
    with av.open(path, "w") as c:
        s = c.add_stream("libx264", rate=6)
        s.width, s.height, s.pix_fmt = frames[0].shape[1], frames[0].shape[0], "yuv420p"
        s.options = {"crf": str(crf)}
        for fr in frames:
            c.mux(s.encode(av.VideoFrame.from_ndarray(fr, format="rgb24")))
        c.mux(s.encode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default=os.path.join(REPO, "configs", "splits", "holdout_v1_val_windows.csv"))
    ap.add_argument("--out", default=r"C:\Dacon\WM_Shared\holdout_v1_clips")
    args = ap.parse_args()
    for d in ("images", "actions", "gt_videos"):
        os.makedirs(os.path.join(args.out, d), exist_ok=True)
    rows = list(csv.DictReader(open(args.windows, encoding="utf-8")))
    idx = []
    for i, r in enumerate(rows):
        sid = f"val_{i:06d}"
        s = int(r["start_frame"])
        a = np.stack(pl.read_parquet(os.path.join(ROOT, r["parquet"]), columns=["action"])["action"].to_numpy())
        np.save(os.path.join(args.out, "actions", sid + ".npy"), a[s:s + 16].astype(np.float32))
        frames = [cv2.resize(f, (640, 480), interpolation=cv2.INTER_AREA) for f in read_frames(os.path.join(ROOT, r["video"]), s, 16)]
        assert len(frames) == 16, (sid, r)
        cv2.imwrite(os.path.join(args.out, "images", sid + ".png"), cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR))
        write_mp4(os.path.join(args.out, "gt_videos", sid + ".mp4"), frames)
        idx.append(dict(sample_id=sid, **{k: r[k] for k in ("user", "dataset", "episode_index", "start_frame")}))
    with open(os.path.join(args.out, "index.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(idx[0]))
        w.writeheader()
        w.writerows(idx)
    print(len(idx), "clips ->", args.out)


if __name__ == "__main__":
    main()

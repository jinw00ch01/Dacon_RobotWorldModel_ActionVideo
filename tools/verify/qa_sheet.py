"""QA for generated videos: checks plus contact sheets.

Per sample: frame count (must be 16), frame 0 vs conditioning image (PSNR), share of pixels that move,
background drift (mean abs change of pixels outside the moving region), and the direction the moving
region's centroid travels. Sheets: one row per sample = conditioning image + 16 frames.

  python tools/verify/qa_sheet.py --pred <dir of mp4> --images <dir of png> --out <dir>
"""
import argparse, csv, glob, os

import av, cv2, numpy as np

TW, TH = 128, 96


def read_video(path):
    with av.open(path) as c:
        return [f.to_ndarray(format="bgr24") for f in c.decode(video=0)]


def psnr(a, b):
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    return 99.0 if mse == 0 else float(10 * np.log10(255 ** 2 / mse))


def check(frames, cond):
    r = dict(n_frames=len(frames))
    if cond is not None:
        h, w = frames[0].shape[:2]
        ch, cw = cond.shape[:2]
        if abs(h / w - ch / cw) > 0.01:  # e.g. 512x320 letterboxed output vs 640x480 input: letterbox the input
            s = min(h / ch, w / cw)
            nh, nw = round(ch * s), round(cw * s)
            ref = np.zeros_like(frames[0])
            y, x = (h - nh) // 2, (w - nw) // 2
            ref[y:y + nh, x:x + nw] = cv2.resize(cond, (nw, nh), interpolation=cv2.INTER_AREA)
            r["frame0_psnr"] = round(psnr(frames[0], ref), 2)
        else:
            f0 = cv2.resize(frames[0], (cw, ch), interpolation=cv2.INTER_AREA)
            r["frame0_psnr"] = round(psnr(f0, cond), 2)
    g = np.stack([cv2.cvtColor(cv2.resize(f, (320, 240), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY) for f in frames]).astype(np.float32)
    diff = np.abs(g - g[0])  # T,H,W
    moving = cv2.dilate((diff.max(0) > 25).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    r["moving_frac"] = round(float(moving.mean()), 4)
    r["bg_drift"] = round(float(diff[:, ~moving].mean()) if (~moving).any() else 0.0, 3)
    r["temporal_flicker"] = round(float(np.abs(np.diff(g, axis=0))[:, ~moving].mean()) if (~moving).any() else 0.0, 3)
    cents = []
    for t in range(1, len(g)):
        m = diff[t] > 25
        if m.sum() > 50:
            ys, xs = np.nonzero(m)
            cents.append((t, xs.mean(), ys.mean()))
    if len(cents) >= 2:
        r["motion_dx"] = round(cents[-1][1] - cents[0][1], 1)
        r["motion_dy"] = round(cents[-1][2] - cents[0][2], 1)
    r["static_video"] = bool(diff.max() < 5)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--images", default=r"C:\Dacon\WM_Shared\holdout_v1_clips\images")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-sheet", type=int, default=24)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows, strips = [], []
    for vp in sorted(glob.glob(os.path.join(args.pred, "*.mp4"))):
        sid = os.path.splitext(os.path.basename(vp))[0]
        frames = read_video(vp)
        ip = os.path.join(args.images, sid + ".png")
        cond = cv2.imread(ip) if os.path.exists(ip) else None
        r = dict(sample_id=sid, **check(frames, cond))
        rows.append(r)
        cells = [cv2.resize(cond if cond is not None else np.zeros_like(frames[0]), (TW, TH))]
        cells += [cv2.resize(f, (TW, TH)) for f in frames[:16]]
        cells += [np.zeros((TH, TW, 3), np.uint8)] * (17 - len(cells))
        strip = np.hstack(cells)
        bad = r["n_frames"] != 16 or r.get("frame0_psnr", 99) < 25
        cv2.rectangle(strip, (0, 0), (TW * 17 - 1, 13), (0, 0, 160) if bad else (0, 0, 0), -1)
        cv2.putText(strip, f"{sid} n={r['n_frames']} f0psnr={r.get('frame0_psnr', '-')} move={r['moving_frac']} "
                           f"bg={r['bg_drift']} d=({r.get('motion_dx', '-')},{r.get('motion_dy', '-')})",
                    (2, 10), 0, 0.35, (255, 255, 255), 1)
        strips.append(strip)
    for i in range(0, len(strips), args.per_sheet):
        cv2.imwrite(os.path.join(args.out, f"sheet_{i // args.per_sheet:03d}.jpg"), np.vstack(strips[i:i + args.per_sheet]),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
    keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
    with open(os.path.join(args.out, "qa.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    n = len(rows)
    print(f"n={n} not16={sum(r['n_frames'] != 16 for r in rows)} "
          f"frame0_psnr<25={sum(r.get('frame0_psnr', 99) < 25 for r in rows)} static={sum(r['static_video'] for r in rows)} "
          f"bg_drift_mean={np.mean([r['bg_drift'] for r in rows]):.3f}")


if __name__ == "__main__":
    main()

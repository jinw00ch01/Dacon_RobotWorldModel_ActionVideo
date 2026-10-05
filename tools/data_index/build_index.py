"""Build the training data index (datasets, episodes, action stats, first-frame thumbnails).

Reads only open/data/train. Writes to --out (default C:\\Dacon\\WM_Shared\\data_index):
  datasets.csv, episodes.parquet, summary.json, thumbs/<idx>.jpg, train_grid.jpg, grid_index.csv
"""
import argparse, glob, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor

import av, cv2, numpy as np, polars as pl

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "open", "data", "train")
STATS = json.load(open(os.path.join(ROOT, "so100_action_statistics.json"), encoding="utf-8"))
MU, SD = np.array(STATS["mean"]), np.array(STATS["std"])
WIN = 16


def video_meta(path):
    with av.open(path) as c:
        s = c.streams.video[0]
        return dict(v_frames=int(s.frames or 0), v_codec=s.codec_context.name, v_w=s.width, v_h=s.height,
                    v_fps=float(s.average_rate or 0))


def first_frame(path):
    cap = cv2.VideoCapture(path)
    ok, fr = cap.read()
    cap.release()
    return fr if ok else None


def index_dataset(ds_dir):
    info = json.load(open(os.path.join(ds_dir, "meta", "info.json"), encoding="utf-8"))
    user, ds = os.path.basename(os.path.dirname(ds_dir)), os.path.basename(ds_dir)
    vkey = [k for k, v in info["features"].items() if v.get("dtype") == "video"][0]
    eps_meta = {}
    for line in open(os.path.join(ds_dir, "meta", "episodes.jsonl"), encoding="utf-8"):
        e = json.loads(line)
        eps_meta[e["episode_index"]] = e
    rows, acts = [], []
    for ei, e in sorted(eps_meta.items()):
        chunk = ei // info.get("chunks_size", 1000)
        pq = os.path.join(ds_dir, info["data_path"].format(episode_chunk=chunk, episode_index=ei))
        vp = os.path.join(ds_dir, info["video_path"].format(episode_chunk=chunk, video_key=vkey, episode_index=ei))
        r = dict(user=user, dataset=ds, episode_index=ei, meta_length=e["length"],
                 task=(e.get("tasks") or [""])[0], parquet=os.path.relpath(pq, ROOT).replace("\\", "/"),
                 video=os.path.relpath(vp, ROOT).replace("\\", "/"), parquet_ok=os.path.exists(pq),
                 video_ok=os.path.exists(vp))
        if r["parquet_ok"]:
            t = pl.read_parquet(pq, columns=["action", "observation.state", "frame_index"])
            a = np.stack(t["action"].to_numpy()).astype(np.float64)
            s = np.stack(t["observation.state"].to_numpy()).astype(np.float64)
            acts.append(a)
            z = (a - MU) / SD
            d = np.abs(np.diff(z, axis=0)).mean(1) if len(z) > 1 else np.zeros(1)
            r.update(rows=len(t), frame_index_contig=bool((np.diff(t["frame_index"].to_numpy()) == 1).all()),
                     z_mean=z.mean(0).round(4).tolist(), z_std=z.std(0).round(4).tolist(),
                     step_motion_z=float(d.mean()), static_frac=float((d < 0.01).mean()),
                     act_next_state_mae=float(np.abs(a[:-1] - s[1:]).mean()) if len(a) > 1 else None,
                     windows16=max(0, len(t) - WIN + 1), nan_any=bool(np.isnan(a).any() or np.isnan(s).any()))
        if r["video_ok"]:
            try:
                r.update(video_meta(vp))
            except Exception as ex:  # keep going, flag it
                r["video_error"] = str(ex)[:200]
        rows.append(r)
    A = np.concatenate(acts) if acts else np.zeros((0, 6))
    vi = info["features"][vkey].get("info", {})
    first = next((r for r in rows if r["video_ok"]), None)
    dsrow = dict(user=user, dataset=ds, path=os.path.relpath(ds_dir, ROOT).replace("\\", "/"),
                 robot_type=info.get("robot_type"), fps=info.get("fps"), original_fps=info.get("original_fps"),
                 downsample_stride=info.get("downsample_stride"), episodes=len(rows), frames=int(sum(r.get("rows", 0) for r in rows)),
                 meta_total_frames=info["total_frames"], codec=vi.get("video.codec"), width=vi.get("video.width"),
                 height=vi.get("video.height"), action_names=",".join(info["features"]["action"].get("names") or []),
                 n_tasks=len({r["task"] for r in rows}), task=rows[0]["task"] if rows else "",
                 len_min=int(min(r.get("rows", 0) for r in rows)), len_med=float(np.median([r.get("rows", 0) for r in rows])),
                 len_max=int(max(r.get("rows", 0) for r in rows)), windows16=int(sum(r.get("windows16", 0) for r in rows)),
                 act_mean=A.mean(0).round(3).tolist(), act_std=A.std(0).round(3).tolist(),
                 act_min=A.min(0).round(3).tolist(), act_max=A.max(0).round(3).tolist(),
                 z_mean=((A.mean(0) - MU) / SD).round(3).tolist(),
                 step_motion_z=float(np.mean([r["step_motion_z"] for r in rows if "step_motion_z" in r])),
                 video_frame_mismatch=int(sum(1 for r in rows if r.get("v_frames") and r.get("rows") and r["v_frames"] != r["rows"])),
                 missing_files=int(sum(1 for r in rows if not (r["parquet_ok"] and r["video_ok"]))),
                 first_video=first["video"] if first else None)
    return dsrow, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=r"C:\Dacon\WM_Shared\data_index")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    os.makedirs(os.path.join(args.out, "thumbs"), exist_ok=True)
    dss = sorted(os.path.dirname(m) for m in glob.glob(os.path.join(ROOT, "*", "*", "meta")))
    if args.limit:
        dss = dss[: args.limit]
    with ProcessPoolExecutor(args.workers) as ex:
        res = list(ex.map(index_dataset, dss))
    ds_rows = [r for r, _ in res]
    ep_rows = [e for _, eps in res for e in eps]
    for i, r in enumerate(ds_rows):
        r["ds_idx"] = i
    dsdf = pl.DataFrame(ds_rows, infer_schema_length=None)
    lists = [c for c, t in dsdf.schema.items() if isinstance(t, pl.List)]
    dsdf.with_columns([pl.col(c).list.eval(pl.element().cast(pl.Utf8)).list.join(" ") for c in lists]) \
        .select(["ds_idx"] + [c for c in dsdf.columns if c != "ds_idx"]).write_csv(os.path.join(args.out, "datasets.csv"))
    epdf = pl.DataFrame(ep_rows, infer_schema_length=None)
    epdf.write_parquet(os.path.join(args.out, "episodes.parquet"))

    # thumbnails + grid (first frame of first episode per dataset)
    ims, gidx = [], []
    for r in ds_rows:
        fr = first_frame(os.path.join(ROOT, r["first_video"])) if r["first_video"] else None
        fr = np.zeros((480, 640, 3), np.uint8) if fr is None else cv2.resize(fr, (640, 480))
        cv2.imwrite(os.path.join(args.out, "thumbs", f"{r['ds_idx']:03d}.jpg"), fr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        t = cv2.resize(fr, (240, 180))
        cv2.rectangle(t, (0, 0), (240, 16), (0, 0, 0), -1)
        cv2.putText(t, f"{r['ds_idx']} {r['user'][:22]}", (2, 12), 0, 0.4, (0, 255, 255), 1)
        ims.append(t)
        gidx.append(dict(ds_idx=r["ds_idx"], user=r["user"], dataset=r["dataset"]))
    while len(ims) % 12:
        ims.append(np.zeros_like(ims[0]))
    g = np.vstack([np.hstack(ims[i:i + 12]) for i in range(0, len(ims), 12)])
    cv2.imwrite(os.path.join(args.out, "train_grid.jpg"), g, [cv2.IMWRITE_JPEG_QUALITY, 88])
    pl.DataFrame(gidx).write_csv(os.path.join(args.out, "grid_index.csv"))

    lens = epdf["rows"].drop_nulls().to_numpy()
    summary = dict(datasets=len(ds_rows), users=dsdf["user"].n_unique(), episodes=len(ep_rows), frames=int(lens.sum()),
                   episodes_lt16=int((lens < WIN).sum()), windows16_stride1=int(dsdf["windows16"].sum()),
                   ep_len_pct={str(p): float(np.percentile(lens, p)) for p in (0, 5, 50, 95, 100)},
                   codecs=dict(dsdf["codec"].value_counts().iter_rows()),
                   resolutions={f"{w}x{h}": n for w, h, n in dsdf.group_by(["width", "height"]).len().iter_rows()},
                   fps=dict((str(k), v) for k, v in dsdf["fps"].value_counts().iter_rows()),
                   video_frame_mismatch_eps=int(sum(r["video_frame_mismatch"] for r in ds_rows)),
                   missing_files_eps=int(sum(r["missing_files"] for r in ds_rows)),
                   nan_eps=int(epdf["nan_any"].sum()), noncontig_eps=int((~epdf["frame_index_contig"]).sum()),
                   built_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), seconds=round(time.time() - t0, 1))
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    sys.exit(main())

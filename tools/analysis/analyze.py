import json, glob, os, collections
import numpy as np, polars as pl

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "open", "data")
rows = []
infos = sorted(glob.glob(os.path.join(ROOT, "train", "*", "*", "meta", "info.json")))
codecs = collections.Counter(); robots = collections.Counter(); fps = collections.Counter()
res = collections.Counter(); strides = collections.Counter(); ofps = collections.Counter()
for p in infos:
    ds = os.path.dirname(os.path.dirname(p))
    info = json.load(open(p, encoding="utf-8"))
    vkeys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
    for k in vkeys:
        vi = info["features"][k].get("info", {})
        codecs[vi.get("video.codec")] += 1
        res[(vi.get("video.width"), vi.get("video.height"))] += 1
    robots[info.get("robot_type")] += 1; fps[info.get("fps")] += 1
    strides[info.get("downsample_stride")] += 1; ofps[info.get("original_fps")] += 1
    tasks = set()
    lens = []
    for line in open(os.path.join(ds, "meta", "episodes.jsonl"), encoding="utf-8"):
        e = json.loads(line); lens.append(e["length"]); tasks.update(e.get("tasks", []))
    rows.append(dict(user=os.path.basename(os.path.dirname(ds)), ds=os.path.basename(ds),
                     eps=info["total_episodes"], frames=info["total_frames"], vkeys=len(vkeys),
                     act_shape=info["features"]["action"]["shape"][0],
                     names=",".join(info["features"]["action"].get("names") or []),
                     ntasks=len(tasks), task=list(tasks)[0][:60] if tasks else "",
                     len_min=min(lens), len_med=int(np.median(lens)), len_max=max(lens)))
df = pl.DataFrame(rows)
print("datasets", len(df), "users", df["user"].n_unique(), "episodes", df["eps"].sum(), "frames", df["frames"].sum())
print("codecs", codecs, "res", res, "robots", robots, "fps", fps, "orig_fps", ofps, "stride", strides)
print("video keys per ds", collections.Counter(df["vkeys"].to_list()), "act dims", collections.Counter(df["act_shape"].to_list()))
print("action name sets", collections.Counter(df["names"].to_list()).most_common(5))
allens = []
for p in infos:
    ds = os.path.dirname(os.path.dirname(p))
    for line in open(os.path.join(ds, "meta", "episodes.jsonl"), encoding="utf-8"):
        allens.append(json.loads(line)["length"])
allens = np.array(allens)
print("episode length pct", np.percentile(allens, [0, 5, 25, 50, 75, 95, 100]), "eps<17:", (allens < 17).sum())
print("16-frame windows (stride1):", int(np.clip(allens - 16, 0, None).sum()), "(stride4):", int((np.clip(allens - 16, 0, None) // 4).sum()))
with pl.Config(tbl_rows=200, tbl_cols=20, fmt_str_lengths=60, tbl_width_chars=250):
    print(df.sort("frames", descending=True).select(["user", "ds", "eps", "frames", "len_med", "ntasks", "task"]))
print("frames per user top10:")
print(df.group_by("user").agg(pl.col("frames").sum(), pl.col("eps").sum()).sort("frames", descending=True).head(10))

# parquet stats
acts, states, nxt = [], [], []
for p in infos:
    ds = os.path.dirname(os.path.dirname(p))
    fs = sorted(glob.glob(os.path.join(ds, "data", "*", "*.parquet")))
    for f in fs[:: max(1, len(fs) // 5)]:
        t = pl.read_parquet(f)
        a = np.stack(t["action"].to_numpy()); s = np.stack(t["observation.state"].to_numpy())
        acts.append(a); states.append(s)
        nxt.append((np.abs(a[:-1] - s[1:]).mean(0), np.abs(a - s).mean(0), np.abs(s[1:] - s[:-1]).mean(0)))
A = np.concatenate(acts); S = np.concatenate(states)
print("cols sample:", pl.read_parquet(fs[0]).columns)
np.set_printoptions(precision=2, suppress=True, linewidth=200)
print("train action min", A.min(0), "\n max", A.max(0), "\n mean", A.mean(0), "\n std", A.std(0))
n = np.array(nxt)
print("MAE action[t] vs state[t+1]", n[:, 0].mean(0), "\nMAE action[t] vs state[t]", n[:, 1].mean(0), "\nstate step delta", n[:, 2].mean(0))

E = np.stack([np.load(f) for f in sorted(glob.glob(os.path.join(ROOT, "eval", "actions", "*.npy")))])
print("eval actions", E.shape, E.dtype)
print("eval min", E.reshape(-1, 6).min(0), "\n max", E.reshape(-1, 6).max(0), "\n mean", E.reshape(-1, 6).mean(0), "\n std", E.reshape(-1, 6).std(0))
print("eval per-step delta mean", np.abs(np.diff(E, axis=1)).mean((0, 1)))
print("eval total motion per sample pct", np.percentile(np.abs(E[:, -1] - E[:, 0]).sum(1), [5, 25, 50, 75, 95]))
print("eval samples near-static (sum|a15-a0|<5):", int((np.abs(E[:, -1] - E[:, 0]).sum(1) < 5).sum()))
import cv2
v = glob.glob(os.path.join(ROOT, "train", "*", "*", "videos", "*", "*", "*.mp4"))[0]
cap = cv2.VideoCapture(v); ok, fr = cap.read()
print("cv2 decode", v[-60:], ok, None if fr is None else fr.shape, cap.get(cv2.CAP_PROP_FRAME_COUNT))
im = cv2.imread(os.path.join(ROOT, "eval", "images", "sample_000000.png")); print("eval img", im.shape)


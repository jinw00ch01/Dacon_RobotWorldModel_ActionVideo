import json, glob, os, collections
import numpy as np, polars as pl, cv2

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "open", "data")
dss = sorted(os.path.dirname(os.path.dirname(p)) for p in glob.glob(os.path.join(ROOT, "train", "*", "*", "meta", "info.json")))
np.set_printoptions(precision=2, suppress=True, linewidth=200)

def thumb(img):
    t = cv2.resize(img, (32, 24), interpolation=cv2.INTER_AREA).astype(np.float32)
    return (t - t.mean()) / (t.std() + 1e-6)

# action/state lag + per-dataset mean action, thumbnails
lag = collections.defaultdict(list); ds_mean = {}; ds_thumbs = {}
for ds in dss:
    name = os.path.relpath(ds, os.path.join(ROOT, "train"))
    fs = sorted(glob.glob(os.path.join(ds, "data", "*", "*.parquet")))
    As = []
    for f in fs[:: max(1, len(fs) // 6)]:
        t = pl.read_parquet(f)
        a = np.stack(t["action"].to_numpy()); s = np.stack(t["observation.state"].to_numpy())
        As.append(a)
        if len(a) > 4:
            for k in range(0, 4):
                lag[k].append(np.abs(a[: len(a) - k] - s[k:]).mean(0))
    ds_mean[name] = np.concatenate(As).mean(0)
    vids = sorted(glob.glob(os.path.join(ds, "videos", "*", "*", "*.mp4")))
    th = []
    for v in vids[:: max(1, len(vids) // 4)][:4]:
        cap = cv2.VideoCapture(v)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for idx in (0, n // 2):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx); ok, fr = cap.read()
            if ok:
                if fr.shape[:2] != (480, 640): fr = cv2.resize(fr, (640, 480))
                th.append(thumb(fr))
    ds_thumbs[name] = np.stack(th)
for k in range(4):
    print(f"MAE action[t] vs state[t+{k}]", np.mean(lag[k], 0), "sum", np.mean(lag[k], 0).sum())

evimgs = sorted(glob.glob(os.path.join(ROOT, "eval", "images", "*.png")))
E = np.stack([np.load(f.replace("images", "actions").replace(".png", ".npy")) for f in evimgs])
names = list(ds_thumbs)
match, dists = [], []
for i, f in enumerate(evimgs):
    t = thumb(cv2.imread(f))
    d = [np.min(((ds_thumbs[n] - t) ** 2).mean((1, 2, 3))) for n in names]
    j = int(np.argmin(d)); match.append(names[j]); dists.append(d[j])
c = collections.Counter(match)
print("eval -> nearest train dataset (image thumb):", len(c), "distinct datasets")
for n, k in c.most_common(): print(f"  {k:3d}  {n}")
dists = np.array(dists); print("match dist pct", np.percentile(dists, [5, 50, 95]))
# action mean consistency between eval and matched dataset
diff = np.array([np.abs(E[i].mean(0) - ds_mean[match[i]]) for i in range(len(E))])
print("eval mean-action vs matched dataset mean, median abs diff per joint", np.median(diff, 0))
pl.DataFrame({"sample": [os.path.basename(f)[:-4] for f in evimgs], "nearest_ds": match, "dist": dists}).write_csv(
    os.path.join(os.path.dirname(__file__), "eval_nearest_ds.csv"))


"""Independent recomputation of the score terms between generated and reference videos.

Written separately from the lead's wmscore so the two can be compared. Uses only public weights:
timm vit_small_patch14_dinov2.lvd142m and torchvision r3d_18 (Kinetics-400). Never touches the submission kit.
DINO runs at 518x518 (timm's default for this model) on a square zero-padded letterbox of the 320x512 frame.
The action term (optional) uses the lead's IDM weights with our own preprocessing (cv2 letterbox to 128x208, fp32).

  python tools/verify/feature_scores.py --ref <dir of reference mp4> --pred <dir of mp4 | "repeat"> --out scores.csv \
      [--idm C:/Dacon/WM_Shared/idm/idm_v1.pt]
"repeat" builds the first-frame-repeat video from each reference's frame 0 (calibration point).
Target actions default to <ref>/../actions/<id>.npy.
"""
import argparse, csv, glob, json, os, sys

import av, cv2, numpy as np, timm, torch, torch.nn.functional as F
from torchvision.models.video import R3D_18_Weights, r3d_18

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, REPO)
H, W = 320, 512
IDM_H, IDM_W = 128, 208
IMNET = (np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225]))
KINETICS = (np.array([0.43216, 0.394666, 0.37645]), np.array([0.22803, 0.22145, 0.216989]))


def read_video(path):
    with av.open(path) as c:
        return [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]


def letterbox(fr, oh=H, ow=W):
    h, w = fr.shape[:2]
    s = min(oh / h, ow / w)
    nh, nw = round(h * s), round(w * s)
    out = np.zeros((oh, ow, 3), np.uint8)
    y, x = (oh - nh) // 2, (ow - nw) // 2
    out[y:y + nh, x:x + nw] = cv2.resize(fr, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    return out


def to_tensor(frames, size, norm):
    x = np.stack([cv2.resize(f, (size[1], size[0]), interpolation=cv2.INTER_AREA) if f.shape[:2] != size else f
                  for f in frames]).astype(np.float32) / 255
    x = (x - norm[0]) / norm[1]
    return torch.from_numpy(x.astype(np.float32)).permute(0, 3, 1, 2)  # T,C,H,W


class Scorer:
    def __init__(self, device, dino_size=518, idm=None):
        self.dev, self.ds = device, dino_size
        self.dino = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0,
                                      img_size=dino_size).eval().to(device)
        r3d = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
        r3d.fc = torch.nn.Identity()
        self.r3d = r3d.eval().to(device)
        self.idm = None
        if idm:
            from wmscore.idm import load_idm
            self.idm = load_idm(idm, device)

    @torch.no_grad()
    def feats(self, frames):
        lb = [letterbox(f) for f in frames]
        sq = [letterbox(f, self.ds, self.ds) for f in lb]
        d = self.dino(to_tensor(sq, (self.ds, self.ds), IMNET).to(self.dev))  # 16,384
        v = self.r3d(to_tensor(lb, (112, 112), KINETICS).permute(1, 0, 2, 3)[None].to(self.dev))  # 1,512
        return d, v[0]

    @torch.no_grad()
    def actions(self, frames):
        x = np.stack([letterbox(f, IDM_H, IDM_W) for f in frames]).astype(np.float32) / 255
        x = torch.from_numpy(x).permute(0, 3, 1, 2)[None].to(self.dev)
        return self.idm(x)[0].float().cpu().numpy()

    def compare(self, pred, ref):
        dp, vp = self.feats(pred)
        dr, vr = self.feats(ref)
        return dict(dino=(1 - F.cosine_similarity(dp, dr, dim=1)).mean().item(),
                    dino_flat=(1 - F.cosine_similarity(dp.flatten(), dr.flatten(), dim=0)).item(),
                    r3d=(1 - F.cosine_similarity(vp, vr, dim=0)).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=r"C:\Dacon\WM_Shared\holdout_v1\gt_videos")
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dino-size", type=int, default=518)
    ap.add_argument("--idm", default=None, help="IDM checkpoint for the action term")
    ap.add_argument("--actions", default=None, help="dir of target .npy actions (default <ref>/../actions)")
    args = ap.parse_args()
    torch.set_num_threads(max(1, os.cpu_count() - 2))
    sc = Scorer(args.device, args.dino_size, args.idm)
    stats = json.load(open(os.path.join(REPO, "open", "data", "train", "so100_action_statistics.json"), encoding="utf-8"))
    mu, sd = np.array(stats["mean"]), np.array(stats["std"])
    act_dir = args.actions or os.path.join(os.path.dirname(os.path.abspath(args.ref)), "actions")
    keys = ("dino", "dino_flat", "r3d", "action")
    rows = []
    for rp in sorted(glob.glob(os.path.join(args.ref, "*.mp4"))):
        sid = os.path.splitext(os.path.basename(rp))[0]
        ref = read_video(rp)
        row = dict(sample_id=sid, n_frames=0, **{k: None for k in keys}, error="")
        if args.pred == "repeat":
            pred = [ref[0]] * len(ref)
        else:
            pp = os.path.join(args.pred, sid + ".mp4")
            if not os.path.exists(pp):
                rows.append({**row, "error": "missing"})
                continue
            pred = read_video(pp)
        row["n_frames"] = len(pred)
        if len(pred) != 16:
            rows.append({**row, "error": "not 16 frames"})
            continue
        row.update({k: round(v, 6) for k, v in sc.compare(pred, ref).items()})
        if sc.idm is not None:
            target = (np.load(os.path.join(act_dir, sid + ".npy")) - mu) / sd
            row["action"] = round(float(np.abs(sc.actions(pred) - target).mean()), 6)
        rows.append(row)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    ok = [r for r in rows if not r["error"]]
    m = {k: float(np.mean([r[k] for r in ok])) for k in keys if ok and ok[0][k] is not None}
    if "action" in m:
        m["score"] = 0.3 * m["dino"] + 0.3 * m["r3d"] + 0.4 * m["action"]
    print(f"n={len(ok)}/{len(rows)} " + " ".join(f"{k}={v:.4f}" for k, v in m.items()))


if __name__ == "__main__":
    main()

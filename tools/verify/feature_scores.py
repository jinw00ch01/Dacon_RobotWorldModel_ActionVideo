"""Independent recomputation of the feature terms (DINO, R3D) between generated and reference videos.

Written separately from the lead's wmscore so the two can be compared. Uses only public weights:
timm vit_small_patch14_dinov2.lvd142m and torchvision r3d_18 (Kinetics-400). Never touches the submission kit.

  python tools/verify/feature_scores.py --ref <dir of reference mp4> --pred <dir of mp4 | "repeat"> --out scores.csv
"repeat" builds the first-frame-repeat video from each reference's frame 0 (calibration point).
"""
import argparse, csv, glob, os

import av, cv2, numpy as np, timm, torch, torch.nn.functional as F
from torchvision.models.video import R3D_18_Weights, r3d_18

H, W = 320, 512
IMNET = (np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225]))
KINETICS = (np.array([0.43216, 0.394666, 0.37645]), np.array([0.22803, 0.22145, 0.216989]))


def read_video(path):
    with av.open(path) as c:
        return [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]


def letterbox(fr):
    h, w = fr.shape[:2]
    s = min(H / h, W / w)
    nh, nw = round(h * s), round(w * s)
    out = np.zeros((H, W, 3), np.uint8)
    y, x = (H - nh) // 2, (W - nw) // 2
    out[y:y + nh, x:x + nw] = cv2.resize(fr, (nw, nh), interpolation=cv2.INTER_AREA)
    return out


def to_tensor(frames, size, norm):
    x = np.stack([cv2.resize(f, (size[1], size[0]), interpolation=cv2.INTER_AREA) for f in frames]).astype(np.float32) / 255
    x = (x - norm[0]) / norm[1]
    return torch.from_numpy(x.astype(np.float32)).permute(0, 3, 1, 2)  # T,C,H,W


class Scorer:
    def __init__(self, device):
        self.dev = device
        self.dino = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=224).eval().to(device)
        r3d = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
        r3d.fc = torch.nn.Identity()
        self.r3d = r3d.eval().to(device)

    @torch.no_grad()
    def feats(self, frames):
        lb = [letterbox(f) for f in frames]
        d = self.dino(to_tensor(lb, (224, 224), IMNET).to(self.dev))  # 16,384
        v = self.r3d(to_tensor(lb, (112, 112), KINETICS).permute(1, 0, 2, 3)[None].to(self.dev))  # 1,512
        return d, v[0]

    def compare(self, pred, ref):
        dp, vp = self.feats(pred)
        dr, vr = self.feats(ref)
        dino = (1 - F.cosine_similarity(dp, dr, dim=1)).mean().item()
        dino_flat = (1 - F.cosine_similarity(dp.flatten(), dr.flatten(), dim=0)).item()
        r3d = (1 - F.cosine_similarity(vp, vr, dim=0)).item()
        return dino, dino_flat, r3d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=r"C:\Dacon\WM_Shared\holdout_v1_clips\gt_videos")
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    torch.set_num_threads(max(1, os.cpu_count() - 2))
    sc = Scorer(args.device)
    rows = []
    for rp in sorted(glob.glob(os.path.join(args.ref, "*.mp4"))):
        sid = os.path.splitext(os.path.basename(rp))[0]
        ref = read_video(rp)
        if args.pred == "repeat":
            pred = [ref[0]] * len(ref)
        else:
            pp = os.path.join(args.pred, sid + ".mp4")
            if not os.path.exists(pp):
                rows.append(dict(sample_id=sid, n_frames=0, dino=None, dino_flat=None, r3d=None, error="missing"))
                continue
            pred = read_video(pp)
        if len(pred) != 16:
            rows.append(dict(sample_id=sid, n_frames=len(pred), dino=None, dino_flat=None, r3d=None, error="not 16 frames"))
            continue
        dino, dino_flat, r3d = sc.compare(pred, ref)
        rows.append(dict(sample_id=sid, n_frames=16, dino=round(dino, 6), dino_flat=round(dino_flat, 6), r3d=round(r3d, 6), error=""))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    ok = [r for r in rows if not r["error"]]
    print(f"n={len(ok)}/{len(rows)} " + " ".join(f"{k}={np.mean([r[k] for r in ok]):.4f}" for k in ("dino", "dino_flat", "r3d")))


if __name__ == "__main__":
    main()

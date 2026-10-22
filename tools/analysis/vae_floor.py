"""How much DINO / R3D distance the Wan2.1 VAE round trip costs at the generation size (GPU, holdout ground truth only).

Each GT clip (17 frames: the input image and the 16 targets) is encoded and decoded at H x W exactly as the generator's
latents are, upsampled to 640x480 like gen_cosmos_ac output, frame 0 set back to the input image, then optionally
background-anchored (t20), and scored against the untouched clip. Together with tools/analysis/res_floor.py this splits
the generated videos' DINO gap into resolution, VAE and generation parts.

python tools/analysis/vae_floor.py --holdout C:/Dacon/WM_Shared/holdout_v1_sub64 --sizes 240x320 320x432
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from wmgen import cosmos_ac as ca  # noqa: E402
from wmgen.anchor import AnchorParams, anchor  # noqa: E402
from wmgen.video_io import read_mp4  # noqa: E402
from wmscore.features import FeatureModels, cosine_distance, framewise_cosine_distance, to_eval_frames  # noqa: E402


@torch.no_grad()
def round_trip(vae, mean, inv_std, clip: np.ndarray, h: int, w: int, device) -> np.ndarray:
    """(17, 480, 640, 3) uint8 -> the same after VAE encode/decode at h x w, upsampled back (bicubic)."""
    x = ca.to_model_frames(clip, h, w).to(device)  # T,3,h,w in [-1,1]
    lat = ca.encode_frames(vae, mean, inv_std, x.permute(1, 0, 2, 3)[None])
    rec = ca.decode_latents(vae, mean, inv_std, lat)[0]  # 3,T,h,w
    rec = F.interpolate(rec.permute(1, 0, 2, 3), size=clip.shape[1:3], mode="bicubic", align_corners=False)
    return ((rec.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", type=Path, required=True)
    ap.add_argument("--sizes", nargs="+", default=["240x320", "320x432"])
    args = ap.parse_args()
    device = torch.device("cuda")
    vae, mean, inv_std = ca.load_vae(device)
    models = FeatureModels(device)
    from PIL import Image

    ids = sorted(p.stem for p in (args.holdout / "gt_videos").glob("*.mp4"))
    res = {f"{s}{a}": {"dino": [], "r3d": []} for s in args.sizes for a in ("", "_anchor_t20")}
    for i in ids:
        gt = read_mp4(args.holdout / "gt_videos" / f"{i}.mp4")
        image = np.asarray(Image.open(args.holdout / "images" / f"{i}.png").convert("RGB"))
        f_gt = to_eval_frames(gt)
        d_gt, r_gt = models.dino_features(f_gt), models.r3d_features(f_gt)
        clip = np.concatenate([image[None], gt[1:], gt[-1:]])  # 17 frames as the cache builds them (input + targets)
        for s in args.sizes:
            h, w = (int(x) for x in s.split("x"))
            rec = round_trip(vae, mean, inv_std, clip, h, w, device)[:16]
            rec[0] = image
            for name, video in ((s, rec), (f"{s}_anchor_t20", anchor(rec, image, AnchorParams(threshold=20.0)))):
                f = to_eval_frames(video)
                res[name]["dino"].append(framewise_cosine_distance(models.dino_features(f), d_gt))
                res[name]["r3d"].append(cosine_distance(models.r3d_features(f), r_gt))
    out = {k: {m: float(np.mean(v)) for m, v in d.items()} for k, d in res.items()}
    print(json.dumps({"n": len(ids), "vae_floor": out}, indent=1))


if __name__ == "__main__":
    main()

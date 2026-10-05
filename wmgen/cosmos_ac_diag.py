"""Find the inference settings under which the converted action-cond model is coherent.

Zero Bridge actions on a few holdout images, for RoPE scale x fps x conditioning-timestep variants.
Writes a contact sheet per variant and the mean |frame_k - frame_0| at model resolution: a coherent,
nearly still clip scores low; a broken setting scores high.

python -m wmgen.cosmos_ac_diag --eval-root C:/Dacon/WM_Shared/holdout_v1 --out C:/Dacon/WM_Shared/cosmos_ac/diag
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from wmgen import cosmos_ac as ca

VARIANTS = {
    "rope1_fpsNone": dict(rope=(1.0, 1.0, 1.0), fps=None, cond_t=None),
    "rope1_fps4": dict(rope=(1.0, 1.0, 1.0), fps=4, cond_t=None),
    "rope1_fps24": dict(rope=(1.0, 1.0, 1.0), fps=24, cond_t=None),
    "rope3_fpsNone": dict(rope=(1.0, 3.0, 3.0), fps=None, cond_t=None),
    "rope1_condt": dict(rope=(1.0, 1.0, 1.0), fps=None, cond_t=1e-4),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=320)
    args = ap.parse_args()

    device = torch.device("cuda")
    transformer = ca.load_transformer(device).eval()
    embedder = ca.load_bridge_embedder().to(device, torch.bfloat16).eval()
    vae, mean, inv_std = ca.load_vae(device)
    text = ca.load_text_embedding(device)
    args.out.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        act_D, act_3D = embedder(torch.zeros(1, 16, 7, device=device, dtype=torch.bfloat16))
    images = sorted((args.eval_root / "images").glob("*.png"))[: args.n]
    report = {}
    for name, v in VARIANTS.items():
        ca.set_rope_scale(transformer, v["rope"])
        diffs, rows = [], []
        for img_path in images:
            image = np.asarray(Image.open(img_path).convert("RGB"))
            first = ca.to_model_frames(image[None], args.height, args.width).to(device)
            video = first[:, :, None].expand(-1, -1, 17, -1, -1)
            cond = ca.encode_frames(vae, mean, inv_std, video)
            sched = ca.make_scheduler()
            lat = torch.randn(cond.shape, generator=torch.Generator().manual_seed(0)).to(device)
            mask = torch.zeros(1, 1, *cond.shape[2:], device=device)
            mask[:, :, :1] = 1
            gt_v = (lat - cond) * mask
            sched.set_timesteps(args.steps, device=device)
            with torch.no_grad():
                for i, ts in enumerate(sched.timesteps):
                    sigma = sched.sigmas[i].expand(1).to(device, torch.float32)
                    vel = ca.velocity(transformer, lat, cond, mask, sigma, text, act_D, act_3D,
                                      cond_t=v["cond_t"], fps=None if v["fps"] is None else float(v["fps"]))
                    vel = gt_v + vel * (1 - mask)
                    lat = sched.step(vel, ts, lat, return_dict=False)[0]
                frames = ca.decode_latents(vae, mean, inv_std, lat)[0]  # 3,17,H,W
            f = ((frames.permute(1, 2, 3, 0) + 1) * 127.5).clamp(0, 255).byte().cpu().numpy()
            diffs.append([float(np.abs(f[k].astype(int) - f[0]).mean()) for k in (1, 4, 8, 16)])
            rows.append(np.concatenate([f[k] for k in (0, 1, 4, 8, 16)], axis=1))
        Image.fromarray(np.concatenate(rows, axis=0)).save(args.out / f"{name}.jpg", quality=85)
        report[name] = np.mean(diffs, axis=0).round(2).tolist()
        print(name, report[name], flush=True)
    (args.out / "report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

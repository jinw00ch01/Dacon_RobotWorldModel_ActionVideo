"""S2: run the converted Cosmos-Predict2.5-2B action-cond model with zero Bridge actions.

Checks that the conversion is sound (the clip should stay coherent and nearly still) and measures
8GB memory and time at our working size. Our 6-D joint actions need a newly trained embedder (S3).

python -m wmgen.cosmos_ac_zeroshot --eval-root C:/Dacon/WM_Shared/holdout_v1 \
    --out C:/Dacon/WM_Shared/holdout_v1/pred/cosmos_ac_zero/videos --limit 16
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from wmgen import cosmos_ac as ca
from wmgen.video_io import NUM_FRAMES, write_mp4

GEN_FRAMES = 17  # 1 + 4 * 4 latent steps; the 17th frame is dropped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--steps", type=int, default=35)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda")
    transformer = ca.load_transformer(device).eval()
    embedder = ca.load_bridge_embedder().to(device, torch.bfloat16).eval()
    vae, mean, inv_std = ca.load_vae(device)
    text = ca.load_text_embedding(device)
    scheduler = ca.make_scheduler()
    with torch.no_grad():
        act_D, act_3D = embedder(torch.zeros(1, GEN_FRAMES - 1, 7, device=device, dtype=torch.bfloat16))

    images = sorted((args.eval_root / "images").glob("*.png"))[: args.limit or None]
    args.out.mkdir(parents=True, exist_ok=True)
    times = []
    for img_path in images:
        image = np.asarray(Image.open(img_path).convert("RGB"))
        torch.cuda.synchronize()
        t0 = time.time()
        first = ca.to_model_frames(image[None], args.height, args.width).to(device)  # (1,3,H,W)
        video = torch.cat([first[:, :, None], first[:, :, None].expand(-1, -1, GEN_FRAMES - 1, -1, -1)], dim=2)
        cond_latent = ca.encode_frames(vae, mean, inv_std, video)
        g = torch.Generator().manual_seed(args.seed)
        lat = ca.sample(transformer, scheduler, cond_latent, text, act_D, act_3D, steps=args.steps, generator=g)
        frames = ca.decode_latents(vae, mean, inv_std, lat)[0, :, :NUM_FRAMES]  # (3,16,H,W)
        frames = F.interpolate(frames.permute(1, 0, 2, 3), size=image.shape[:2], mode="bicubic", align_corners=False)
        frames = ((frames.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
        frames[0] = image
        torch.cuda.synchronize()
        times.append(time.time() - t0)
        write_mp4(frames, args.out / f"{img_path.stem}.mp4")
        print(f"{img_path.stem} {times[-1]:.1f}s peak {torch.cuda.max_memory_allocated() / 2**30:.2f}GiB", flush=True)
    stats = {"model": "Cosmos-Predict2.5-2B robot/action-cond, zero actions", "steps": args.steps,
             "size": [args.height, args.width], "n": len(times), "sec_per_sample": float(np.mean(times)),
             "peak_gib": torch.cuda.max_memory_allocated() / 2**30}
    (args.out.parent / "gen_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats))


if __name__ == "__main__":
    main()

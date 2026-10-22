"""Latent inverse-dynamics model: Wan2.1-VAE clip latents (16, 5, h, w) -> the clip's 16x6 relative actions (z units).

Frozen, it gives wmgen.train_cosmos_ac an action-alignment loss (--aux-idm): the generator's clean-latent estimate has to
carry the commanded motion, not only look plausible. The second-place solution (github.com/gxonu/Worldmodel-AI-Challenge,
configs/training_config.json) trained one the same way (batch 64, lr 3e-4, 25k steps, latent noise up to 0.1, EMA 0.999)
and used it on 30% of the steps with weight 0.3. Targets are relative, (a_t - a_0) / std, so a robot's joint zero does
not matter. Latent frame k >= 1 covers video frames 4k-3..4k, i.e. actions 4k-4..4k-1. Trained on the training latent
cache only (rule 3); 2% of the clips (by index) are kept out for the reported validation MAE.

python -m wmgen.latent_idm --latents C:/Dacon/WM_Shared/latents_240x320 C:/Dacon/WM_Shared/latents_240x320_extra \
    --out C:/Dacon/WM_Shared/cosmos_ac/latent_idm/latent_idm_v1.pt
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from wmscore.data import load_action_stats

STEPS, JOINTS, PER_LATENT = 16, 6, 4
VAL_EVERY = 50  # clip i of a shard is a validation clip when i % VAL_EVERY == 0


class ResBlock3d(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride=(1, 2, 2)):
        super().__init__()
        self.conv1 = nn.Conv3d(c_in, c_out, 3, stride=stride, padding=1)
        self.norm1 = nn.GroupNorm(8, c_out)
        self.conv2 = nn.Conv3d(c_out, c_out, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, c_out)
        self.skip = nn.Conv3d(c_in, c_out, 1, stride=stride)

    def forward(self, x):
        h = F.silu(self.norm1(self.conv1(x)))
        return F.silu(self.norm2(self.conv2(h)) + self.skip(x))


class LatentIDM(nn.Module):
    def __init__(self, c_in: int = 16, width: int = 96):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv3d(2 * c_in, width, 3, padding=1), nn.GroupNorm(8, width), nn.SiLU())
        self.blocks = nn.Sequential(ResBlock3d(width, width), ResBlock3d(width, 2 * width), ResBlock3d(2 * width, 256))
        self.temporal = nn.Sequential(nn.Conv1d(256, 256, 3, padding=1), nn.SiLU(), nn.Conv1d(256, 256, 3, padding=1), nn.SiLU())
        self.head = nn.Linear(2 * 256, PER_LATENT * JOINTS)  # per generated latent frame, next to the first frame's features

    def forward(self, lat: torch.Tensor) -> torch.Tensor:
        """(B, 16, 5, h, w) -> (B, 16, 6)."""
        x = torch.cat([lat, torch.diff(lat, dim=2, prepend=lat[:, :, :1])], dim=1)  # latents and their frame differences
        f = self.blocks(self.stem(x)).mean(dim=(3, 4))  # B, 256, T
        f = self.temporal(f).transpose(1, 2)  # B, T, 256
        pair = torch.cat([f[:, 1:], f[:, :1].expand(-1, f.shape[1] - 1, -1)], dim=-1)
        return self.head(pair).reshape(lat.shape[0], STEPS, JOINTS)


def relative_z(actions: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """(B, 16, 6) raw joint targets -> (a_t - a_0) / std."""
    return (actions - actions[:, :1]) / std


def load(path: Path, device) -> LatentIDM:
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = LatentIDM(**state.get("config", {}))
    model.load_state_dict(state["ema"])
    return model.to(device).eval().requires_grad_(False)


def corrupt(lat: torch.Tensor, noise_max: float, gen: torch.Generator) -> torch.Tensor:
    """Small latent noise, and on half the clips a spatial low-pass, so blurry x0 estimates still read correctly."""
    b = lat.shape[0]
    s = torch.rand(b, 1, 1, 1, 1, generator=gen, device=lat.device) * noise_max
    out = (1 - s) * lat + s * torch.randn(lat.shape, generator=gen, device=lat.device)
    blur = torch.rand(b, generator=gen, device=lat.device) < 0.5
    if blur.any():
        x = out[blur]
        bb, c, t, h, w = x.shape
        y = F.avg_pool2d(x.transpose(1, 2).reshape(bb * t, c, h, w), 2, ceil_mode=True)
        y = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)
        out[blur] = y.reshape(bb, t, c, h, w).transpose(1, 2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=25000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--noise-max", type=float, default=0.1)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--width", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    paths = [p for r in args.latents for p in sorted(Path(r).glob("*.pt"))]
    shards = [torch.load(p, mmap=True, weights_only=True) for p in paths]
    index = np.array([(k, i) for k, s in enumerate(shards) for i in range(len(s["start"]))])
    is_val = index[:, 1] % VAL_EVERY == 0
    train_idx, val_idx = index[~is_val], index[is_val]
    _, std = load_action_stats()
    std_t = torch.from_numpy(std).to(device)
    model = LatentIDM(width=args.width).to(device)
    ema = copy.deepcopy(model).requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.02)
    rng = np.random.default_rng(args.seed)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    print(f"{len(train_idx)} train / {len(val_idx)} val clips in {len(shards)} shards; "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params", flush=True)

    def fetch(rows):
        lat = torch.stack([shards[k]["latents"][i] for k, i in rows]).to(device).float()
        act = torch.stack([shards[k]["actions"][i] for k, i in rows]).to(device).float()
        return lat, relative_z(act, std_t)

    @torch.no_grad()
    def validate(net, n=2048):
        rows = val_idx[np.random.default_rng(1).choice(len(val_idx), min(n, len(val_idx)), replace=False)]
        err, still = [], []
        for j in range(0, len(rows), args.batch):
            lat, y = fetch(rows[j : j + args.batch])
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                p = net(lat).float()
            err.append((p - y).abs().mean(dim=(1, 2)).cpu())
            still.append(y.abs().mean(dim=(1, 2)).cpu())  # MAE of predicting "no motion"
        return float(torch.cat(err).mean()), float(torch.cat(still).mean())

    t0, log = time.time(), []
    model.train()
    for step in range(1, args.steps + 1):
        lat, y = fetch(train_idx[rng.integers(len(train_idx), size=args.batch)])
        lat = corrupt(lat, args.noise_max, gen)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            pred = model(lat).float()
        loss = F.smooth_l1_loss(pred, y, beta=0.05)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - args.ema)
        if step % 500 == 0 or step == args.steps:
            val, still = validate(ema)
            log.append({"step": step, "loss": round(float(loss.detach()), 4), "val_mae_z": round(val, 4), "still_mae_z": round(still, 4),
                        "sec": round(time.time() - t0)})
            print(json.dumps(log[-1]), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    state = {"ema": ema.state_dict(), "config": {"width": args.width}, "args": vars(args) | {"latents": [str(p) for p in args.latents],
             "out": str(args.out)}, "log": log}
    torch.save(state, args.out.with_suffix(".tmp"))
    args.out.with_suffix(".tmp").replace(args.out)
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()

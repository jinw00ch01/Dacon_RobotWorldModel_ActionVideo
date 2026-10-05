"""Train the inverse dynamics model on cached train-split episodes.

python -m wmscore.train_idm --cache C:/Dacon/WM_Shared/idm_cache --out C:/Dacon/WM_Shared/idm/idm_v1.pt
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from wmscore.cache_frames import load_episode
from wmscore.data import NUM_FRAMES, load_action_stats
from wmscore.idm import InverseDynamics


class Windows(Dataset):
    def __init__(self, files, mean, std, augment: bool, samples: int, seed: int):
        self.eps = [f for f in files if len(load_episode(f)[1]) >= NUM_FRAMES]
        self.mean, self.std, self.augment = mean, std, augment
        rng = random.Random(seed)
        self.items = [(rng.randrange(len(self.eps)), rng.random()) for _ in range(samples)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ep, u = self.items[i]
        if self.augment:
            ep, u = random.randrange(len(self.eps)), random.random()
        frames, actions = load_episode(self.eps[ep])
        s = int(u * (len(actions) - NUM_FRAMES + 1))
        x = torch.from_numpy(frames[s : s + NUM_FRAMES].copy()).permute(0, 3, 1, 2).float() / 255.0
        y = torch.from_numpy((actions[s : s + NUM_FRAMES] - self.mean) / self.std).float()
        if self.augment:
            x = self._augment(x)
        return x, y

    @staticmethod
    def _augment(x):
        # clip-level colour jitter and a small translation; geometry stays consistent across frames
        gain = 1 + 0.25 * (torch.rand(1, 3, 1, 1) - 0.5)
        bias = 0.1 * (torch.rand(1, 1, 1, 1) - 0.5)
        x = (x * gain + bias).clamp(0, 1)
        dy, dx = random.randint(-6, 6), random.randint(-8, 8)
        return torch.roll(x, shifts=(dy, dx), dims=(2, 3))


def evaluate(model, loader, device):
    model.eval()
    err, n = torch.zeros(6), 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for x, y in loader:
            p = model(x.to(device, non_blocking=True)).float().cpu()
            err += (p - y).abs().mean(dim=1).sum(dim=0)
            n += len(y)
    model.train()
    per_joint = (err / n).tolist()
    return float(np.mean(per_joint)), per_joint


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda")
    mean, std = load_action_stats()
    manifest = json.loads((args.cache / "manifest.json").read_text())
    train = Windows([args.cache / "train" / n for n in manifest["train"]], mean, std, True, args.steps * args.batch, args.seed)
    val = Windows([args.cache / "val" / n for n in manifest["val"]], mean, std, False, 1024, args.seed + 1)
    tl = DataLoader(train, batch_size=args.batch, num_workers=args.workers, pin_memory=True,
                    persistent_workers=True, drop_last=True)
    vl = DataLoader(val, batch_size=args.batch, num_workers=2, pin_memory=True)

    model = InverseDynamics().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    log, best, t0 = [], float("inf"), time.time()
    for step, (x, y) in enumerate(tl, start=1):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = (model(x.to(device, non_blocking=True)).float() - y.to(device)).abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % args.eval_every == 0 or step == args.steps:
            v, per_joint = evaluate(model, vl, device)
            log.append({"step": step, "train_l1": float(loss.detach()), "val_mae": v, "val_per_joint": per_joint,
                        "sec": round(time.time() - t0)})
            print(json.dumps(log[-1]), flush=True)
            state = {"model": model.state_dict(), "step": step, "val_mae": v, "args": {k: str(a) for k, a in vars(args).items()}}
            torch.save(state, args.out.with_suffix(".last.pt"))
            if v < best:
                best = v
                torch.save(state, args.out)
    (args.out.with_suffix(".log.json")).write_text(json.dumps(log, indent=1))
    print(f"best val MAE (z) {best:.4f}")


if __name__ == "__main__":
    main()

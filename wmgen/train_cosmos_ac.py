"""S3: adapt Cosmos-Predict2.5-2B action-cond to SO-100 joint actions on cached latents (8GB).

Trains a new 6-joint action embedder (first layer new, second layer from the released Bridge
embedder) plus LoRA on self-attention and MLP. Rectified-flow loss on latent frames 1..4; latent
frame 0 is the clean conditioning image, as at inference.

python -m wmgen.train_cosmos_ac --latents C:/Dacon/WM_Shared/latents_240x320 --out C:/Dacon/WM_Shared/cosmos_ac/runs/v1
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from wmgen import cosmos_ac as ca
from wmgen.actions import FEATURES_PER_STEP, action_features

LORA_TARGETS = r".*transformer_blocks\.\d+\.(attn1\.(to_q|to_k|to_v|to_out\.0)|ff\.net\.0\.proj|ff\.net\.2)"
# also adapt the AdaLN modulation MLPs, through which the action embedding reaches every block
LORA_TARGETS_ADALN = LORA_TARGETS[:-1] + r"|norm[123]\.linear_[12])"


def lora_targets(adaln: bool) -> str:
    return LORA_TARGETS_ADALN if adaln else LORA_TARGETS


class LatentClips:
    """Memory-mapped per-dataset shards, sampled with weight ~ sqrt(size) to soften dataset imbalance."""

    def __init__(self, root: Path, seed: int):
        self.shards = [torch.load(p, mmap=True, weights_only=True) for p in sorted(root.glob("*.pt"))]
        sizes = np.array([len(s["start"]) for s in self.shards], dtype=np.float64)
        self.p = np.sqrt(sizes) / np.sqrt(sizes).sum()
        self.rng = np.random.default_rng(seed)
        self.total = int(sizes.sum())

    def batch(self, n: int):
        lat, act = [], []
        for _ in range(n):
            s = self.shards[self.rng.choice(len(self.shards), p=self.p)]
            i = int(self.rng.integers(len(s["start"])))
            lat.append(s["latents"][i].float())
            act.append(s["actions"][i])
        return torch.stack(lat), torch.stack(act)


def build_embedder(device, frame_tokens: bool = False) -> ca.ActionEmbedder:
    """New first layer with zero weights and the released bias, released second layer.

    At step 0 every input maps to the released embedder's output for a zero Bridge action, which
    gives coherent, nearly still clips (wmgen.cosmos_ac_diag). A randomly initialised first layer
    instead injects large random embeddings and the clips fall apart.
    """
    emb = ca.ActionEmbedder(d_in=ca.STEPS_PER_LATENT * FEATURES_PER_STEP, frame_tokens=frame_tokens)
    bridge = torch.load(ca.OUT / "action_embedder_bridge.pt", map_location="cpu", weights_only=True)
    for head in ("to_D", "to_3D"):
        mlp = getattr(emb, head)
        mlp.fc1.weight.data.zero_()
        mlp.fc1.bias.data.copy_(bridge[f"{head}.fc1.bias"])
        mlp.fc2.weight.data.copy_(bridge[f"{head}.fc2.weight"])
        mlp.fc2.bias.data.copy_(bridge[f"{head}.fc2.bias"])
    return emb.to(device)


def save(out: Path, transformer, embedder, opt, step: int, args) -> None:
    from peft.utils import get_peft_model_state_dict

    state = {"lora": get_peft_model_state_dict(transformer), "embedder": embedder.state_dict(),
             "step": step, "args": vars(args) | {"latents": str(args.latents), "out": str(args.out)}}
    torch.save(state, out / "adapter_last.pt.tmp")
    (out / "adapter_last.pt.tmp").replace(out / "adapter_last.pt")
    torch.save(opt.state_dict(), out / "optimizer_last.pt.tmp")
    (out / "optimizer_last.pt.tmp").replace(out / "optimizer_last.pt")
    if step % args.keep_every == 0:
        torch.save(state, out / f"adapter_{step:06d}.pt")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--embedder-lr", type=float, default=3e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--action-dropout", type=float, default=0.1)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--keep-every", type=int, default=5000)
    ap.add_argument("--frame-tokens", action="store_true", help="also add a per-frame action offset to the tokens")
    ap.add_argument("--lora-adaln", action="store_true", help="also put LoRA on the AdaLN modulation MLPs")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from peft import LoraConfig
    from peft.utils import set_peft_model_state_dict

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps(vars(args), default=str, indent=1))

    transformer = ca.load_transformer(device)
    transformer.requires_grad_(False)
    transformer.add_adapter(LoraConfig(r=args.rank, lora_alpha=args.rank, target_modules=lora_targets(args.lora_adaln),
                                       init_lora_weights="gaussian"))
    for p in transformer.parameters():
        if p.requires_grad:
            p.data = p.data.float()  # fp32 master weights for LoRA
    transformer.enable_gradient_checkpointing()
    embedder = build_embedder(device, args.frame_tokens)
    text = ca.load_text_embedding(device)
    lora_params = [p for p in transformer.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": lora_params, "lr": args.lr},
                             {"params": embedder.parameters(), "lr": args.embedder_lr}], weight_decay=0.0)
    start = 0
    if (args.out / "adapter_last.pt").exists():
        state = torch.load(args.out / "adapter_last.pt", map_location="cpu", weights_only=False)
        set_peft_model_state_dict(transformer, state["lora"])
        embedder.load_state_dict(state["embedder"])
        opt.load_state_dict(torch.load(args.out / "optimizer_last.pt", map_location="cpu", weights_only=False))
        start = state["step"]
        print(f"resumed at step {start}", flush=True)

    data = LatentClips(args.latents, args.seed + start)
    print(f"{data.total} clips in {len(data.shards)} shards; trainable LoRA "
          f"{sum(p.numel() for p in lora_params) / 1e6:.1f}M, embedder {sum(p.numel() for p in embedder.parameters()) / 1e6:.1f}M",
          flush=True)
    warmup = 200
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))),
        last_epoch=start - 1 if start else -1)
    t0, log = time.time(), []
    transformer.train()
    for step in range(start + 1, args.steps + 1):
        x0, actions = data.batch(args.batch)
        x0 = x0.to(device)
        feats = action_features(actions).to(device)
        drop = torch.rand(len(feats), device=device) < args.action_dropout
        feats = torch.where(drop[:, None, None], torch.zeros_like(feats), feats)
        b, c, t, h, w = x0.shape
        sigma = torch.sigmoid(torch.randn(b, device=device)).clamp(0.01, 0.995)
        noise = torch.randn_like(x0)
        xt = (1 - sigma.view(b, 1, 1, 1, 1)) * x0 + sigma.view(b, 1, 1, 1, 1) * noise
        cond_mask = torch.zeros(b, 1, t, h, w, device=device)
        cond_mask[:, :, :1] = 1
        with torch.autocast("cuda", dtype=torch.bfloat16):
            act_D, act_3D = embedder(feats)
            v = ca.velocity(transformer, xt, x0, cond_mask, sigma, text, act_D, act_3D,
                            action_tok=embedder.tokens(feats))
        target = noise - x0
        loss = F.mse_loss(v[:, :, 1:], target[:, :, 1:])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_params + list(embedder.parameters()), 1.0)
        opt.step()
        sched.step()
        if step % 50 == 0:
            log.append({"step": step, "loss": float(loss.detach()), "sec": round(time.time() - t0),
                        "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)})
            print(json.dumps(log[-1]), flush=True)
        if step % args.save_every == 0 or step == args.steps:
            save(args.out, transformer, embedder, opt, step, args)
            with (args.out / "train_log.jsonl").open("a") as f:
                f.writelines(json.dumps(r) + "\n" for r in log)
            log = []
    print("done", flush=True)


if __name__ == "__main__":
    main()

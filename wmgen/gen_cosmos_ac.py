"""Generate 16-frame videos with an adapted Cosmos action-cond model (holdout or eval layout).

python -m wmgen.gen_cosmos_ac --adapter C:/Dacon/WM_Shared/cosmos_ac/runs/v1/adapter_last.pt \
    --eval-root C:/Dacon/WM_Shared/holdout_v1 --out C:/Dacon/WM_Shared/holdout_v1/pred/cosmos_ac_v1/videos
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
from wmgen.actions import FEATURES_PER_STEP, action_features
from wmgen.offset_route import ENVELOPES, load_envelope, routed_joints
from wmgen.video_io import NUM_FRAMES, write_mp4

GEN_FRAMES = 17


def load_adapted(adapter: Path, device):
    from peft import LoraConfig
    from peft.utils import set_peft_model_state_dict

    state = torch.load(adapter, map_location="cpu", weights_only=False)
    transformer = ca.load_transformer(device)
    rank = int(state["args"]["rank"])
    from wmgen.train_cosmos_ac import lora_targets

    a = state["args"]
    transformer.add_adapter(LoraConfig(r=rank, lora_alpha=rank, target_modules=lora_targets(bool(a.get("lora_adaln")))))
    set_peft_model_state_dict(transformer, state["lora"])
    transformer.to(device, torch.bfloat16).eval()
    embedder = ca.ActionEmbedder(d_in=ca.STEPS_PER_LATENT * FEATURES_PER_STEP, frame_tokens=bool(a.get("frame_tokens")))
    embedder.load_state_dict(state["embedder"])
    return transformer, embedder.to(device, torch.bfloat16).eval(), state


@torch.no_grad()
def generate(transformer, embedder, vae, mean, inv_std, text, scheduler, image: np.ndarray, actions: np.ndarray,
             height: int, width: int, steps: int, guidance: float, seed: int, device,
             rel_joints: np.ndarray | None = None) -> np.ndarray:
    feats = action_features(actions, rel_joints)[None].to(device, torch.bfloat16)
    act_D, act_3D = embedder(feats)
    null_D, null_3D = embedder(torch.zeros_like(feats))
    first = ca.to_model_frames(image[None], height, width).to(device)
    video = first[:, :, None].expand(-1, -1, GEN_FRAMES, -1, -1)
    cond_latent = ca.encode_frames(vae, mean, inv_std, video)
    lat = ca.sample(transformer, scheduler, cond_latent, text, act_D, act_3D, steps=steps,
                    generator=torch.Generator().manual_seed(seed), guidance=guidance, null_D=null_D, null_3D=null_3D,
                    action_tok=embedder.tokens(feats), null_tok=embedder.tokens(torch.zeros_like(feats)))
    frames = ca.decode_latents(vae, mean, inv_std, lat)[0, :, :NUM_FRAMES]
    frames = F.interpolate(frames.permute(1, 0, 2, 3), size=image.shape[:2], mode="bicubic", align_corners=False)
    frames = ((frames.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
    frames[0] = image  # frame 0 is the input image itself
    return frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--eval-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--abs-mode", choices=["keep", "auto", "liftlo", "rel"], default="keep",
                    help="keep: absolute channel as trained; auto: relative for joints outside the training "
                         "calibration envelope (wmgen.offset_route, sub5 rule); liftlo: auto plus shoulder_lift below "
                         "the envelope (submission 6 rule); rel: relative for every joint")
    args = ap.parse_args()

    device = torch.device("cuda")
    transformer, embedder, state = load_adapted(args.adapter, device)
    vae, mean, inv_std = ca.load_vae(device)
    text = ca.load_text_embedding(device)
    scheduler = ca.make_scheduler()
    images = sorted((args.eval_root / "images").glob("*.png"))[: args.limit or None]
    args.out.mkdir(parents=True, exist_ok=True)
    env_path = {"auto": ENVELOPES["sub5"], "liftlo": ENVELOPES["s6"]}.get(args.abs_mode)
    env = load_envelope(env_path) if env_path else None
    times, routed = [], {}
    for img_path in images:
        out = args.out / f"{img_path.stem}.mp4"
        if out.exists():
            continue
        image = np.asarray(Image.open(img_path).convert("RGB"))
        actions = np.load(args.eval_root / "actions" / f"{img_path.stem}.npy")
        rel_joints = None
        if args.abs_mode == "rel":
            rel_joints = np.ones(actions.shape[-1], bool)
        elif env is not None:
            rel_joints = routed_joints(actions, env)
            if rel_joints.any():
                routed[img_path.stem] = [int(j) for j in np.flatnonzero(rel_joints)]
        t0 = time.time()
        frames = generate(transformer, embedder, vae, mean, inv_std, text, scheduler, image, actions,
                          args.height, args.width, args.steps, args.guidance, args.seed, device, rel_joints)
        times.append(time.time() - t0)
        write_mp4(frames, out)
    stats = {"adapter": str(args.adapter), "adapter_step": state["step"], "steps": args.steps,
             "guidance": args.guidance, "seed": args.seed, "abs_mode": args.abs_mode, "envelope": str(env_path) if env_path else None, "routed": routed,
             "size": [args.height, args.width], "n": len(times),
             "sec_per_sample": float(np.mean(times)) if times else None,
             "peak_gib": torch.cuda.max_memory_allocated() / 2**30}
    if times:  # a resume that generated nothing keeps the earlier stats
        (args.out.parent / "gen_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats))


if __name__ == "__main__":
    main()

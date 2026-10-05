"""S2: zero-shot Wan2.1-VACE-1.3B first-frame-to-video (no action conditioning yet).

Measures 8GB memory/time and gives a visual-quality reference on the holdout.

python -m wmgen.wan_vace_zeroshot --eval-root C:/Dacon/WM_Shared/holdout_v1 \
    --out C:/Dacon/WM_Shared/holdout_v1/pred/wan_vace_zs/videos --steps 20 --limit 48
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from wmgen.video_io import NUM_FRAMES, write_mp4

MODEL = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
PROMPT = ("A fixed camera films a robot arm doing a tabletop manipulation task. "
          "The arm moves smoothly; the background and the camera stay still.")
NEGATIVE = ("camera motion, zoom, cut, flicker, blurred details, deformed robot, extra arms, "
            "worst quality, low quality, JPEG compression residue, subtitles, overall gray")
EMBED_CACHE = Path(r"C:\Dacon\WM_Shared\wan_vace\prompt_embeds.pt")


def prompt_embeds() -> dict:
    """Encode the fixed prompt once on CPU (umT5-XXL does not fit next to the DiT in 8GB)."""
    if EMBED_CACHE.exists():
        return torch.load(EMBED_CACHE, weights_only=True)
    from transformers import AutoTokenizer, UMT5EncoderModel

    tok = AutoTokenizer.from_pretrained(MODEL, subfolder="tokenizer")
    enc = UMT5EncoderModel.from_pretrained(MODEL, subfolder="text_encoder", torch_dtype=torch.bfloat16)
    out = {}
    for key, text in (("prompt", PROMPT), ("negative", NEGATIVE)):
        ids = tok([text], padding="max_length", max_length=512, truncation=True, return_tensors="pt")
        with torch.no_grad():
            h = enc(ids.input_ids, attention_mask=ids.attention_mask).last_hidden_state
        n = int(ids.attention_mask.sum())
        h[:, n:] = 0  # the pipeline zero-pads past the real tokens
        out[key] = h.to(torch.bfloat16)
    EMBED_CACHE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, EMBED_CACHE)
    return out


def load_pipeline(device):
    from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanVACEPipeline

    vae = AutoencoderKLWan.from_pretrained(MODEL, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanVACEPipeline.from_pretrained(MODEL, vae=vae, text_encoder=None, torch_dtype=torch.bfloat16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)
    pipe.transformer.to(device)
    pipe.vae.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--guidance", type=float, default=5.0)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda")
    emb = prompt_embeds()
    pipe = load_pipeline(device)
    images = sorted((args.eval_root / "images").glob("*.png"))
    if args.limit:
        images = images[: args.limit]
    num_frames = 17  # Wan needs 4k+1 frames; keep the first 16
    times = []
    for img_path in images:
        out = args.out / f"{img_path.stem}.mp4"
        if out.exists():
            continue
        image = Image.open(img_path).convert("RGB")
        first = image.resize((args.width, args.height), Image.BICUBIC)
        gray = Image.new("RGB", (args.width, args.height), (128, 128, 128))
        video = [first] + [gray] * (num_frames - 1)
        mask = [Image.new("L", first.size, 0)] + [Image.new("L", first.size, 255)] * (num_frames - 1)
        torch.cuda.synchronize()
        t0 = time.time()
        frames = pipe(
            video=video, mask=mask,
            prompt_embeds=emb["prompt"].to(device), negative_prompt_embeds=emb["negative"].to(device),
            height=args.height, width=args.width, num_frames=num_frames,
            num_inference_steps=args.steps, guidance_scale=args.guidance,
            generator=torch.Generator(device="cpu").manual_seed(args.seed), output_type="np",
        ).frames[0]
        torch.cuda.synchronize()
        times.append(time.time() - t0)
        frames = (np.clip(frames[:NUM_FRAMES], 0, 1) * 255).round().astype(np.uint8)
        if frames.shape[1:3] != (image.height, image.width):
            frames = np.stack([np.asarray(Image.fromarray(f).resize(image.size, Image.BICUBIC)) for f in frames])
        frames[0] = np.asarray(image)  # frame 0 is the input image itself
        write_mp4(frames, out)
        print(f"{img_path.stem} {times[-1]:.1f}s peak {torch.cuda.max_memory_allocated() / 2**30:.2f}GiB", flush=True)
    stats = {"model": MODEL, "steps": args.steps, "guidance": args.guidance, "size": [args.height, args.width],
             "n": len(times), "sec_per_sample": float(np.mean(times)) if times else None,
             "peak_gib": torch.cuda.max_memory_allocated() / 2**30}
    (args.out.parent / "gen_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats))


if __name__ == "__main__":
    main()

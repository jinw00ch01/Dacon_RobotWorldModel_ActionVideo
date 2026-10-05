"""Cosmos-Predict2.5-2B robot action-conditioned DiT, run through diffusers' CosmosTransformer3DModel.

The released checkpoint (robot/action-cond) is the per-latent-frame action variant of NVIDIA's
ActionChunkConditionedMinimalV1LVGDiT: two MLPs map the actions of each latent frame (4 steps x 7-D
Bridge end-effector deltas) to vectors added to the timestep embedding (D) and to the AdaLN-LoRA
input (3D); the conditioning latent frame gets zeros. We reproduce that with a wrapper around
diffusers' time embedding, so the backbone is plain PyTorch (no transformer_engine) and fits 8GB.

python -m wmgen.cosmos_ac convert   # one-off: write diffusers weights to C:/Dacon/WM_Shared/cosmos_ac
"""
from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CKPT_GLOB = "~/.cache/huggingface/hub/models--nvidia--Cosmos-Predict2.5-2B/snapshots/*/robot/action-cond/*"
SHARED = Path(os.environ.get("WM_SHARED", r"C:\Dacon\WM_Shared"))  # Colab sets this to a Drive folder
OUT = SHARED / "cosmos_ac"
VAE_REPO = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"  # the Wan2.1 VAE, identical to Cosmos-Predict2.5's tokenizer
TRANSFORMER_TYPE = "Cosmos-2.5-Predict-Base-2B"
STEPS_PER_LATENT = 4


class Mlp(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.activation = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))


class ActionEmbedder(nn.Module):
    """Per-latent-frame action embedding -> (B, T_lat, D) and (B, T_lat, 3D); first latent frame zeros."""

    def __init__(self, d_in: int, model_dim: int = 2048, hidden: int = 8192):
        super().__init__()
        self.to_D = Mlp(d_in, hidden, model_dim)
        self.to_3D = Mlp(d_in, hidden, 3 * model_dim)

    def forward(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """actions (B, n_steps, A) with n_steps a multiple of 4."""
        b, n, a = actions.shape
        x = actions.reshape(b, n // STEPS_PER_LATENT, STEPS_PER_LATENT * a)
        d, d3 = self.to_D(x), self.to_3D(x)
        return F.pad(d, (0, 0, 1, 0)), F.pad(d3, (0, 0, 1, 0))


class ActionTimeEmbed(nn.Module):
    """Drop-in for diffusers' CosmosEmbedding that adds the current action embedding before the norm."""

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
        self.action_D: torch.Tensor | None = None
        self.action_3D: torch.Tensor | None = None

    def forward(self, hidden_states: torch.Tensor, timestep: torch.Tensor):
        proj = self.base.time_proj(timestep).type_as(hidden_states)
        temb = self.base.t_embedder(proj)
        if self.action_D is not None:
            if timestep.numel() != self.action_D.shape[0] * self.action_D.shape[1]:
                raise ValueError("action conditioning needs per-latent-frame timesteps (B,1,T,1,1)")
            proj = proj + self.action_D.reshape(proj.shape).to(proj.dtype)
            temb = temb + self.action_3D.reshape(temb.shape).to(temb.dtype)
        return temb, self.base.norm(proj)


def _converter():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from third_party import convert_cosmos_to_diffusers as conv

    return conv


def convert(out: Path = OUT) -> None:
    files = glob.glob(os.path.expanduser(CKPT_GLOB))
    ckpt = [p for p in files if p.endswith("_ema_bf16.pt")][0]
    conv = _converter()
    sd = conv.get_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False))
    action = {k.removeprefix("net."): sd.pop(k) for k in [k for k in sd if "action_embedder" in k]}
    transformer = conv.convert_transformer(TRANSFORMER_TYPE, sd)
    transformer.save_pretrained(out / "transformer", safe_serialization=True)
    renamed = {k.replace("action_embedder_B_D.", "to_D.").replace("action_embedder_B_3D.", "to_3D."): v
               for k, v in action.items()}
    torch.save(renamed, out / "action_embedder_bridge.pt")
    emb = [p for p in files if p.endswith("text_embeddings.pt")][0]
    torch.save(torch.load(emb, map_location="cpu", weights_only=False), out / "empty_text_embeddings.pt")
    print("converted to", out)


def load_transformer(device, dtype=torch.bfloat16):
    from diffusers import CosmosTransformer3DModel

    tr = CosmosTransformer3DModel.from_pretrained(OUT / "transformer", torch_dtype=dtype)
    tr.time_embed = ActionTimeEmbed(tr.time_embed)
    return tr.to(device)


def load_bridge_embedder() -> ActionEmbedder:
    """The released 7-D Bridge action MLPs (input 4 steps x 7)."""
    emb = ActionEmbedder(d_in=STEPS_PER_LATENT * 7)
    emb.load_state_dict(torch.load(OUT / "action_embedder_bridge.pt", map_location="cpu", weights_only=True))
    return emb


def load_text_embedding(device, dtype=torch.bfloat16) -> torch.Tensor:
    emb = torch.load(OUT / "empty_text_embeddings.pt", map_location="cpu", weights_only=False)
    if isinstance(emb, dict):
        emb = next(iter(emb.values()))
    emb = torch.as_tensor(emb)
    if emb.ndim == 2:
        emb = emb[None]
    return emb.to(device, dtype)


def load_vae(device, dtype=torch.bfloat16):
    from diffusers import AutoencoderKLWan

    vae = AutoencoderKLWan.from_pretrained(VAE_REPO, subfolder="vae", torch_dtype=dtype).to(device)
    mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
    inv_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
    return vae, mean.to(device), inv_std.to(device)


def make_scheduler():
    from diffusers import UniPCMultistepScheduler

    return UniPCMultistepScheduler(use_karras_sigmas=True, use_flow_sigmas=True, prediction_type="flow_prediction",
                                   sigma_max=200.0, sigma_min=0.01)


@torch.no_grad()
def encode_frames(vae, mean, inv_std, frames: torch.Tensor) -> torch.Tensor:
    """frames (B,3,T,H,W) in [-1,1] -> normalised latents (B,16,T',h,w) float32."""
    lat = vae.encode(frames.to(vae.dtype)).latent_dist.mode().float()
    return (lat - mean) * inv_std


@torch.no_grad()
def decode_latents(vae, mean, inv_std, lat: torch.Tensor) -> torch.Tensor:
    """-> (B,3,T,H,W) in [-1,1]."""
    return vae.decode((lat / inv_std + mean).to(vae.dtype), return_dict=False)[0].float().clamp(-1, 1)


def velocity(transformer, latents, cond_latent, cond_mask, sigma, text, action_D, action_3D, cond_t=1e-4):
    """One flow-velocity prediction with the first latent frame clamped to the conditioning image."""
    b, _, t, h, w = latents.shape
    cond_ind = cond_mask[:, :, :, :1, :1]
    timestep = cond_ind * cond_t + (1 - cond_ind) * sigma.view(b, 1, 1, 1, 1)
    x = (cond_mask * cond_latent + (1 - cond_mask) * latents).to(transformer.dtype)
    transformer.time_embed.action_D, transformer.time_embed.action_3D = action_D, action_3D
    try:
        v = transformer(hidden_states=x, condition_mask=cond_mask.to(transformer.dtype),
                        timestep=timestep.to(transformer.dtype), encoder_hidden_states=text.expand(b, -1, -1),
                        padding_mask=x.new_zeros(1, 1, h * 8, w * 8), return_dict=False)[0]
    finally:
        transformer.time_embed.action_D = transformer.time_embed.action_3D = None
    return v.float()


@torch.no_grad()
def sample(transformer, scheduler, cond_latent, text, action_D, action_3D, steps=35, generator=None, guidance=0.0,
           null_D=None, null_3D=None):
    """Rectified-flow sampling as in diffusers' Cosmos2_5_PredictBasePipeline, with optional action guidance."""
    b, c, t, h, w = cond_latent.shape
    latents = torch.randn((b, c, t, h, w), generator=generator, device="cpu").to(cond_latent.device)
    cond_mask = torch.zeros((b, 1, t, h, w), device=cond_latent.device)
    cond_mask[:, :, :1] = 1
    gt_velocity = (latents - cond_latent) * cond_mask
    scheduler.set_timesteps(steps, device=cond_latent.device)
    for i, ts in enumerate(scheduler.timesteps):
        sigma = scheduler.sigmas[i].expand(b).to(cond_latent.device, torch.float32)
        v = velocity(transformer, latents, cond_latent, cond_mask, sigma, text, action_D, action_3D)
        v = gt_velocity + v * (1 - cond_mask)
        if guidance > 0 and null_D is not None:
            vn = velocity(transformer, latents, cond_latent, cond_mask, sigma, text, null_D, null_3D)
            vn = gt_velocity + vn * (1 - cond_mask)
            v = v + guidance * (v - vn)
        latents = scheduler.step(v, ts, latents, return_dict=False)[0]
    return latents


def to_model_frames(images: np.ndarray, height: int, width: int) -> torch.Tensor:
    """(B,H,W,3) uint8 -> (B,3,H',W') in [-1,1], area-resized."""
    x = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 127.5 - 1
    return F.interpolate(x, size=(height, width), mode="area")


if __name__ == "__main__":
    if sys.argv[1:] == ["convert"]:
        convert()
    else:
        raise SystemExit(__doc__)

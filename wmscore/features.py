"""DINOv2-S/14 per-frame and R3D-18 clip features with the leaderboard's 320x512 letterbox."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

EVAL_H, EVAL_W = 320, 512
_IMAGENET = (torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225]))
_KINETICS = (torch.tensor([0.43216, 0.394666, 0.37645]), torch.tensor([0.22803, 0.22145, 0.216989]))


def letterbox_tensor(video: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """(T,H,W,3) uint8 -> (T,3,height,width) float in [0,1], bilinear, centred zero padding."""
    x = video.permute(0, 3, 1, 2).float() / 255.0
    h, w = x.shape[-2:]
    scale = min(height / h, width / w)
    rh, rw = max(1, round(h * scale)), max(1, round(w * scale))
    x = F.interpolate(x, size=(rh, rw), mode="bilinear", align_corners=False)
    top, left = (height - rh) // 2, (width - rw) // 2
    return F.pad(x, (left, width - rw - left, top, height - rh - top))


def to_eval_frames(video: np.ndarray) -> torch.Tensor:
    """Raw decoded video -> (T,3,320,512) in [0,1], quantised to uint8 like the leaderboard pipeline."""
    x = letterbox_tensor(torch.from_numpy(video), EVAL_H, EVAL_W)
    return (x.clamp(0, 1) * 255.0).to(torch.uint8).float() / 255.0


class FeatureModels:
    def __init__(self, device: torch.device):
        import timm
        from torchvision.models.video import R3D_18_Weights, r3d_18

        self.device = device
        self.dino = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0)
        self.dino.eval().to(device)
        self.dino_size = int(self.dino.patch_embed.img_size[0])
        self.r3d = r3d_18(weights=R3D_18_Weights.DEFAULT)
        self.r3d.fc = torch.nn.Identity()
        self.r3d.eval().to(device)

    @torch.no_grad()
    def dino_features(self, frames: torch.Tensor) -> torch.Tensor:
        """frames (T,3,320,512) in [0,1] -> (T,384)."""
        s = self.dino_size
        x = letterbox_tensor((frames.permute(0, 2, 3, 1) * 255).round().to(torch.uint8), s, s)
        mean, std = (t.view(1, 3, 1, 1) for t in _IMAGENET)
        x = ((x - mean) / std).to(self.device)
        return self.dino(x).float().cpu()

    @torch.no_grad()
    def r3d_features(self, frames: torch.Tensor) -> torch.Tensor:
        """frames (T,3,320,512) in [0,1] -> (512,)."""
        x = frames.permute(1, 0, 2, 3).unsqueeze(0)  # 1,3,T,H,W
        x = F.interpolate(x, size=(frames.shape[0], 112, 112), mode="trilinear", align_corners=False)
        mean, std = (t.view(1, 3, 1, 1, 1) for t in _KINETICS)
        x = ((x - mean) / std).to(self.device)
        return self.r3d(x).float().cpu()[0]


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """1 - cos over the flattened features."""
    return float(1.0 - F.cosine_similarity(a.flatten()[None], b.flatten()[None]).item())


def framewise_cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean over frames of 1 - cos, for (T,D) features."""
    return float((1.0 - F.cosine_similarity(a, b, dim=-1)).mean().item())

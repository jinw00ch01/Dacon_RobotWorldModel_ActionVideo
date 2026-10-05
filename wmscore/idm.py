"""Inverse dynamics model: 16 frames -> 16 z-normalised 6-D actions (trained on the train split only)."""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


class InverseDynamics(nn.Module):
    def __init__(self, pretrained: bool = True, hidden: int = 256):
        super().__init__()
        net = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.encoder = nn.Sequential(*list(net.children())[:-1])  # -> (N,512,1,1)
        self.temporal = nn.Conv1d(1024, hidden, kernel_size=3, padding=1)
        self.gru = nn.GRU(hidden, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Linear(2 * hidden, 6)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """frames (B,T,3,H,W) float in [0,1] -> (B,T,6)."""
        b, t = frames.shape[:2]
        x = (frames - _MEAN.to(frames)) / _STD.to(frames)
        f = self.encoder(x.flatten(0, 1)).flatten(1).view(b, t, -1)
        d = torch.cat([torch.zeros_like(f[:, :1]), f[:, 1:] - f[:, :-1]], dim=1)
        h = torch.relu(self.temporal(torch.cat([f, d], dim=-1).transpose(1, 2))).transpose(1, 2)
        h, _ = self.gru(h)
        return self.head(h)


def load_idm(path, device) -> InverseDynamics:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = InverseDynamics(pretrained=False)
    model.load_state_dict(ckpt["model"])
    return model.eval().to(device)

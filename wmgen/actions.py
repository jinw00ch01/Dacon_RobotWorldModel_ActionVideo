"""Action features for the generator: relative to the first action, step deltas, and absolute z-scores.

Eval robots have joint offsets the training data never shows (docs/DATA_ANALYSIS.md section 3),
so relative and delta features carry the motion while the absolute z-score is a weak hint.
"""
from __future__ import annotations

import numpy as np
import torch

from wmscore.data import load_action_stats

FEATURES_PER_STEP = 18


def action_features(actions: np.ndarray | torch.Tensor) -> torch.Tensor:
    """(..., 16, 6) raw joint targets -> (..., 16, 18) float32."""
    a = torch.as_tensor(actions, dtype=torch.float32)
    mean, std = (torch.from_numpy(x) for x in load_action_stats())
    rel = (a - a[..., :1, :]) / std
    delta = torch.diff(a, dim=-2, prepend=a[..., :1, :]) / std * 4.0  # per-step motion is small; rescale
    absolute = ((a - mean) / std).clamp(-4, 4) * 0.25
    return torch.cat([rel, delta, absolute], dim=-1)

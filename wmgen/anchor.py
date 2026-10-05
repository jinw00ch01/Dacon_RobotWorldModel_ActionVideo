"""Background anchoring: keep the input image's pixels wherever the generated frame did not change.

The camera is fixed and only the arm and the objects it touches move, but the generator works at
240x320 and is upsampled, and its colours drift slightly, so its static background is blurrier and
off-colour compared with the input. Per frame we:
1. fit a per-channel gain/bias that maps the generated frame onto the input (robust least squares on
   a downsampled copy; most pixels are static) and apply it,
2. mark where the colour-corrected frame still differs from the input (blurred difference at low
   resolution above a threshold, then dilated, with soft edges),
3. take the corrected generated pixels inside the mask and the input pixels outside it.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

WORK_SIZE = (160, 120)  # (w, h) for colour fitting and change detection


@dataclass(frozen=True)
class AnchorParams:
    threshold: float = 12.0  # mean abs RGB difference (0-255) at WORK_SIZE after blurring
    blur_sigma: float = 1.0  # pre-threshold blur at WORK_SIZE, pixels
    dilate: int = 12  # dilation radius at full resolution, pixels
    feather_sigma: float = 6.0  # soft edge of the mask at full resolution
    color_correct: bool = True
    accumulate: bool = False  # union the mask over time (keeps trails of the moving arm)


def _fit_color(src: np.ndarray, dst: np.ndarray, iters: int = 2, keep: float = 0.7) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel gain/bias with src*gain+bias ~= dst, refit on the best-matching pixels."""
    gain, bias = np.ones(3, np.float32), np.zeros(3, np.float32)
    s, d = src.reshape(-1, 3).astype(np.float32), dst.reshape(-1, 3).astype(np.float32)
    use = np.ones(len(s), bool)
    for _ in range(iters):
        for c in range(3):
            a = np.stack([s[use, c], np.ones(use.sum(), np.float32)], axis=1)
            sol, *_ = np.linalg.lstsq(a, d[use, c], rcond=None)
            gain[c], bias[c] = np.clip(sol[0], 0.8, 1.25), np.clip(sol[1], -30, 30)
        resid = np.abs(s * gain + bias - d).mean(axis=1)
        use = resid <= np.quantile(resid, keep)
    return gain, bias


def anchor_frame(frame: np.ndarray, image: np.ndarray, p: AnchorParams, prev_mask=None):
    """One generated frame -> (blended frame float32, binary mask at full resolution)."""
    h, w = image.shape[:2]
    small_img = cv2.resize(image, WORK_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)
    small_gen = cv2.resize(frame, WORK_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)
    gen = frame.astype(np.float32)
    if p.color_correct:
        gain, bias = _fit_color(small_gen, small_img)
        small_gen = small_gen * gain + bias
        gen = np.clip(gen * gain + bias, 0, 255)
    diff = np.abs(cv2.GaussianBlur(small_gen, (0, 0), p.blur_sigma) - cv2.GaussianBlur(small_img, (0, 0), p.blur_sigma))
    m = cv2.resize((diff.mean(axis=2) > p.threshold).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    if p.dilate > 0:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * p.dilate + 1, 2 * p.dilate + 1)))
    if p.accumulate and prev_mask is not None:
        m = np.maximum(m, prev_mask)
    alpha = m.astype(np.float32)
    if p.feather_sigma > 0:
        alpha = np.maximum(np.clip(cv2.GaussianBlur(alpha, (0, 0), p.feather_sigma) * 1.5, 0, 1), alpha)
    alpha = alpha[..., None]
    return alpha * gen + (1 - alpha) * image.astype(np.float32), m


def anchor(frames: np.ndarray, image: np.ndarray, p: AnchorParams = AnchorParams()) -> np.ndarray:
    """Blend generated frames onto the input image outside the motion mask. Frame 0 becomes the input."""
    if frames.shape[1:] != image.shape:
        raise ValueError(f"frames {frames.shape} vs image {image.shape}")
    out = np.empty_like(frames)
    out[0] = image
    mask = None
    for t in range(1, len(frames)):
        blended, mask = anchor_frame(frames[t], image, p, mask)
        out[t] = np.clip(blended.round(), 0, 255).astype(np.uint8)
    return out

"""Training-data access: episode frames, actions, and the holdout split."""
from __future__ import annotations

import json
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
TRAIN_ROOT = REPO / "open" / "data" / "train"
STATS_PATH = TRAIN_ROOT / "so100_action_statistics.json"
NUM_FRAMES = 16


def load_split(name: str = "holdout_v1") -> dict:
    return json.loads((REPO / "configs" / "splits" / f"{name}.json").read_text(encoding="utf-8"))


def load_action_stats() -> tuple[np.ndarray, np.ndarray]:
    stats = json.loads(STATS_PATH.read_text(encoding="utf-8"))
    mean = stats.get("mean", stats.get("action_mean"))
    std = stats.get("std", stats.get("action_std"))
    return np.asarray(mean, np.float32), np.asarray(std, np.float32)


def read_actions(parquet_rel: str) -> np.ndarray:
    table = pd.read_parquet(TRAIN_ROOT / parquet_rel, columns=["action"])
    return np.stack(table["action"].to_numpy()).astype(np.float32)


def decode_frames(video_rel_or_path, indices=None) -> np.ndarray:
    """Decode RGB frames (all, or the given sorted indices) with PyAV."""
    path = Path(video_rel_or_path)
    if not path.is_absolute():
        path = TRAIN_ROOT / path
    wanted = None if indices is None else set(int(i) for i in indices)
    last = None if indices is None else max(wanted)
    frames = {}
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for i, frame in enumerate(container.decode(stream)):
            if wanted is None or i in wanted:
                frames[i] = frame.to_ndarray(format="rgb24")
            if last is not None and i >= last:
                break
    keys = sorted(frames) if indices is None else list(indices)
    missing = [k for k in keys if k not in frames]
    if missing:
        raise IndexError(f"{path}: missing frames {missing[:5]}")
    return np.stack([frames[k] for k in keys], axis=0)


def letterbox(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    """Aspect-preserving resize into (height, width) with centred black padding, uint8 RGB."""
    h, w = frame.shape[:2]
    scale = min(height / h, width / w)
    rh, rw = max(1, round(h * scale)), max(1, round(w * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (rw, rh), interpolation=interp)
    out = np.zeros((height, width, 3), np.uint8)
    top, left = (height - rh) // 2, (width - rw) // 2
    out[top : top + rh, left : left + rw] = resized
    return out

"""Write and check submission mp4 files (16 RGB frames, H.264)."""
from __future__ import annotations

from pathlib import Path

import av
import numpy as np

NUM_FRAMES = 16


def write_mp4(frames: np.ndarray, path: Path, fps: int = 6, crf: int = 0) -> None:
    """frames: (T, H, W, 3) uint8 RGB. crf 0 with yuv444p is lossless up to colour conversion."""
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"expected (T,H,W,3) uint8, got {frames.shape} {frames.dtype}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.mp4")
    with av.open(str(tmp), mode="w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = frames.shape[2]
        stream.height = frames.shape[1]
        stream.pix_fmt = "yuv444p"
        stream.options = {"crf": str(crf), "preset": "medium"}
        for frame in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    tmp.replace(path)


def read_mp4(path: Path) -> np.ndarray:
    """Decode every frame the same way the submission kit does (PyAV, rgb24)."""
    with av.open(str(path)) as container:
        frames = [f.to_ndarray(format="rgb24") for f in container.decode(container.streams.video[0])]
    return np.stack(frames, axis=0)

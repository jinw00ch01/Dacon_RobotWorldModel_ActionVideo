import tempfile
import unittest
from pathlib import Path

import numpy as np

from wmgen.video_io import NUM_FRAMES, read_mp4, write_mp4


class VideoIoTest(unittest.TestCase):
    def test_roundtrip_keeps_16_frames_and_pixels(self):
        rng = np.random.default_rng(0)
        base = rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8)
        frames = np.repeat(base[None], NUM_FRAMES, axis=0)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "x.mp4"
            write_mp4(frames, path)
            out = read_mp4(path)
        self.assertEqual(out.shape, frames.shape)
        self.assertLess(np.abs(out.astype(np.int16) - frames).mean(), 2.0)


if __name__ == "__main__":
    unittest.main()

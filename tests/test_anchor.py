import unittest

import numpy as np

from wmgen.anchor import AnchorParams, anchor


class AnchorTest(unittest.TestCase):
    def test_static_background_restored_and_moving_block_kept(self):
        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(96, 128, 3), dtype=np.uint8)
        frames = np.repeat(image[None], 4, axis=0).astype(np.int16)
        frames[1:] += rng.integers(-4, 5, size=frames[1:].shape)  # small generator noise everywhere
        frames[2:, 40:60, 50:80] = 255  # a moving object appears
        frames = np.clip(frames, 0, 255).astype(np.uint8)
        out = anchor(frames, image, AnchorParams(dilate=3, feather_sigma=1.0))
        self.assertTrue(np.array_equal(out[0], image))
        self.assertTrue(np.array_equal(out[1, :20, :20], image[:20, :20]))  # noise removed far from motion
        self.assertGreater(out[3, 45:55, 55:75].min(), 240)  # moving object kept (colour-corrected)


if __name__ == "__main__":
    unittest.main()

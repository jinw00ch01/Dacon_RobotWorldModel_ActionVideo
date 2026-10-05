import unittest

import numpy as np
import torch

from wmscore.data import letterbox
from wmscore.features import EVAL_H, EVAL_W, framewise_cosine_distance, to_eval_frames
from wmscore.idm import InverseDynamics


class WmscoreTest(unittest.TestCase):
    def test_letterbox_640x480_into_eval_size(self):
        frame = np.full((480, 640, 3), 200, np.uint8)
        out = letterbox(frame, EVAL_H, EVAL_W)
        self.assertEqual(out.shape, (EVAL_H, EVAL_W, 3))
        self.assertEqual(out[:, :40].max(), 0)  # side bars (427 wide content, 42 px bars)
        self.assertEqual(out[:, 100:400].min(), 200)

    def test_eval_frames_shape(self):
        video = np.zeros((16, 480, 640, 3), np.uint8)
        self.assertEqual(tuple(to_eval_frames(video).shape), (16, 3, EVAL_H, EVAL_W))

    def test_identical_features_have_zero_distance(self):
        f = torch.randn(16, 384)
        self.assertAlmostEqual(framewise_cosine_distance(f, f), 0.0, places=5)

    def test_idm_output_shape(self):
        model = InverseDynamics(pretrained=False)
        out = model(torch.rand(2, 16, 3, 64, 104))
        self.assertEqual(tuple(out.shape), (2, 16, 6))


if __name__ == "__main__":
    unittest.main()

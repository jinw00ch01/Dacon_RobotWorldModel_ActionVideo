import unittest

import torch

from wmgen.cosmos_ac import ActionEmbedder, ActionTimeEmbed


class CosmosActionTest(unittest.TestCase):
    def test_embedder_pads_conditioning_latent(self):
        emb = ActionEmbedder(d_in=4 * 6, model_dim=8, hidden=16)
        d, d3 = emb(torch.randn(2, 16, 6))
        self.assertEqual(tuple(d.shape), (2, 5, 8))
        self.assertEqual(tuple(d3.shape), (2, 5, 24))
        self.assertEqual(float(d[:, 0].abs().sum()), 0.0)

    def test_time_embed_without_action_matches_base(self):
        from diffusers.models.transformers.transformer_cosmos import CosmosEmbedding

        base = CosmosEmbedding(8, 8)
        wrapped = ActionTimeEmbed(base)
        h = torch.zeros(1, 4, 8)
        t = torch.rand(6)
        for a, b in zip(base(h, t), wrapped(h, t)):
            self.assertTrue(torch.allclose(a, b))
        wrapped.action_D, wrapped.action_3D = torch.ones(2, 3, 8), torch.ones(2, 3, 24)
        temb, _ = wrapped(h, t)
        self.assertTrue(torch.allclose(temb, base(h, t)[0] + 1))


if __name__ == "__main__":
    unittest.main()

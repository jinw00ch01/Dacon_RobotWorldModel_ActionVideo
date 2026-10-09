import unittest

import torch

from wmgen.make_soup import soup


def fake_state(seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    return {"lora": {"blk.to_q.lora_A.weight": torch.randn(4, 8, generator=g),
                     "blk.to_q.lora_B.weight": torch.randn(6, 4, generator=g)},
            "embedder": {"fc1.weight": torch.randn(3, 2, generator=g)},
            "step": seed, "args": {"rank": 4}}


class MakeSoupTest(unittest.TestCase):
    def test_concat_is_exact_mean_of_updates(self):
        states = [fake_state(s) for s in (1, 2, 3)]
        out = soup(states, "concat")
        a, b = out["lora"]["blk.to_q.lora_A.weight"], out["lora"]["blk.to_q.lora_B.weight"]
        self.assertEqual(out["args"]["rank"], 12)
        self.assertEqual(tuple(a.shape), (12, 8))
        self.assertEqual(tuple(b.shape), (6, 12))
        want = sum(s["lora"]["blk.to_q.lora_B.weight"] @ s["lora"]["blk.to_q.lora_A.weight"] for s in states) / 3
        torch.testing.assert_close(b @ a, want, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(out["embedder"]["fc1.weight"],
                                   sum(s["embedder"]["fc1.weight"] for s in states) / 3)

    def test_mean_keeps_rank(self):
        states = [fake_state(s) for s in (1, 2)]
        out = soup(states, "mean")
        self.assertEqual(out["args"]["rank"], 4)
        torch.testing.assert_close(out["lora"]["blk.to_q.lora_A.weight"],
                                   (states[0]["lora"]["blk.to_q.lora_A.weight"] + states[1]["lora"]["blk.to_q.lora_A.weight"]) / 2)


if __name__ == "__main__":
    unittest.main()

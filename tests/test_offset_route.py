import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from wmgen.actions import action_features
from wmgen.c2_gate import cluster_ci, gate
from wmgen.offset_route import routed_joints
from wmscore.data import load_action_stats

REPO = Path(__file__).resolve().parents[1]


def clip(z_mean: np.ndarray, seed: int = 0) -> np.ndarray:
    """(16, 6) raw readings whose per-joint z-score averages to z_mean."""
    mean, std = load_action_stats()
    rng = np.random.default_rng(seed)
    z = z_mean + 0.2 * rng.standard_normal((16, 6))
    z -= z.mean(0) - z_mean
    return (z * std + mean).astype(np.float32)


class ActionFeaturesTest(unittest.TestCase):
    def test_default_and_empty_mask_unchanged(self):
        a = clip(np.zeros(6))
        base = action_features(a)
        torch.testing.assert_close(action_features(a, np.zeros(6, bool)), base)

    def test_rel_joints_are_offset_invariant(self):
        _, std = load_action_stats()
        a = clip(np.array([0.1, -0.3, 0.2, 0.0, 0.5, -0.1]))
        shifted = a.copy()
        shifted[:, [1, 2]] += -2.5 * std[[1, 2]]
        mask = np.array([False, True, True, False, False, False])
        f, g = action_features(a, mask), action_features(shifted, mask)
        torch.testing.assert_close(f, g)
        # unrouted joints keep their absolute channel, routed ones get the relative motion
        base = action_features(a)
        torch.testing.assert_close(f[:, [12, 15, 16, 17]], base[:, [12, 15, 16, 17]])
        torch.testing.assert_close(f[:, [13, 14]], f[:, [1, 2]] * 0.25)
        torch.testing.assert_close(f[:, :12], base[:, :12])


class RoutingTest(unittest.TestCase):
    env = {"lo": [-2.6, -1.56, -1.83, -2.2, -1.24, -0.62], "hi": [3.4, 0.81, 1.23, 1.44, 3.4, 1.86], "delta": 0.75}

    def test_in_envelope_not_routed(self):
        self.assertFalse(routed_joints(clip(np.zeros(6)), self.env).any())
        self.assertFalse(routed_joints(clip(np.array([0, 0, 0, 0, 1.89, 0])), self.env).any())

    def test_scene1_like_offset_routes_lift_and_elbow(self):
        r = routed_joints(clip(np.array([0.09, -2.61, -2.68, 0.05, 0.54, -0.09])), self.env)
        self.assertEqual(np.flatnonzero(r).tolist(), [1, 2])

    def test_envelope_file_matches_rule(self):
        env = json.loads((REPO / "configs" / "offset_envelope.json").read_text())
        self.assertEqual(len(env["lo"]), 6)
        self.assertLessEqual(len(env["holdout_routed"]) / env["holdout_windows"], 0.05)


class SimHoldoutTest(unittest.TestCase):
    def test_refuses_competition_eval(self):
        with tempfile.TemporaryDirectory() as t:
            r = subprocess.run([sys.executable, "-m", "wmgen.make_sim_holdout", "--src", str(REPO / "open" / "data" / "eval"),
                                "--out", t], capture_output=True, text=True, cwd=REPO)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("refusing", r.stderr + r.stdout)


class GateTest(unittest.TestCase):
    def write(self, d: Path, name: str, rows: dict) -> None:
        pd.DataFrame([{"sample_id": k, "user": f"u{int(k[1:]) % 4}", **v} for k, v in rows.items()]).to_csv(d / name, index=False)

    def test_gate_pass_and_fail(self):
        ids = [f"h{i:03d}" for i in range(20)]
        base = {i: {"dino": 0.2, "r3d": 0.06, "action": 0.9} for i in ids}
        better = {i: {"dino": 0.19, "r3d": 0.06, "action": 0.85 + 0.001 * k} for k, i in enumerate(ids)}
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            (d / "sim_routed.json").write_text(json.dumps({"routed": {i: [1, 2] for i in ids}}))
            for idm in ("idm_v2", "idm_v1"):
                self.write(d, f"sim_keep_{idm}.csv", base)
                self.write(d, f"sim_auto_{idm}.csv", better)
                self.write(d, f"route_keep_{idm}.csv", {i: base[i] for i in ids[:5]})
                self.write(d, f"route_auto_{idm}.csv", {i: base[i] for i in ids[:5]})
            v = gate(d, {})
            self.assertTrue(v["R1"]["pass"] and v["R2"]["pass"] and v["PASS"])
            for idm in ("idm_v2", "idm_v1"):
                self.write(d, f"sim_auto_{idm}.csv", {i: {"dino": 0.25, "r3d": 0.06, "action": 0.85} for i in ids})
            self.assertFalse(gate(d, {})["PASS"])  # visual got worse

    def test_cluster_ci_is_wider_with_correlated_groups(self):
        x = np.array([-0.05] * 10 + [0.03] * 10)
        g = np.array(["a"] * 10 + ["b"] * 10)
        lo, hi = cluster_ci(x, g)
        self.assertGreater(hi, 0)


if __name__ == "__main__":
    unittest.main()

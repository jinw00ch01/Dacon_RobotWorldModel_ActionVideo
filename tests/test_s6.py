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
from wmgen.offset_route import ENVELOPES, load_envelope, route_table, routed_joints
from wmgen.s6_gate import gate
from wmscore.data import load_action_stats

REPO = Path(__file__).resolve().parents[1]
HOLDOUT = Path(r"C:\Dacon\WM_Shared\holdout_v1")


def clip(z_mean, seed=0):
    mean, std = load_action_stats()
    rng = np.random.default_rng(seed)
    z = np.asarray(z_mean, float) + 0.2 * rng.standard_normal((16, 6))
    z -= z.mean(0) - np.asarray(z_mean, float)
    return (z * std + mean).astype(np.float32)


class S6RuleTest(unittest.TestCase):
    sub5, s6 = load_envelope(ENVELOPES["sub5"]), load_envelope(ENVELOPES["s6"])

    def test_s6_envelope_extends_sub5_only_by_lift_lower_margin(self):
        for k in ("lo", "hi", "delta"):
            self.assertEqual(self.s6[k], self.sub5[k])
        self.assertEqual(self.s6["lower_delta"], {"1": 0.0})
        self.assertNotIn("lower_delta", self.sub5)

    def test_without_lower_delta_rule_is_unchanged(self):
        rng = np.random.default_rng(0)
        lo, hi, d = np.array(self.sub5["lo"]), np.array(self.sub5["hi"]), self.sub5["delta"]
        for _ in range(300):
            a = clip(rng.uniform(-4, 4, 6), int(rng.integers(1 << 30)))
            from wmgen.offset_route import mean_z
            z = mean_z(a)
            np.testing.assert_array_equal(routed_joints(a, self.sub5), (z < lo - d) | (z > hi + d))

    def test_patterns(self):
        def r(z, env):
            return np.flatnonzero(routed_joints(clip(z), env)).tolist()
        base = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        cases = [([0, -1.7, 0, 0, 0, 0], [1], []), ([0, -1.7, -2.7, 0, 0, 0], [1, 2], [2]), ([0, -2.5, 0, 0, 0, 0], [1], [1]),
                 ([0, 1.0, 0, 0, 0, 0], [], []), ([0, 0, -2.0, 0, 0, 0], [], []), ([0, 0, 0, 0, 1.9, 0], [], []), (base, [], [])]
        for z, want_s6, want_sub5 in cases:
            self.assertEqual(r(z, self.s6), want_s6, z)
            self.assertEqual(r(z, self.sub5), want_sub5, z)

    def test_lift_rel_features_exactly_offset_invariant_in_bf16(self):
        _, std = load_action_stats()
        a = clip([0.1, -1.7, 0.3, 0.0, 0.5, -0.1])
        b = a.copy()
        b[:, 1] += -1.3 * std[1]
        mask = np.array([False, True, False, False, False, False])
        self.assertTrue(torch.equal(action_features(a, mask).to(torch.bfloat16), action_features(b, mask).to(torch.bfloat16)))

    @unittest.skipUnless(HOLDOUT.exists(), "holdout_v1 not available")
    def test_holdout_changes_only_hold_000003(self):
        a, b = route_table(HOLDOUT, self.sub5), route_table(HOLDOUT, self.s6)
        changed = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
        self.assertEqual(changed, ["hold_000003"])


class SimShiftsTest(unittest.TestCase):
    @unittest.skipUnless(HOLDOUT.exists(), "holdout_v1 not available")
    def test_per_window_shifts(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            shifts = {"hold_000001": [0, -10.0, 0, 0, 0, 0], "hold_000002": [0, 0, 5.0, 0, 0, 0]}
            (t / "s.json").write_text(json.dumps(shifts))
            subprocess.run([sys.executable, "-m", "wmgen.make_sim_holdout", "--src", str(HOLDOUT), "--out", str(t / "o"),
                            "--shifts", str(t / "s.json")], check=True, cwd=REPO, capture_output=True)
            self.assertEqual(sorted(p.stem for p in (t / "o" / "actions").glob("*.npy")), sorted(shifts))
            for i, s in shifts.items():
                d = np.load(t / "o" / "actions" / f"{i}.npy") - np.load(HOLDOUT / "actions" / f"{i}.npy")
                np.testing.assert_allclose(d, np.broadcast_to(np.asarray(s, np.float32), d.shape), atol=1e-4)
            r = subprocess.run([sys.executable, "-m", "wmgen.make_sim_holdout", "--src", str(HOLDOUT), "--out", str(t / "x"),
                                "--shifts", str(t / "s.json"), "--sigma", "1"], cwd=REPO, capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0)


class S6GateTest(unittest.TestCase):
    def write(self, path: Path, rows: dict) -> None:
        pd.DataFrame([{"sample_id": k, "user": f"u{int(k[-3:]) % 5}", **v} for k, v in rows.items()]).to_csv(path, index=False)

    def setup_dir(self, d: Path, lift_rows: dict, p1_rows: dict) -> dict:
        ids = [f"h{i:03d}" for i in range(64)]
        design = {"windows": {i: {"stratum": "P1" if k % 2 == 0 else "P2", "z_clean": [0, 0, 0, 0, 0, 0],
                                  "z_target": [0, -1.6 - k / 100, 0, 0, 0, 0]} for k, i in enumerate(ids)}}
        base = {i: {"dino": 0.25, "r3d": 0.07, "action": 0.6} for i in ids}
        for idm in ("idm_v2", "idm_v1"):
            self.write(d / f"band_auto_{idm}.csv", base)
            self.write(d / f"band_liftlo_{idm}.csv", lift_rows)
            self.write(d / f"p1clean_liftlo_{idm}.csv", p1_rows)
            self.write(d / f"h003_liftlo_{idm}.csv", {"hold_000003": {"dino": 0.2, "r3d": 0.06, "action": 0.4}})
            self.write(d / f"keep_full_{idm}.csv", {i: {"dino": 0.2, "r3d": 0.06, "action": 0.4} for i in ids})
            self.write(d / f"keep_sub64_{idm}.csv", {"hold_000003": {"dino": 0.2, "r3d": 0.06, "action": 0.4}})
        return design

    def run_gate(self, lift_rows, p1_rows):
        import wmgen.s6_gate as g
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            design = self.setup_dir(d, lift_rows, p1_rows)
            old = g.KEEP_FULL, g.KEEP_SUB64
            g.KEEP_FULL = {i: d / f"keep_full_{i}.csv" for i in ("idm_v2", "idm_v1")}
            g.KEEP_SUB64 = {i: d / f"keep_sub64_{i}.csv" for i in ("idm_v2", "idm_v1")}
            try:
                return gate(d, design)
            finally:
                g.KEEP_FULL, g.KEEP_SUB64 = old

    def test_pass_and_failures(self):
        ids = [f"h{i:03d}" for i in range(64)]
        good = {i: {"dino": 0.21 + 0.0005 * (k % 7), "r3d": 0.065, "action": 0.58} for k, i in enumerate(ids)}
        clean = {i: {"dino": 0.2, "r3d": 0.06, "action": 0.4} for i in ids}
        v = self.run_gate(good, clean)
        self.assertTrue(v["R1"]["pass"] and v["R2"]["pass"] and v["PASS"], json.dumps(v["R1"])[:300])
        small = {i: {"dino": 0.245, "r3d": 0.07, "action": 0.6} for i in ids}  # gain below the minimum effect
        self.assertFalse(self.run_gate(small, clean)["R1"]["pass"])
        p2_bad = {i: (good[i] if k % 2 == 0 else {"dino": 0.3, "r3d": 0.07, "action": 0.6}) for k, i in enumerate(ids)}
        self.assertFalse(self.run_gate(p2_bad, clean)["R1"]["pass"])  # stratum veto
        costly = {i: {"dino": 0.24, "r3d": 0.06, "action": 0.4} for i in ids}
        self.assertFalse(self.run_gate(good, costly)["R2"]["pass"])


if __name__ == "__main__":
    unittest.main()

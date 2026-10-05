import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from wm_ops import jobs, packets, syncthing, worker
from wm_ops.state import ledger, read_ledger, safe_path


class WmOpsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.ultra = self.config("ultra5060")
        self.pro = self.config("pro360")

    def config(self, role):
        root = self.base / role
        root.mkdir()
        return dict(version=1, role=role, project_root=str(root), exchange_root=str(self.base / "exchange"),
                    state_root=str(self.base / (role + "-state")), python="python")

    # Both configs share one exchange root, so each outbox is already the peer's inbox (Syncthing simulated).
    def result_body(self):
        return {"experiment_id": "exp-" + "a" * 32, "attempt_id": "a001", "status": "succeeded"}

    def test_round_trip_imports_once_and_acks(self):
        attach = self.base / "attach"
        (attach / "sub").mkdir(parents=True)
        (attach / "split.csv").write_text("dataset,split\nx,val\n", encoding="utf-8")
        (attach / "sub" / "metrics.json").write_text("{}", encoding="utf-8")
        packet_id, _ = packets.publish(self.ultra, "result", self.result_body(), attach)
        self.assertEqual(packets.import_inbox(self.pro), [packet_id])
        local = Path(self.pro["project_root"]) / "work/packets/inbox" / packet_id
        self.assertTrue((local / "files/split.csv").is_file())
        self.assertEqual(json.loads((local / "result.json").read_text(encoding="utf-8"))["sender"], "ultra5060")
        self.assertEqual(packets.import_inbox(self.pro), [])
        self.assertFalse(read_ledger(self.pro)["packets"][packet_id]["handled"])
        packets.import_inbox(self.ultra)
        self.assertEqual(read_ledger(self.ultra)["acks"][packet_id]["status"], "received")
        packets.mark_handled(self.pro, packet_id)
        self.assertTrue(read_ledger(self.pro)["packets"][packet_id]["handled"])

    def test_partial_sync_waits_and_tampering_quarantines(self):
        packet_id, _ = packets.publish(self.ultra, "result", self.result_body())
        body = self.base / "exchange/ultra_to_pro/experiments/v1" / packet_id / "result.json"
        saved = body.read_bytes()
        body.unlink()
        packets.import_inbox(self.pro)
        self.assertEqual(read_ledger(self.pro)["packets"][packet_id]["status"], "waiting")
        body.write_bytes(saved.replace(b"succeeded", b"failedxxx"))
        packets.import_inbox(self.pro)
        entry = read_ledger(self.pro)["packets"][packet_id]
        self.assertEqual(entry["status"], "quarantined")
        self.assertIn("hash mismatch", entry["detail"])

    def test_authority_and_attachment_rules(self):
        with self.assertRaises(ValueError):
            packets.publish(self.pro, "decision", {"experiment_id": "exp-" + "a" * 32, "decision": "accept", "reason": "x"})
        with self.assertRaises(ValueError):
            packets.publish(self.ultra, "qa", {"subject": "labels"})
        with self.assertRaises(ValueError):
            packets.publish(self.pro, "request", {"subject": ""})
        bad = self.base / "bad"
        bad.mkdir()
        (bad / "run.ps1").write_text("whoami", encoding="utf-8")
        with self.assertRaises(ValueError):
            packets.publish(self.pro, "data", {"subject": "split"}, bad)
        self.assertFalse(any((self.base / "exchange/pro_to_ultra/experiments/v1").glob(".staging-*")))

    def test_forged_sender_is_quarantined(self):
        packet_id, _ = packets.publish(self.pro, "qa", {"subject": "contact sheets"})
        source = self.base / "exchange/pro_to_ultra/experiments/v1" / packet_id
        forged = self.base / "exchange/ultra_to_pro/experiments/v1" / packet_id
        forged.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, forged)
        packets.import_inbox(self.pro)
        self.assertEqual(read_ledger(self.pro)["packets"][packet_id]["status"], "quarantined")

    def test_paths_reject_traversal_and_windows_special_names(self):
        for name in ("../a", "/absolute", "data/../secret", "data\\file", "data/a:stream", "data/CON.txt", "data/a.", "data//a"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                safe_path(self.base, name)

    def test_job_rules_and_launch_command(self):
        with self.assertRaises(ValueError):
            jobs.request(self.pro, "gpu", 60, "train", ["python", "a.py"])
        with self.assertRaises(ValueError):
            jobs.request(self.ultra, "gpu", 60, "train", ["python", "a.py"])  # temp dir is not a clean git tree
        spec = jobs.request(self.pro, "cpu", 60, "metrics", ["python", "b.py"])
        self.assertEqual(read_ledger(self.pro)["jobs"][spec["job_id"]]["status"], "queued")
        with patch.object(jobs.subprocess, "Popen") as popen:
            popen.return_value.pid = 4242
            self.assertEqual(jobs.launch_queued(self.pro, "cfg.json"), [spec["job_id"]])
        argv = popen.call_args.args[0]
        self.assertEqual(argv[1:5], ["-m", "wm_ops", "--config", "cfg.json"])
        self.assertEqual(argv[5:], ["job-run", "--job", spec["job_id"]])
        self.assertEqual(read_ledger(self.pro)["jobs"][spec["job_id"]]["status"], "running")

    def test_only_one_gpu_job_runs_at_a_time(self):
        with patch.object(jobs, "tree_clean", return_value=True), patch.object(jobs, "head_commit", return_value="a" * 40):
            first = jobs.request(self.ultra, "gpu", 60, "a", ["python", "a.py"])
            second = jobs.request(self.ultra, "gpu", 60, "b", ["python", "b.py"])
        with patch.object(jobs.subprocess, "Popen") as popen:
            popen.return_value.pid = 1
            self.assertEqual(jobs.launch_queued(self.ultra, "cfg.json"), [first["job_id"]])
            self.assertEqual(jobs.launch_queued(self.ultra, "cfg.json"), [])
        self.assertEqual(read_ledger(self.ultra)["jobs"][second["job_id"]]["status"], "queued")

    def test_worker_tick_imports_and_writes_heartbeat(self):
        packet_id, _ = packets.publish(self.ultra, "request", {"subject": "build the holdout split"})
        with patch.object(worker, "git") as git:
            git.return_value.stdout = "abc1234"
            result = worker.tick(self.pro, str(self.base / "cfg.json"))
        self.assertEqual(result["imported"], [packet_id])
        self.assertEqual(result["errors"], [])
        beat = json.loads((self.base / "exchange/pro_to_ultra/status/pro360.json").read_text(encoding="utf-8"))
        self.assertEqual(beat["unhandled_packets"], [packet_id])
        self.assertIsNotNone(read_ledger(self.pro)["heartbeat_utc"])

    def test_reconcile_peer_pairs_once_from_nodes_file(self):
        config_path = self.base / "local-node.json"
        cfg = {**self.ultra, "syncthing_home": str(self.base / "st"), "peer_device_id": None}
        config_path.write_text(json.dumps(cfg), encoding="utf-8")
        nodes = Path(cfg["project_root"]) / "configs" / "nodes.json"
        nodes.parent.mkdir(parents=True)
        peer = "-".join(["ABCDEFG"] * 8)
        nodes.write_text(json.dumps({"version": 1, "devices": {"ultra5060": "-".join(["BCDEFGH"] * 8), "pro360": None}}), encoding="utf-8")
        with patch.object(worker.syncthing, "configure") as configure:
            self.assertIsNone(worker.reconcile_peer(cfg, config_path))
            nodes.write_text(json.dumps({"version": 1, "devices": {"pro360": peer}}), encoding="utf-8")
            worker.reconcile_peer(cfg, config_path)
            worker.reconcile_peer(cfg, config_path)
        self.assertEqual(configure.call_count, 1)
        self.assertEqual(configure.call_args.args[3], peer)
        self.assertEqual(json.loads(config_path.read_text(encoding="utf-8"))["peer_device_id"], peer)

    def _node(self, cfg, **extra):
        config_path = self.base / (cfg["role"] + "-local-node.json")
        cfg = {**cfg, "syncthing_home": str(self.base / "st"), "peer_device_id": None, **extra}
        config_path.write_text(json.dumps(cfg), encoding="utf-8")
        return cfg, config_path

    def test_join_code_accepts_only_the_matching_pending_device(self):
        pro_id, stranger = "-".join(["PROPROP"] * 8), "-".join(["STRANGE"] * 8)
        cfg, path = self._node(self.ultra, accept_token="ABCD2345", accept_token_expires_utc="2099-01-01T00:00:00Z")
        pending = {stranger: {"name": "WM-pro360-WRONG234"}, pro_id: {"name": "WM-pro360-ABCD2345"}}
        with patch.object(worker.syncthing, "api", return_value=pending), patch.object(worker.syncthing, "configure") as configure:
            self.assertEqual(worker.pairing_step(cfg, path), {"accepted_peer": pro_id})
            self.assertIsNone(worker.pairing_step(cfg, path))  # the code is single use
        self.assertEqual(configure.call_count, 1)
        self.assertEqual(configure.call_args.args[3], pro_id)
        saved = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(saved["peer_device_id"], pro_id)
        self.assertNotIn("accept_token", saved)

    def test_join_code_expires_and_unmatched_pending_is_ignored(self):
        cfg, path = self._node(self.ultra, accept_token="ABCD2345", accept_token_expires_utc="2099-01-01T00:00:00Z")
        with patch.object(worker.syncthing, "api", return_value={"-".join(["STRANGE"] * 8): {"name": "WM-pro360"}}), \
                patch.object(worker.syncthing, "configure") as configure:
            self.assertIsNone(worker.pairing_step(cfg, path))
        configure.assert_not_called()
        cfg, path = self._node(self.ultra, accept_token="ABCD2345", accept_token_expires_utc="2000-01-01T00:00:00Z")
        self.assertEqual(worker.pairing_step(cfg, path), {"accept_token": "expired"})
        self.assertNotIn("accept_token", json.loads(path.read_text(encoding="utf-8")))

    def test_joiner_announces_code_until_connected(self):
        ultra_id = "-".join(["ULTRAUL"] * 8)
        cfg, path = self._node(self.pro, join_token="ABCD2345", peer_device_id=ultra_id)
        with patch.object(worker.syncthing, "configure") as configure:
            worker.apply_config(cfg)
            self.assertEqual(configure.call_args.kwargs["self_name"], "WM-pro360-ABCD2345")
            with patch.object(worker.syncthing, "api", return_value={"connections": {ultra_id: {"connected": False}}}):
                self.assertIsNone(worker.pairing_step(cfg, path))
            with patch.object(worker.syncthing, "api", return_value={"connections": {ultra_id: {"connected": True}}}):
                self.assertEqual(worker.pairing_step(cfg, path), {"joined_peer": ultra_id})
            self.assertIsNone(configure.call_args.kwargs["self_name"])
        self.assertNotIn("join_token", json.loads(path.read_text(encoding="utf-8")))

    def test_syncthing_folder_plan_and_ignores(self):
        plan = {fid: kind for fid, _, _, kind in syncthing.folder_plan("pro360", self.base / "x", self.base / "open")}
        self.assertEqual(plan, {"wm-ultra-to-pro-v1": "receiveonly", "wm-pro-to-ultra-v1": "sendonly", "wm-data-v1": "receiveonly"})
        plan = {fid: kind for fid, _, _, kind in syncthing.folder_plan("ultra5060", self.base / "x")}
        self.assertEqual(plan, {"wm-ultra-to-pro-v1": "sendonly", "wm-pro-to-ultra-v1": "receiveonly"})
        self.assertIn("/data/train/*/*/videos", syncthing.data_ignore_lines("pro360", True))
        self.assertNotIn("/data/train/*/*/videos", syncthing.data_ignore_lines("ultra5060", True))

    def test_ledger_survives_concurrent_style_updates(self):
        with ledger(self.pro) as book:
            book["jobs"]["x"] = {"status": "queued", "kind": "cpu", "requested_utc": "2026-10-05T00:00:00Z"}
        with ledger(self.pro) as book:
            self.assertIn("x", book["jobs"])


if __name__ == "__main__":
    unittest.main()

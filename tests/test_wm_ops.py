import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from wm_ops import jobs, worker
from wm_ops.state import ledger, read_ledger


class WmOpsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        root = self.base / "project"
        root.mkdir()
        self.cfg = dict(version=1, role="ultra5060", project_root=str(root), state_root=str(self.base / "state"),
                        python="python")

    def test_job_rules_and_launch_command(self):
        with self.assertRaises(ValueError):  # a temp dir is not a clean git checkout
            jobs.request(self.cfg, "gpu", 60, "train", ["python", "a.py"], workdir=self.cfg["project_root"])
        with self.assertRaises(ValueError):
            jobs.request({**self.cfg, "role": "other"}, "gpu", 60, "train", ["python", "a.py"], workdir=self.cfg["project_root"])
        spec = jobs.request(self.cfg, "cpu", 60, "metrics", ["python", "b.py"], workdir=self.cfg["project_root"])
        self.assertEqual(spec["workdir"], str(Path(self.cfg["project_root"])))
        self.assertEqual(read_ledger(self.cfg)["jobs"][spec["job_id"]]["status"], "queued")
        with patch.object(jobs.subprocess, "Popen") as popen:
            popen.return_value.pid = 4242
            self.assertEqual(jobs.launch_queued(self.cfg, "cfg.json"), [spec["job_id"]])
        argv = popen.call_args.args[0]
        self.assertEqual(argv[1:5], ["-m", "wm_ops", "--config", "cfg.json"])
        self.assertEqual(argv[5:], ["job-run", "--job", spec["job_id"]])
        self.assertTrue(popen.call_args.kwargs["creationflags"] & 0x08000000)  # CREATE_NO_WINDOW
        self.assertFalse(popen.call_args.kwargs["creationflags"] & 0x00000008)  # never DETACHED_PROCESS
        self.assertEqual(read_ledger(self.cfg)["jobs"][spec["job_id"]]["status"], "running")

    def test_job_runs_in_the_requesting_checkout_and_starts_the_next_one(self):
        clone = self.base / "agent-clone"
        clone.mkdir()
        with patch.object(jobs, "checkout_root", return_value=str(clone)), \
                patch.object(jobs, "tree_clean", return_value=True), patch.object(jobs, "head_commit", return_value="c" * 40):
            spec = jobs.request(self.cfg, "gpu", 60, "infer", ["python", "-m", "x"])
            nxt = jobs.request(self.cfg, "gpu", 60, "next", ["python", "-m", "y"])
        with ledger(self.cfg) as book:
            book["jobs"][spec["job_id"]]["status"] = "running"
        self.assertEqual((spec["workdir"], spec["code_commit"]), (str(clone), "c" * 40))
        completed = jobs.subprocess.CompletedProcess([], 0, '{"status": "passed", "run": "r"}', "")
        with patch.object(jobs.subprocess, "run", return_value=completed) as run, \
                patch.object(jobs.subprocess, "Popen") as popen:
            popen.return_value.pid = 7
            self.assertEqual(jobs.run(self.cfg, spec["job_id"], "cfg.json"), 0)
        self.assertEqual(run.call_args.kwargs["cwd"], str(clone))
        book = read_ledger(self.cfg)
        self.assertEqual(book["jobs"][spec["job_id"]]["status"], "succeeded")
        self.assertEqual(book["jobs"][nxt["job_id"]]["status"], "running")  # chained without waiting for a poll

    def test_only_one_gpu_job_runs_at_a_time(self):
        with patch.object(jobs, "tree_clean", return_value=True), patch.object(jobs, "head_commit", return_value="a" * 40):
            first = jobs.request(self.cfg, "gpu", 60, "a", ["python", "a.py"])
            second = jobs.request(self.cfg, "gpu", 60, "b", ["python", "b.py"])
        with patch.object(jobs.subprocess, "Popen") as popen:
            popen.return_value.pid = 1
            self.assertEqual(jobs.launch_queued(self.cfg, "cfg.json"), [first["job_id"]])
            self.assertEqual(jobs.launch_queued(self.cfg, "cfg.json"), [])
        self.assertEqual(read_ledger(self.cfg)["jobs"][second["job_id"]]["status"], "queued")

    def test_stale_gpu_lock_is_replaced_but_a_live_one_blocks(self):
        import os
        from harness import __main__ as harness
        lock = self.base / "gpu.lock"
        lock.write_text(json.dumps({"supervisor_pid": 999999999}), encoding="utf-8")  # no such process
        harness.acquire_gpu_lock(lock, self.base / "run")
        self.assertEqual(json.loads(lock.read_text(encoding="utf-8"))["supervisor_pid"], os.getpid())
        with self.assertRaises(RuntimeError):  # held by this (live) process
            harness.acquire_gpu_lock(lock, self.base / "run2")

    def test_tick_launches_and_writes_heartbeat(self):
        spec = jobs.request(self.cfg, "cpu", 60, "metrics", ["python", "b.py"], workdir=self.cfg["project_root"])
        with patch.object(jobs.subprocess, "Popen") as popen:
            popen.return_value.pid = 5
            result = worker.tick(self.cfg, str(self.base / "cfg.json"))
        self.assertEqual(result["launched_jobs"], [spec["job_id"]])
        self.assertEqual(result["errors"], [])
        self.assertIsNotNone(read_ledger(self.cfg)["heartbeat_utc"])

    def test_serve_exits_on_stop_file(self):
        import sys
        config = self.base / "local-node.json"
        config.write_text(json.dumps({**self.cfg, "python": sys.executable}), encoding="utf-8")
        (Path(self.cfg["state_root"])).mkdir(parents=True, exist_ok=True)
        (Path(self.cfg["state_root"]) / "STOP").write_text("x", encoding="utf-8")
        self.assertEqual(worker.serve(str(config)), 0)


if __name__ == "__main__":
    unittest.main()

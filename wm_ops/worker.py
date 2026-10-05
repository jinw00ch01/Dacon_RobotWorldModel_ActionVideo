"""The job runner: `python -m wm_ops serve`, run windowless (pythonw) by the Task Scheduler task
WM-Jobs-ultra5060. Every poll it marks dead jobs lost, launches queued jobs (one GPU job at a time)
and fast-forwards a clean main checkout. It never starts Claude and never opens a window.
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
import time
import traceback

from . import jobs
from .state import atomic_json, file_lock, git, ledger, load_config, ops_root, utc_text

ORIGIN = "https://github.com/jinw00ch01/Dacon_RobotWorldModel_ActionVideo.git"
GIT_EVERY_SECONDS = 300


def sync_code(cfg, state):
    """Fast-forward a clean main checkout every 5 minutes; never merge, rebase or discard local work."""
    if not cfg.get("auto_git") or time.time() - state.get("last_git_check", 0) < GIT_EVERY_SECONDS:
        return
    state["last_git_check"] = time.time()
    if git(cfg, "status", "--porcelain").stdout.strip():
        state["git_status"] = "local_changes_preserved"
        return
    origin = git(cfg, "remote", "get-url", "origin").stdout.strip()
    if origin.rstrip("/").removesuffix(".git") != ORIGIN.removesuffix(".git") or git(cfg, "branch", "--show-current").stdout.strip() != "main":
        state["git_status"] = "not_on_main_or_unexpected_origin"
        return
    if git(cfg, "-c", "credential.interactive=never", "fetch", "origin", "main", timeout=60).returncode:
        state["git_status"] = "fetch_failed"
        return
    merged = git(cfg, "merge", "--ff-only", "origin/main", timeout=60)
    state["git_status"] = "current_or_fast_forwarded" if merged.returncode == 0 else "divergence_preserved"


def tick(cfg, config_path):
    """One non-blocking iteration under the runner lock. Returns a small status dict for the log."""
    root = ops_root(cfg)
    with file_lock(root / "runner.lock", wait_seconds=1):
        state_path = root / "runner-state.json"
        state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
        result = {"lost_jobs": [], "launched_jobs": [], "errors": []}
        for name, action in (("lost_jobs", lambda: jobs.monitor(cfg)),
                             ("launched_jobs", lambda: jobs.launch_queued(cfg, config_path)),
                             ("git", lambda: sync_code(cfg, state))):
            try:
                value = action()
                if name in result:
                    result[name] = value
            except Exception as error:  # one failing step must not stop the others
                result["errors"].append(f"{name}: {type(error).__name__}: {error}")
        with ledger(cfg) as book:
            book["heartbeat_utc"] = utc_text()
        state["last_tick_utc"] = utc_text()
        atomic_json(state_path, state)
        return result


def _code_stamp():
    return sorted((p.name, p.stat().st_mtime_ns) for p in Path(__file__).parent.glob("*.py"))


def _reload():
    for name in ("wm_ops.state", "wm_ops.jobs", "wm_ops.worker"):
        importlib.reload(sys.modules[name])


def serve(config_path):
    """Loop until <state_root>/STOP exists. Picks up new wm_ops code (git fast-forward) between polls."""
    stamp = _code_stamp()
    log = None
    while True:
        cfg = load_config(config_path)
        root = ops_root(cfg)
        log = root / "runner.log"
        if (Path(cfg["state_root"]) / "STOP").exists():
            _log(log, "STOP file found; exiting")
            return 0
        try:
            if _code_stamp() != stamp:
                stamp = _code_stamp()
                _reload()
                _log(log, "reloaded wm_ops code")
            result = sys.modules["wm_ops.worker"].tick(cfg, config_path)
            if result["launched_jobs"] or result["lost_jobs"] or result["errors"]:
                _log(log, json.dumps(result, ensure_ascii=False))
        except Exception as error:  # keep the runner alive; the log is the evidence
            _log(log, f"runner error: {type(error).__name__}: {error}\n{traceback.format_exc()}")
        time.sleep(cfg.get("poll_seconds", 15))


def _log(path, message):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(f"{utc_text()} {message}\n")

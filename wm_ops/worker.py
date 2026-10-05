"""One supervisor tick, run every poll by scripts/start_exchange.ps1 (a Task Scheduler task).

import packets -> watch jobs -> launch queued jobs -> fast-forward main -> pair the peer listed in
configs/nodes.json -> publish a heartbeat the peer can read. Never starts Claude.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import time

from . import jobs, packets, syncthing
from .state import ROLES, atomic_json, dirs, file_lock, git, ledger, ops_root, read_ledger, strict_load, utc_text

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


def nodes_file(cfg):
    return Path(cfg["project_root"]) / "configs" / "nodes.json"


def reconcile_peer(cfg, config_path):
    """Pair with the peer device listed in configs/nodes.json (git is the trust channel for device IDs)."""
    path = nodes_file(cfg)
    if not path.exists() or not cfg.get("syncthing_home"):
        return None
    peer_role = next(r for r in ROLES if r != cfg["role"])
    peer_id = (strict_load(path).get("devices") or {}).get(peer_role)
    if not peer_id or peer_id == cfg.get("peer_device_id"):
        return None
    result = syncthing.configure(cfg["syncthing_home"], cfg["role"], cfg["exchange_root"], peer_id,
                                 cfg.get("data_root") if cfg.get("share_data") else None,
                                 cfg.get("skip_train_videos", False), cfg.get("listen_port", 22010))
    cfg["peer_device_id"] = peer_id
    raw = json.loads(Path(config_path).read_text(encoding="utf-8-sig"))
    raw["peer_device_id"] = peer_id
    Path(config_path).write_text(json.dumps(raw, indent=2), encoding="utf-8")
    return result


def heartbeat(cfg, state):
    outbox, _, _ = dirs(cfg)
    book = read_ledger(cfg)
    counts = {}
    for job in book["jobs"].values():
        counts[job["status"]] = counts.get(job["status"], 0) + 1
    atomic_json(outbox / "status" / f"{cfg['role']}.json", {
        "role": cfg["role"], "updated_utc": utc_text(), "git_status": state.get("git_status"),
        "git_head": git(cfg, "rev-parse", "--short", "HEAD").stdout.strip(),
        "jobs": counts, "unhandled_packets": sorted(p for p, e in book["packets"].items() if e.get("status") == "imported" and not e.get("handled")),
        "disk_free_gib": round(shutil.disk_usage(cfg["project_root"]).free / 1024**3, 1),
    })


def tick(cfg, config_path):
    """One non-blocking iteration under the worker lock. Returns a small status dict for the log."""
    root = ops_root(cfg)
    with file_lock(root / "worker.lock", wait_seconds=1):
        state_path = root / "worker-state.json"
        state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
        result = {"imported": [], "lost_jobs": [], "launched_jobs": [], "paired": None, "errors": []}
        for name, action in (("imported", lambda: packets.import_inbox(cfg)), ("lost_jobs", lambda: jobs.monitor(cfg)),
                             ("launched_jobs", lambda: jobs.launch_queued(cfg, config_path)),
                             ("git", lambda: sync_code(cfg, state)), ("paired", lambda: reconcile_peer(cfg, config_path)),
                             ("heartbeat", lambda: heartbeat(cfg, state))):
            try:
                value = action()
                if name in result:
                    result[name] = value
            except Exception as error:  # one failing step must not stop the others
                result["errors"].append(f"{name}: {type(error).__name__}: {error}")
        with ledger(cfg) as book:
            book["heartbeat_utc"] = utc_text()
        state["last_tick_utc"] = utc_text()
        state["git_status"] = state.get("git_status")
        atomic_json(state_path, state)
        if os.environ.get("WM_TICK_VERBOSE"):
            print(json.dumps(result, ensure_ascii=False))
        return result

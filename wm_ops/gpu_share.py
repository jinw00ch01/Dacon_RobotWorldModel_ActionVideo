"""Sharing this PC's one GPU with another project (the runner side of wm_ops/gpu_turn.py).

While <share>/gpu.request names a live process, every runner poll suspends the workload processes of our
running GPU job (not the wm_ops/harness supervisors), holds queued GPU jobs, and writes <share>/gpu.granted
for that pid. When the request is gone (or its process died) the same processes are resumed exactly once.
Nothing is killed, so the job continues where it stopped; its harness timeout keeps counting while paused.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

from .state import atomic_json, ops_root, read_ledger, utc_text

STALE_UNREADABLE_SECONDS = 60


def share_dir(cfg) -> Path:
    return Path(cfg.get("gpu_share_dir") or os.environ.get("WM_GPU_SHARE_DIR") or "C:/Dacon/WM_Runtime")


def requested(cfg):
    """The live outside request, or None. Requests whose process has exited are removed."""
    import psutil
    path = share_dir(cfg) / "gpu.request"
    try:
        request = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError):  # being written right now, or garbage left behind
        try:
            if time.time() - path.stat().st_mtime > STALE_UNREADABLE_SECONDS:
                path.unlink(missing_ok=True)
                return None
        except OSError:
            return None
        return {"pid": None, "who": "?"}
    pid = request.get("pid")
    if not pid or not psutil.pid_exists(pid):
        path.unlink(missing_ok=True)
        return None
    return request


def _workload(job):
    """Processes under the job's wrapper that do the work (skips the wm_ops / harness supervisors)."""
    import psutil
    try:
        family = psutil.Process(job["wrapper_pid"]).children(recursive=True)
    except (psutil.Error, KeyError, TypeError):
        return []
    work = []
    for process in family:
        try:
            line = " ".join(process.cmdline())
            if "-m wm_ops" in line or "-m harness" in line:
                continue
            work.append(process)
        except psutil.Error:
            pass
    return work


def tick(cfg):
    """Pause our GPU job for an outside request, or resume it when the request is gone."""
    import psutil
    state_path = ops_root(cfg) / "gpu-share.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    suspended = state.get("suspended", [])  # [[pid, create_time], ...]
    request = requested(cfg)
    grant = share_dir(cfg) / "gpu.granted"
    if request:
        known = {pid for pid, _ in suspended}
        paused = []
        for job in read_ledger(cfg)["jobs"].values():
            if job["kind"] != "gpu" or job["status"] not in {"starting", "running"}:
                continue
            paused.append(job["name"])
            for process in _workload(job):
                if process.pid in known:
                    continue
                try:
                    created = process.create_time()
                    process.suspend()
                    suspended.append([process.pid, created])
                except psutil.Error:
                    pass
        changed = suspended != state.get("suspended", [])
        if not state.get("since_utc"):
            state["since_utc"], changed = utc_text(), True
        state.update(suspended=suspended, who=request.get("who"))
        if changed:
            atomic_json(state_path, state)
        if request.get("pid"):
            current = None
            try:
                current = json.loads(grant.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
            if not current or current.get("pid") != request["pid"]:
                atomic_json(grant, {"pid": request["pid"], "who": request.get("who"), "granted_utc": utc_text(),
                                    "paused_jobs": paused})
        return {"turn": request.get("who"), "paused_jobs": paused} if changed else None
    for pid, created in suspended:
        try:
            process = psutil.Process(pid)
            if abs(process.create_time() - created) < 1:
                process.resume()
        except psutil.Error:
            pass
    if state:
        state_path.unlink(missing_ok=True)
    try:
        grant.unlink(missing_ok=True)
    except PermissionError:  # the other side has it open this instant; the next poll removes it
        pass
    return {"turn": None, "resumed": len(suspended)} if suspended else None

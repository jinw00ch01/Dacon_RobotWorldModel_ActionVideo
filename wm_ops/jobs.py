"""Detached long jobs (training, inference, evaluation) outside any Claude process (ported from AFDA).

A Claude session queues a job with `python -m wm_ops job start ...`; the job runner (Task Scheduler
task WM-Jobs-ultra5060, a windowless pythonw loop outside the Claude app) launches a wrapper that runs
`harness execute` and writes done.json. The job therefore survives the end of the session or a restart
of the Claude app. It runs in the checkout it was requested from (main folder or an agent clone), at the
commit recorded at request time. No process here opens a console window.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

from . import gpu_share
from .state import (NO_WINDOW, atomic_json, checkout_root, head_commit, ledger, ops_root, parse_utc, read_ledger,
                    tree_clean, utc_text)

# CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW. Not DETACHED_PROCESS: a venv python.exe is a launcher that
# starts the real interpreter, which would then get a new (visible, Windows Terminal) console.
DETACHED = 0x00000200 | NO_WINDOW
BREAKAWAY = 0x01000000


def job_dir(cfg, job_id):
    return ops_root(cfg) / "jobs" / job_id


def request(cfg, kind, timeout, name, command, allow_dirty=False, workdir=None):
    if kind not in {"cpu", "gpu"} or timeout <= 0 or not command:
        raise ValueError("Need --kind cpu|gpu, a positive --timeout and a command after --")
    if kind == "gpu" and cfg["role"] != "ultra5060":
        raise ValueError("GPU jobs run only on Ultra")
    workdir = checkout_root(cfg, workdir)
    commit = head_commit(cfg, workdir)
    clean = tree_clean(cfg, workdir)
    if kind == "gpu" and not clean and not allow_dirty:
        raise ValueError("Commit your code first: GPU jobs record a clean code commit (or pass --allow-dirty for smoke tests)")
    job_id = uuid.uuid4().hex[:12]
    folder = job_dir(cfg, job_id)
    folder.mkdir(parents=True)
    spec = {"job_id": job_id, "name": name, "kind": kind, "timeout": timeout, "command": list(command),
            "workdir": workdir, "code_commit": commit, "tree_clean": clean, "requested_utc": utc_text()}
    atomic_json(folder / "request.json", spec)
    with ledger(cfg) as book:
        book["jobs"][job_id] = {**spec, "status": "queued"}
    return spec


def _running_gpu(book):
    return [j for j in book["jobs"].values() if j["kind"] == "gpu" and j["status"] in {"starting", "running"}]


def launch_queued(cfg, config_path):
    """Called by the supervisor tick: start queued jobs, one GPU job at a time."""
    started = []
    with ledger(cfg) as book:
        queued = sorted((j for j in book["jobs"].values() if j["status"] == "queued"), key=lambda j: j["requested_utc"])
        for job in queued:
            if job["kind"] == "gpu" and (_running_gpu(book) or gpu_share.requested(cfg)):
                continue  # one GPU job at a time, and none while another project holds a GPU turn
            folder = job_dir(cfg, job["job_id"])
            argv = [cfg["python"], "-m", "wm_ops", "--config", str(config_path), "job-run", "--job", job["job_id"]]
            with (folder / "wrapper.log").open("ab") as log:
                flags = DETACHED if os.name == "nt" else 0
                try:
                    process = subprocess.Popen(argv, cwd=cfg["project_root"], stdout=log, stderr=subprocess.STDOUT,
                                               stdin=subprocess.DEVNULL, creationflags=flags | (BREAKAWAY if os.name == "nt" else 0),
                                               start_new_session=os.name != "nt")
                except OSError:
                    process = subprocess.Popen(argv, cwd=cfg["project_root"], stdout=log, stderr=subprocess.STDOUT,
                                               stdin=subprocess.DEVNULL, creationflags=flags, start_new_session=os.name != "nt")
            job.update(status="running", wrapper_pid=process.pid, started_utc=utc_text())
            started.append(job["job_id"])
    return started


def run(cfg, job_id, config_path=None):
    """Wrapper process body: supervise the job through harness execute, record done.json, then start
    whatever is queued next (so a GPU job queued behind this one starts without waiting for a poll)."""
    folder = job_dir(cfg, job_id)
    spec = json.loads((folder / "request.json").read_text(encoding="utf-8"))
    argv = [cfg["python"], "-m", "harness", "execute", "--profile", cfg["role"], "--kind", spec["kind"],
            "--timeout", str(spec["timeout"]), "--name", spec["name"], "--", *spec["command"]]
    with ledger(cfg) as book:
        book["jobs"][job_id].update(status="running", wrapper_pid=os.getpid(),
                                    started_utc=book["jobs"][job_id].get("started_utc") or utc_text())
    workdir = spec.get("workdir") or cfg["project_root"]
    completed = subprocess.run(argv, cwd=workdir, capture_output=True, text=True, encoding="utf-8",
                               errors="replace", creationflags=NO_WINDOW)
    harness = None
    try:
        harness = json.loads(completed.stdout[completed.stdout.index("{"):])
    except ValueError:
        pass
    done = {"job_id": job_id, "returncode": completed.returncode, "finished_utc": utc_text(),
            "harness_status": (harness or {}).get("status"), "run_dir": (harness or {}).get("run"),
            "error": (harness or {}).get("error"), "stdout_tail": completed.stdout[-6000:], "stderr_tail": completed.stderr[-3000:]}
    atomic_json(folder / "done.json", done)
    with ledger(cfg) as book:
        book["jobs"][job_id].update(status="succeeded" if completed.returncode == 0 else "failed",
                                    finished_utc=done["finished_utc"], run_dir=done["run_dir"], error=done["error"])
    if config_path:
        try:
            launch_queued(cfg, config_path)
        except Exception:  # the runner's next poll will launch it instead
            pass
    return completed.returncode


def monitor(cfg):
    """Mark jobs whose wrapper died without done.json as lost (never as success)."""
    try:
        import psutil
    except ImportError:
        return []
    lost = []
    with ledger(cfg) as book:
        for job in book["jobs"].values():
            if job["status"] != "running" or not job.get("wrapper_pid"):
                continue
            if (job_dir(cfg, job["job_id"]) / "done.json").exists():
                continue
            alive = psutil.pid_exists(job["wrapper_pid"])
            if alive:
                try:
                    alive = psutil.Process(job["wrapper_pid"]).create_time() <= parse_utc(job["started_utc"]).timestamp() + 120
                except psutil.Error:
                    alive = False
            if not alive:
                job.update(status="lost", finished_utc=utc_text(),
                           error="Wrapper exited without done.json (reboot or kill). Inspect runs/ before retrying as a new job.")
                lost.append(job["job_id"])
    return lost


def cancel(cfg, job_id):
    import psutil
    with ledger(cfg) as book:
        job = book["jobs"][job_id]
        if job["status"] == "queued":
            job.update(status="cancelled", finished_utc=utc_text())
            return job
        if job["status"] == "running" and job.get("wrapper_pid") and psutil.pid_exists(job["wrapper_pid"]):
            root = psutil.Process(job["wrapper_pid"])
            for process in reversed([root, *root.children(recursive=True)]):
                try:
                    process.kill()
                except psutil.Error:
                    pass
        job.update(status="cancelled", finished_utc=utc_text())
        return job


def summary(cfg, limit=15):
    jobs = sorted(read_ledger(cfg)["jobs"].values(), key=lambda j: j["requested_utc"], reverse=True)[:limit]
    return [{k: j.get(k) for k in ("job_id", "name", "kind", "status", "requested_utc", "finished_utc", "run_dir", "error",
                                   "code_commit", "workdir")} for j in jobs]


if __name__ == "__main__":  # pragma: no cover
    sys.exit("Use python -m wm_ops")

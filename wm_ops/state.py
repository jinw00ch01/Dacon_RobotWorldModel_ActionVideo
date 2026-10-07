"""Shared helpers: node config, git helpers and the locked per-PC job ledger.

Ported from the AFDA project's agent_bridge. The ledger is the only state the job runner trusts.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
ROLES = ("ultra5060",)
# Console programs started by the windowless runner must not open windows. Windows 11 hands a new
# console to Windows Terminal, so every child process gets CREATE_NO_WINDOW (a hidden console).
NO_WINDOW = 0x08000000 if os.name == "nt" else 0


def utc_now():
    return datetime.now(timezone.utc)


def utc_text(moment=None):
    return (moment or utc_now()).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_utc(text):
    if not isinstance(text, str) or not re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d+)?Z", text):
        raise ValueError(f"Expected RFC3339 UTC timestamp ending in Z: {text!r}")
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".partial")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    _retry(lambda: temp.replace(path))


def load_config(path):
    """configs/local-node.json, written by scripts/setup_node.ps1 (git-ignored, absolute paths)."""
    cfg = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if cfg.get("version") != 1 or cfg.get("role") not in ROLES:
        raise ValueError("Invalid local node configuration")
    for key in ("project_root", "state_root", "python"):
        if not Path(cfg[key]).is_absolute():
            raise ValueError("Local configuration paths must be absolute")
    return cfg


def default_config_path():
    """configs/local-node.json of this checkout, or of the main folder an agent clone (work/agents/<name>) sits in."""
    if os.environ.get("WM_NODE_CONFIG"):
        return os.environ["WM_NODE_CONFIG"]
    for base in (ROOT, *ROOT.parents):
        candidate = base / "configs" / "local-node.json"
        if candidate.is_file():
            return str(candidate)
    return str(ROOT / "configs" / "local-node.json")


def sandbox_package(state_root, local=None):
    """Name of the app package whose sandbox captures writes to state_root, else None.

    Shells inside MSIX apps (the Store build of the Claude desktop app, Codex) silently redirect new
    files under %LOCALAPPDATA% to Packages\\<pkg>\\LocalCache, so the job runner (a Task Scheduler task)
    never sees them. Keep state outside %LOCALAPPDATA% (C:\\Dacon\\WM_Runtime) and check with a probe.
    """
    local = Path(local or os.environ.get("LOCALAPPDATA") or "")
    root = Path(state_root)
    if not str(local) or not root.is_relative_to(local) or not (local / "Packages").is_dir():
        return None
    root.mkdir(parents=True, exist_ok=True)
    probe = root / f".wm-probe-{uuid.uuid4().hex}"
    probe.write_text("probe", encoding="utf-8")
    try:
        relative = probe.relative_to(local)
        for package in (local / "Packages").iterdir():
            if (package / "LocalCache" / "Local" / relative).exists():
                return package.name
        return None
    finally:
        probe.unlink(missing_ok=True)


def ops_root(cfg):
    path = Path(cfg["state_root"]) / "ops"
    path.mkdir(parents=True, exist_ok=True)
    return path


def git_at(cfg, where, *args, timeout=60):
    """Run git in `where` without opening a window; a missing git behaves like a failed call."""
    argv = [cfg.get("git_exe") or "git", *args]
    try:
        return subprocess.run(argv, cwd=str(where), capture_output=True, text=True, encoding="utf-8", errors="replace",
                              timeout=timeout, env={**os.environ, "GIT_TERMINAL_PROMPT": "0"}, creationflags=NO_WINDOW,
                              stdin=subprocess.DEVNULL)
    except OSError as error:
        return subprocess.CompletedProcess(argv, 127, "", str(error))
    except subprocess.TimeoutExpired:  # git has hung for hours on this laptop; treat it as a failed call
        return subprocess.CompletedProcess(argv, 124, "", f"git timed out after {timeout}s")


def git(cfg, *args, timeout=60):
    return git_at(cfg, cfg["project_root"], *args, timeout=timeout)


def head_commit(cfg, where=None):
    value = git_at(cfg, where or cfg["project_root"], "rev-parse", "HEAD").stdout.strip()
    return value if re.fullmatch(r"[a-f0-9]{40}", value) else None


def tree_clean(cfg, where=None):
    result = git_at(cfg, where or cfg["project_root"], "status", "--porcelain")
    return result.returncode == 0 and not result.stdout.strip()


def checkout_root(cfg, where=None):
    """Top folder of the git checkout (main folder or an agent clone) that contains `where`, else the project root."""
    top = git_at(cfg, where or os.getcwd(), "rev-parse", "--show-toplevel").stdout.strip()
    return str(Path(top)) if top else str(Path(cfg["project_root"]))


def update_local_config(config_path, values=None, drop=()):
    """Rewrite configs/local-node.json with some keys set and others removed."""
    path = Path(config_path)
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    raw.update(values or {})
    for key in drop:
        raw.pop(key, None)
    path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    return raw


EMPTY_LEDGER = {"version": 1, "jobs": {}, "heartbeat_utc": None}


@contextmanager
def file_lock(path, wait_seconds=60):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if path.stat().st_size == 0:
            handle.write(b"0")
            handle.flush()
        deadline = time.monotonic() + wait_seconds
        while True:
            handle.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise RuntimeError(f"Timed out waiting for lock {path.name}")
                time.sleep(0.2)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _retry(action, attempts=25, delay=0.2):
    """Windows briefly refuses to replace a file another process is reading; retry for a few seconds."""
    for attempt in range(attempts):
        try:
            return action()
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay)


def _load_ledger(path):
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for key, value in EMPTY_LEDGER.items():
        data.setdefault(key, json.loads(json.dumps(value)))
    return data


@contextmanager
def ledger(cfg):
    """Read-modify-write the ledger under an OS lock shared by the runner and CLI calls."""
    root = ops_root(cfg)
    path = root / "ledger.json"
    with file_lock(root / "ledger.lock"):
        data = _retry(lambda: _load_ledger(path))
        yield data
        atomic_json(path, data)


def read_ledger(cfg):
    return _retry(lambda: _load_ledger(ops_root(cfg) / "ledger.json"))

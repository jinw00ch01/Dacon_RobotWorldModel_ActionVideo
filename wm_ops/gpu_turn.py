"""Take a turn on this PC's GPU from another project (standard library only, any Python 3.8+).

Our job runner (WM-Jobs-ultra5060) pauses its GPU job while someone holds a turn and resumes it after:

    python C:\\Dacon\\RobotWorldModel_ActionVideo\\wm_ops\\gpu_turn.py --who hardness -- python -m src.train_cnn ...

1. Writes <share>/gpu.request with this process's pid (one outside user at a time; others wait).
2. Waits for the runner's <share>/gpu.granted naming this pid (it suspends our GPU job first, within ~15 s).
3. Runs the command, then removes the grant and the request, so our job resumes on the next poll.
If this process dies, the runner sees the dead pid and resumes on its own. <share> is C:/Dacon/WM_Runtime
(override with WM_GPU_SHARE_DIR).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def share_dir() -> Path:
    return Path(os.environ.get("WM_GPU_SHARE_DIR", "C:/Dacon/WM_Runtime"))


def pid_alive(pid) -> bool:
    if not pid:
        return False
    if os.name == "nt":
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        kernel32.CloseHandle(handle)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    return True


def _read(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _remove(path: Path) -> None:
    for _ in range(50):  # Windows refuses while the runner has the file open for a moment
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            time.sleep(0.1)
    path.unlink(missing_ok=True)


def take_request(share: Path, who: str, poll: float) -> Path:
    share.mkdir(parents=True, exist_ok=True)
    request = share / "gpu.request"
    told = False
    while True:
        try:
            with request.open("x", encoding="utf-8") as handle:
                json.dump({"pid": os.getpid(), "who": who, "since_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, handle)
            return request
        except FileExistsError:
            holder = (_read(request) or {}).get("pid")
            if holder and not pid_alive(holder):
                _remove(request)
                continue
            if not told:
                print(f"gpu_turn: another GPU turn is in progress ({request}), waiting", flush=True)
                told = True
            time.sleep(poll)


def wait_for_grant(share: Path, timeout: float, poll: float) -> dict:
    grant = share / "gpu.granted"
    deadline = time.time() + timeout
    while time.time() < deadline:
        data = _read(grant)
        if data and data.get("pid") == os.getpid():
            return data
        time.sleep(poll)
    raise TimeoutError(f"no grant in {timeout:.0f}s: is the WM-Jobs-ultra5060 runner running?")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--who", default=Path.cwd().name, help="shown to the runner and in its log")
    ap.add_argument("--wait-timeout", type=float, default=300.0, help="seconds to wait for the grant once requested")
    ap.add_argument("--poll", type=float, default=2.0)
    ap.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args(argv)
    cmd = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not cmd:
        ap.error("give the command after --")
    share = share_dir()
    request = take_request(share, args.who, args.poll)
    try:
        grant = wait_for_grant(share, args.wait_timeout, args.poll)
        print(f"gpu_turn: GPU granted (paused: {', '.join(grant.get('paused_jobs') or []) or 'nothing'})", flush=True)
        return subprocess.call(cmd)
    finally:
        grant = share / "gpu.granted"
        if ((_read(grant) or {}).get("pid")) == os.getpid():
            _remove(grant)
        _remove(request)


if __name__ == "__main__":
    sys.exit(main())

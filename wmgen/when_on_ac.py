"""Run a command only while the laptop is on AC power (for GPU jobs queued across moves and restarts).

Waits for AC power, starts the command without a console window, and if the laptop stays on battery
for --grace seconds, stops it and waits for AC again before starting it over (so use it for resumable
or short commands). Exits with the command's exit code once it finishes on AC.

python -m wmgen.when_on_ac -- C:/.../python.exe -m wmgen.latent_cache --out ...
"""
from __future__ import annotations

import argparse
import ctypes
import subprocess
import sys
import time

CREATE_NO_WINDOW = 0x08000000


class _PowerStatus(ctypes.Structure):
    _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
                ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]


def on_ac() -> bool:
    status = _PowerStatus()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
        return True  # unknown: do not block
    return status.ACLineStatus != 0  # 1 = AC, 255 = unknown


def _stop(proc: subprocess.Popen) -> None:
    subprocess.call(["taskkill", "/T", "/F", "/PID", str(proc.pid)], creationflags=CREATE_NO_WINDOW,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    proc.wait()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grace", type=float, default=120.0, help="seconds on battery before stopping the command")
    ap.add_argument("--poll", type=float, default=10.0)
    ap.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    cmd = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not cmd:
        raise SystemExit("no command given")

    while True:
        if not on_ac():
            print("waiting for AC power", flush=True)
            while not on_ac():
                time.sleep(args.poll)
        print("on AC power, starting:", " ".join(cmd), flush=True)
        proc = subprocess.Popen(cmd, creationflags=CREATE_NO_WINDOW, stdout=sys.stdout, stderr=sys.stderr)
        battery_since = None
        while proc.poll() is None:
            time.sleep(args.poll)
            if on_ac():
                battery_since = None
            elif battery_since is None:
                battery_since = time.time()
            elif time.time() - battery_since > args.grace:
                print("on battery too long, stopping the command until AC returns", flush=True)
                _stop(proc)
                break
        else:
            sys.exit(proc.returncode)


if __name__ == "__main__":
    main()

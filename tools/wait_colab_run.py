"""Wait until the Colab run on Google Drive reaches a milestone, then print what is there (for a background waiter on Ultra).

Milestones: "start" = the latent IDM or the first training log exists; "done" = adapter_<steps>.pt exists (or no new log line
for --stall-hours). Drive hiccups (the G: drive briefly missing) are retried, not treated as an exit condition.

python tools/wait_colab_run.py --run v5 --until start
python tools/wait_colab_run.py --run v5 --until done --steps 32000
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def my_drive() -> Path | None:
    try:
        for d in Path("G:/").iterdir():
            if d.is_dir() and (d / "Colab Notebooks").exists():
                return d
    except OSError:
        return None
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v5")
    ap.add_argument("--until", choices=["start", "done"], required=True)
    ap.add_argument("--steps", type=int, default=32000)
    ap.add_argument("--poll", type=int, default=120)
    ap.add_argument("--stall-hours", type=float, default=2.0)
    args = ap.parse_args()
    last_change, last_size = time.time(), -1
    while True:
        root = my_drive()
        if root is not None:
            ca = root / "dacon_wm" / "cosmos_ac"
            run, log = ca / "runs" / args.run, ca / "runs" / args.run / "train_log.jsonl"
            try:
                if args.until == "start" and ((ca / "latent_idm" / "latent_idm_v1.pt").exists() or log.exists()):
                    break
                if args.until == "done":
                    if (run / f"adapter_{args.steps:06d}.pt").exists():
                        break
                    size = log.stat().st_size if log.exists() else 0
                    if size != last_size:
                        last_change, last_size = time.time(), size
                    elif size > 0 and time.time() - last_change > args.stall_hours * 3600:
                        print(f"no new log line for {args.stall_hours} h", flush=True)
                        break
            except OSError:
                pass
        time.sleep(args.poll)
    adapters = sorted(p.name for p in run.glob("adapter_0*.pt")) if run.exists() else []
    tail = log.read_text(encoding="utf-8").splitlines()[-2:] if log.exists() else []
    print(json.dumps({"until": args.until, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                      "latent_idm": (ca / "latent_idm" / "latent_idm_v1.pt").exists(), "adapters": adapters[-6:],
                      "n_adapters": len(adapters), "log_tail": tail}, indent=1), flush=True)


if __name__ == "__main__":
    main()

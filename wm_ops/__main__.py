"""CLI: python -m wm_ops <command>. Config defaults to configs/local-node.json (written by setup_node.ps1)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from . import jobs, worker
from .state import default_config_path, load_config, read_ledger, sandbox_package


def _print(value):
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(value, ensure_ascii=False, indent=2))


def build_parser():
    parser = argparse.ArgumentParser(prog="python -m wm_ops", description=__doc__)
    parser.add_argument("--config", default=default_config_path())
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=argparse.SUPPRESS)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", parents=[common], help="Runner heartbeat and recent jobs")
    sub.add_parser("tick", parents=[common], help="One runner iteration (debug)")
    sub.add_parser("serve", parents=[common], help="Runner loop (used by the WM-Jobs task through pythonw)")
    sub.add_parser("stop", parents=[common], help="Ask the runner to exit (running jobs continue)")
    sub.add_parser("resume", parents=[common], help="Remove the STOP request")
    job = sub.add_parser("job", parents=[common], help="Queue and inspect detached long jobs")
    job_sub = job.add_subparsers(dest="job_command", required=True)
    start = job_sub.add_parser("start", parents=[common])
    start.add_argument("--kind", required=True, choices=["cpu", "gpu"])
    start.add_argument("--timeout", required=True, type=float, help="seconds")
    start.add_argument("--name", required=True)
    start.add_argument("--allow-dirty", action="store_true")
    start.add_argument("child", nargs=argparse.REMAINDER)
    job_sub.add_parser("list", parents=[common])
    job_sub.add_parser("show", parents=[common]).add_argument("job_id")
    job_sub.add_parser("cancel", parents=[common]).add_argument("job_id")
    run = sub.add_parser("job-run", parents=[common], help=argparse.SUPPRESS)
    run.add_argument("--job", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    writes_state = args.command in {"tick", "serve", "stop", "resume", "job-run"} or (
        args.command == "job" and args.job_command in {"start", "cancel"})
    package = sandbox_package(cfg["state_root"]) if writes_state else None
    if package:
        _print({"error": f"This shell runs inside the app sandbox ({package}); state written here never reaches the "
                         "job runner. Keep state_root outside %LOCALAPPDATA% (setup_node.ps1 uses C:\\Dacon\\WM_Runtime).",
                "command": args.command})
        return 2

    if args.command == "status":
        book = read_ledger(cfg)
        _print({"role": cfg["role"], "runner_heartbeat_utc": book.get("heartbeat_utc"),
                "stop_requested": (Path(cfg["state_root"]) / "STOP").exists(), "jobs": jobs.summary(cfg, 8)})
    elif args.command == "tick":
        _print(worker.tick(cfg, args.config))
    elif args.command == "serve":
        return worker.serve(args.config)
    elif args.command in {"stop", "resume"}:
        flag = Path(cfg["state_root"]) / "STOP"
        if args.command == "stop":
            flag.write_text("stop requested", encoding="utf-8")
        else:
            flag.unlink(missing_ok=True)
        _print({"stop_requested": flag.exists()})
    elif args.command == "job":
        if args.job_command == "start":
            child = args.child[1:] if args.child[:1] == ["--"] else args.child
            _print(jobs.request(cfg, args.kind, args.timeout, args.name, child, args.allow_dirty))
        elif args.job_command == "list":
            _print(jobs.summary(cfg, 30))
        elif args.job_command == "show":
            folder = jobs.job_dir(cfg, args.job_id)
            done = folder / "done.json"
            _print({"ledger": read_ledger(cfg)["jobs"].get(args.job_id),
                    "done": json.loads(done.read_text(encoding="utf-8")) if done.exists() else None})
        else:
            _print(jobs.cancel(cfg, args.job_id))
    elif args.command == "job-run":
        return jobs.run(cfg, args.job, args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

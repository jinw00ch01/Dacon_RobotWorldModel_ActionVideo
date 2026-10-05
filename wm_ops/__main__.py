"""CLI: python -m wm_ops <command>. Config defaults to configs/local-node.json (written by setup_node.ps1)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from . import jobs, packets, syncthing, worker
from .state import ROLES, default_config_path, dirs, load_config, read_ledger, sandbox_package, strict_load


def _print(value):
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(value, ensure_ascii=False, indent=2))


def build_parser():
    parser = argparse.ArgumentParser(prog="python -m wm_ops", description=__doc__)
    parser.add_argument("--config", default=default_config_path())
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=argparse.SUPPRESS)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", parents=[common], help="Packets, jobs, Syncthing and peer heartbeat")
    sub.add_parser("tick", parents=[common], help="One supervisor iteration (normally run by the exchange task)")
    sub.add_parser("inbox", parents=[common], help="Imported packets and their local folders")
    handled = sub.add_parser("handled", parents=[common], help="Mark an imported packet as handled")
    handled.add_argument("packet_id")
    publish = sub.add_parser("publish", parents=[common], help="Publish a packet from a JSON body (+ optional attachment folder)")
    publish.add_argument("--kind", required=True, choices=sorted(packets.KINDS - {"ack"}))
    publish.add_argument("--body", required=True, type=Path)
    publish.add_argument("--attach", type=Path)
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
    sync = sub.add_parser("syncthing", parents=[common], help="Configure or inspect the WM Syncthing instance")
    sync.add_argument("action", choices=["configure", "status", "shutdown"])
    sync.add_argument("--peer-id")
    sub.add_parser("register-device", parents=[common], help="Write this PC's Syncthing device ID into configs/nodes.json")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    writes_state = args.command in {"tick", "publish", "handled", "job-run"} or (
        args.command == "job" and args.job_command in {"start", "cancel"})
    package = sandbox_package(cfg["state_root"]) if writes_state else None
    if package:
        _print({"error": f"This shell runs inside the app sandbox ({package}); state written here never reaches the "
                         "exchange task. Keep state_root outside %LOCALAPPDATA% (setup_node.ps1 uses C:\\Dacon\\WM_Runtime).",
                "command": args.command})
        return 2

    if args.command == "status":
        book = read_ledger(cfg)
        _, inbox, other = dirs(cfg)
        peer_status = inbox / "status" / f"{other}.json"
        result = {"role": cfg["role"], "heartbeat_utc": book.get("heartbeat_utc"),
                  "packets_unhandled": [{"packet_id": p, "kind": e.get("kind"), "subject": e.get("subject"), "path": e.get("local_path")}
                                        for p, e in book["packets"].items() if e.get("status") == "imported" and not e.get("handled")],
                  "packets_waiting": [p for p, e in book["packets"].items() if e.get("status") == "waiting"],
                  "quarantined": [p for p, e in book["packets"].items() if e.get("status") == "quarantined"],
                  "sent_unacked": [p for p in book["sent"] if p not in book["acks"]],
                  "jobs": jobs.summary(cfg, 8),
                  "peer_heartbeat": json.loads(peer_status.read_text(encoding="utf-8")) if peer_status.exists() else None}
        if cfg.get("syncthing_home"):
            try:
                result["syncthing"] = syncthing.status(cfg["syncthing_home"], cfg.get("peer_device_id"))
            except Exception as error:
                result["syncthing"] = {"error": f"{type(error).__name__}: {error}"}
        _print(result)
    elif args.command == "tick":
        _print(worker.tick(cfg, args.config))
    elif args.command == "inbox":
        book = read_ledger(cfg)
        _print([{"packet_id": p, **{k: e.get(k) for k in ("kind", "status", "handled", "subject", "experiment_id", "imported_utc",
                                                          "local_path", "detail")}}
                for p, e in sorted(book["packets"].items(), key=lambda item: item[1].get("imported_utc") or "")])
    elif args.command == "handled":
        _print(packets.mark_handled(cfg, args.packet_id))
    elif args.command == "publish":
        packet_id, manifest = packets.publish(cfg, args.kind, strict_load(args.body), args.attach)
        _print({"packet_id": packet_id, "manifest_sha256": manifest, "kind": args.kind})
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
        return jobs.run(cfg, args.job)
    elif args.command == "syncthing":
        home = cfg["syncthing_home"]
        if args.action == "configure":
            peer = args.peer_id or cfg.get("peer_device_id")
            result = syncthing.configure(home, cfg["role"], cfg["exchange_root"], peer,
                                         cfg.get("data_root") if cfg.get("share_data") else None,
                                         cfg.get("skip_train_videos", False), cfg.get("listen_port", 22010))
            raw = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
            raw.update(device_id=result["device_id"], peer_device_id=peer)
            Path(args.config).write_text(json.dumps(raw, indent=2), encoding="utf-8")
            _print(result)
        elif args.action == "status":
            _print(syncthing.status(home, cfg.get("peer_device_id")))
        else:
            syncthing.api(home, "system/shutdown", {}, "POST")
            _print({"shutdown": True})
    elif args.command == "register-device":
        device = syncthing.api(cfg["syncthing_home"], "system/status")["myID"]
        path = worker.nodes_file(cfg)
        data = strict_load(path) if path.exists() else {"version": 1, "devices": {r: None for r in ROLES}}
        data.setdefault("devices", {})[cfg["role"]] = device
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        _print({"nodes_file": str(path), "role": cfg["role"], "device_id": device,
                "next": "commit and push configs/nodes.json so the peer pairs automatically"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Experiment packets over the Syncthing exchange folders (ported from AFDA, schema wm.experiment.v1).

Layout: <own outbox>/experiments/v1/<packet_id>/{<kind>.json, files/..., manifest.json, COMMITTED.json}.
COMMITTED.json is written last. The receiver verifies every byte, imports into
work/packets/inbox/<packet_id>/ and answers with an ack packet.
Packets carry data and text only; they never carry commands to execute.
"""
from __future__ import annotations

from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import uuid

from .state import ROLES, atomic_json, digest, dirs, head_commit, ledger, parse_utc, safe_path, strict_load, utc_text

SCHEMA = "wm.experiment.v1"
# Ultra leads experiments; Pro reviews and does data/QA work; both may request and ack.
AUTHORS = {"spec": {"ultra5060"}, "result": set(ROLES), "decision": {"ultra5060"},
           "review": {"pro360"}, "qa": {"pro360"}, "data": set(ROLES), "request": set(ROLES), "ack": set(ROLES)}
KINDS = set(AUTHORS)
SUFFIXES = {".json", ".csv", ".md", ".txt", ".png", ".jpg", ".jpeg", ".log", ".npz", ".npy", ".parquet", ".mp4", ".pt",
            ".safetensors"}
MAX_FILE = 2 * 1024**3
MAX_TOTAL = 8 * 1024**3
MAX_FILES = 5000
PACKET_ID = re.compile(r"[a-f0-9]{32}")
EXPERIMENT_ID = re.compile(r"exp-[a-f0-9]{32}")
ATTEMPT_ID = re.compile(r"a\d{3}")
SHA = re.compile(r"[a-f0-9]{64}")


def inbox_root(cfg):
    return Path(cfg["project_root"]).resolve() / "work" / "packets" / "inbox"


def _ref(value, field):
    if value is None:
        return
    if not isinstance(value, dict) or set(value) - {"packet_id", "manifest_sha256", "kind"}:
        raise ValueError(f"{field} must be null or {{packet_id, manifest_sha256}}")
    if not PACKET_ID.fullmatch(str(value.get("packet_id"))):
        raise ValueError(f"{field}.packet_id invalid")
    if value.get("manifest_sha256") is not None and not SHA.fullmatch(str(value["manifest_sha256"])):
        raise ValueError(f"{field}.manifest_sha256 invalid")


def validate(body):
    """Validate the envelope and the minimum per-kind fields. Unknown extra fields are data."""
    if not isinstance(body, dict):
        raise ValueError("Packet body must be a JSON object")
    if body.get("example_only"):
        raise ValueError("example_only packets are templates and are always rejected")
    if body.get("schema_version") != SCHEMA:
        raise ValueError(f"schema_version must be {SCHEMA}")
    kind = body.get("kind")
    if kind not in KINDS:
        raise ValueError(f"Unsupported kind: {kind!r}")
    if not PACKET_ID.fullmatch(str(body.get("packet_id"))):
        raise ValueError("packet_id must be 32 lowercase hex characters")
    sender, target = body.get("sender"), body.get("target")
    if sender not in ROLES or target not in ROLES:
        raise ValueError("sender/target must be ultra5060 or pro360")
    if sender not in AUTHORS[kind]:
        raise ValueError(f"{sender} may not author {kind} packets")
    if sender == target:
        raise ValueError("A packet must target the other PC")
    parse_utc(body.get("created_utc"))
    if not isinstance(body.get("provenance"), dict):
        raise ValueError("provenance object is required")
    for field in ("parent", "supersedes"):
        _ref(body.get(field), field)
    if kind in {"spec", "result", "review", "decision"}:
        if not EXPERIMENT_ID.fullmatch(str(body.get("experiment_id"))):
            raise ValueError("experiment_id must be exp-<32 hex>")
    if kind in {"result", "review"} and not ATTEMPT_ID.fullmatch(str(body.get("attempt_id"))):
        raise ValueError("attempt_id must look like a001")
    if kind == "spec":
        for field in ("stage", "purpose", "hypothesis"):
            if not body.get(field):
                raise ValueError(f"spec.{field} is required")
    elif kind == "result":
        if body.get("status") not in {"succeeded", "failed", "interrupted"}:
            raise ValueError("result.status must be succeeded/failed/interrupted")
    elif kind == "review":
        for field in ("integrity_pass", "metrics_reproduced"):
            if body.get(field) not in {True, False, None}:
                raise ValueError(f"review.{field} must be true/false/null")
    elif kind == "decision":
        if body.get("decision") not in {"accept", "reject", "request_revision", "stop"}:
            raise ValueError("decision must be accept/reject/request_revision/stop")
        if not body.get("reason"):
            raise ValueError("decision.reason is required")
    elif kind == "ack":
        _ref(body.get("ack_of"), "ack_of")
        if body.get("ack_of") is None or body.get("status") not in {"received", "quarantined"}:
            raise ValueError("ack needs ack_of and status received/quarantined")
    elif kind in {"qa", "request", "data"}:
        if not body.get("subject") or not isinstance(body.get("subject"), str):
            raise ValueError(f"{kind}.subject is required")
    return body


def _attachments(folder):
    if folder is None:
        return []
    folder = Path(folder).resolve()
    if not folder.is_dir():
        raise ValueError(f"Attachment folder not found: {folder}")
    rows, total = [], 0
    for path in sorted(folder.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symlinks are not allowed: {path}")
        if not path.is_file():
            continue
        name = "files/" + path.relative_to(folder).as_posix()
        if PurePosixPath(name).suffix.lower() not in SUFFIXES or any(p.startswith(".") for p in PurePosixPath(name).parts):
            raise ValueError(f"Attachment type not allowed: {name}")
        size = path.stat().st_size
        total += size
        if size > MAX_FILE or total > MAX_TOTAL or len(rows) >= MAX_FILES:
            raise ValueError("Packet exceeds size limits (2 GiB/file, 8 GiB, 5000 files)")
        rows.append((name, path))
    return rows


def publish(cfg, kind, body, attachments=None):
    """Write an immutable packet into this PC's outbox. Returns (packet_id, manifest_sha256)."""
    outbox, _, other = dirs(cfg)
    body = dict(body)
    body.pop("example_only", None)
    packet_id = uuid.uuid4().hex
    body.update(schema_version=SCHEMA, kind=kind, packet_id=packet_id, sender=cfg["role"], target=other,
                created_utc=utc_text())
    provenance = dict(body.get("provenance") or {})
    provenance.setdefault("tool", "wm_ops")
    provenance.setdefault("code_commit", head_commit(cfg))
    body["provenance"] = provenance
    validate(body)
    files = _attachments(attachments)
    base = outbox / "experiments" / "v1"
    staging = base / (".staging-" + packet_id)
    final = base / packet_id
    staging.mkdir(parents=True)
    try:
        atomic_json(staging / f"{kind}.json", body)
        for name, source in files:
            destination = safe_path(staging, name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        rows = []
        for path in sorted(p for p in staging.rglob("*") if p.is_file()):
            rows.append({"path": path.relative_to(staging).as_posix(), "bytes": path.stat().st_size, "sha256": digest(path)})
        atomic_json(staging / "manifest.json", {"version": 1, "packet_id": packet_id, "files": rows})
        manifest_sha = digest(staging / "manifest.json")
        staging.rename(final)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    atomic_json(final / "COMMITTED.json", {"packet_id": packet_id, "kind": kind, "manifest_sha256": manifest_sha})
    if kind != "ack":
        with ledger(cfg) as book:
            book["sent"][packet_id] = {"kind": kind, "target": body["target"], "manifest_sha256": manifest_sha,
                                       "created_utc": body["created_utc"], "experiment_id": body.get("experiment_id"),
                                       "subject": body.get("subject")}
    return packet_id, manifest_sha


def _check(packet, committed):
    """Return (status, detail, body). status: ready / waiting / quarantined."""
    manifest = packet / "manifest.json"
    if not manifest.is_file():
        return "waiting", "manifest not yet synced", None
    if digest(manifest) != committed["manifest_sha256"]:
        return "quarantined", "manifest SHA differs from COMMITTED", None
    data = strict_load(manifest, 8 * 1024**2)
    if data.get("version") != 1 or data.get("packet_id") != packet.name or not isinstance(data.get("files"), list):
        return "quarantined", "manifest schema invalid", None
    rows = data["files"]
    total = 0
    for row in rows:
        if set(row) != {"path", "bytes", "sha256"} or not isinstance(row["bytes"], int) or not SHA.fullmatch(str(row["sha256"])):
            return "quarantined", "manifest row invalid", None
        total += row["bytes"]
        target = safe_path(packet, row["path"])
        if not target.is_file() or target.stat().st_size != row["bytes"]:
            return "waiting", "payload not yet complete: " + row["path"], None
    if total > MAX_TOTAL or len(rows) > MAX_FILES + 1:
        return "quarantined", "packet exceeds limits", None
    for row in rows:
        if digest(safe_path(packet, row["path"])) != row["sha256"]:
            return "quarantined", "payload hash mismatch: " + row["path"], None
    body_name = committed["kind"] + ".json"
    if body_name not in {r["path"] for r in rows}:
        return "quarantined", "missing " + body_name, None
    body = strict_load(packet / body_name)
    validate(body)
    if body["packet_id"] != packet.name or body["kind"] != committed["kind"]:
        return "quarantined", "packet_id/kind mismatch", None
    return "ready", rows, body


def import_inbox(cfg):
    """Verify and import newly committed packets from the peer. Returns imported packet ids."""
    _, inbox, other = dirs(cfg)
    base = inbox / "experiments" / "v1"
    imported = []
    if not base.is_dir():
        return imported
    for marker in sorted(base.glob("*/COMMITTED.json")):
        packet = marker.parent
        if not PACKET_ID.fullmatch(packet.name):
            continue
        with ledger(cfg) as book:
            if book["packets"].get(packet.name, {}).get("status") in {"imported", "quarantined"}:
                continue
        try:
            committed = strict_load(marker, 4096)
            if (committed.get("packet_id") != packet.name or committed.get("kind") not in KINDS
                    or not SHA.fullmatch(str(committed.get("manifest_sha256")))):
                raise ValueError("COMMITTED.json invalid")
            status, detail, body = _check(packet, committed)
            if status == "ready" and (body["sender"] != other or body["target"] != cfg["role"]):
                status, detail = "quarantined", "sender/target does not match the folder owner"
        except (ValueError, OSError) as error:
            status, detail, body, committed = "quarantined", f"{type(error).__name__}: {error}", None, {"manifest_sha256": None, "kind": None}
        if status == "waiting":
            with ledger(cfg) as book:
                entry = book["packets"].setdefault(packet.name, {})
                entry.update(status="waiting", detail=detail, seen_utc=entry.get("seen_utc") or utc_text())
            continue
        local = inbox_root(cfg) / packet.name
        if status == "ready":
            local.mkdir(parents=True, exist_ok=True)
            for row in detail:
                target = safe_path(local, row["path"])
                if target.exists():
                    if digest(target) != row["sha256"]:
                        raise ValueError(f"Local copy of packet {packet.name} differs; preserved")
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(safe_path(packet, row["path"]), target)
                target.chmod(stat.S_IREAD)
        with ledger(cfg) as book:
            book["packets"][packet.name] = {
                "status": "imported" if status == "ready" else "quarantined",
                "kind": committed.get("kind"), "manifest_sha256": committed.get("manifest_sha256"),
                "sender": other, "imported_utc": utc_text(),
                "local_path": str(local) if status == "ready" else str(packet),
                "experiment_id": body.get("experiment_id") if body else None,
                "subject": body.get("subject") if body else None,
                "detail": None if status == "ready" else detail,
                # Acks never need handling; everything else is listed until a session marks it handled.
                "handled": committed.get("kind") == "ack",
            }
            if status == "ready" and body["kind"] == "ack":
                book["acks"][body["ack_of"]["packet_id"]] = {"status": body["status"], "received_utc": utc_text()}
        if committed.get("kind") != "ack" and PACKET_ID.fullmatch(packet.name):
            ack = {"ack_of": {"packet_id": packet.name, "manifest_sha256": committed.get("manifest_sha256"),
                              "kind": committed.get("kind")},
                   "status": "received" if status == "ready" else "quarantined",
                   "detail": None if status == "ready" else detail,
                   "provenance": {"tool": "wm_ops"}}
            if ack["ack_of"]["manifest_sha256"] is None:
                ack["ack_of"].pop("manifest_sha256")
            publish(cfg, "ack", ack)
        imported.append(packet.name)
    return imported


def mark_handled(cfg, packet_id):
    with ledger(cfg) as book:
        entry = book["packets"].get(packet_id)
        if entry is None:
            raise ValueError(f"Unknown packet {packet_id}")
        entry["handled"] = True
        entry["handled_utc"] = utc_text()
        return entry

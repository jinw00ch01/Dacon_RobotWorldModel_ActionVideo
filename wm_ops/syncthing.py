"""Configure only the dedicated WM Syncthing instance through its loopback REST API (ported from AFDA).

Folders (send-only on the owner, receive-only on the peer):
  wm-ultra-to-pro-v1  <exchange_root>/ultra_to_pro   packets written by Ultra
  wm-pro-to-ultra-v1  <exchange_root>/pro_to_ultra   packets written by Pro
  wm-data-v1          <project_root>/open            competition data, Ultra -> Pro (optional)
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import re
import urllib.request
import xml.etree.ElementTree as ET

from .state import FOLDERS, ROLES

DEVICE_ID = re.compile(r"[A-Z2-7]{7}(?:-[A-Z2-7]{7}){7}")
DATA_FOLDER = "wm-data-v1"


def api(home, endpoint, body=None, method=None):
    xml = ET.parse(Path(home) / "config.xml").getroot()
    address = xml.findtext("gui/address")
    if not re.fullmatch(r"127\.0\.0\.1:\d+", address or ""):
        raise ValueError("WM Syncthing administration must stay on IPv4 loopback")
    key = xml.findtext("gui/apikey")
    request = urllib.request.Request("http://" + address + "/rest/" + endpoint,
                                     data=None if body is None else json.dumps(body).encode(),
                                     headers={"X-API-Key": key, "Content-Type": "application/json"}, method=method)
    with urllib.request.urlopen(request, timeout=15) as response:
        data = response.read()
    return json.loads(data) if data else None


def folder_plan(role, exchange_root, data_root=None):
    """[(folder_id, label, path, type)] for this role."""
    plan = []
    for owner, name in FOLDERS.items():
        plan.append(("wm-" + name.replace("_", "-") + "-v1", "WM " + name, Path(exchange_root) / name,
                     "sendonly" if owner == role else "receiveonly"))
    if data_root:
        plan.append((DATA_FOLDER, "WM data (open)", Path(data_root), "sendonly" if role == "ultra5060" else "receiveonly"))
    return plan


def data_ignore_lines(role, skip_train_videos):
    lines = ["(?d)*.partial", "(?d).staging-*"]
    if role == "pro360" and skip_train_videos:
        # Train videos are 8.3 of 8.7 GB; the Pro can work from parquet/meta/eval and fetch videos later.
        lines.append("/data/train/*/*/videos")
    return lines


def configure(home, role, exchange_root, peer_id=None, data_root=None, skip_train_videos=False, listen_port=22010,
              self_name=None):
    """Apply devices, folders and options. self_name carries the one-time join code while pairing."""
    if role not in ROLES:
        raise ValueError("Unknown role")
    if peer_id and not DEVICE_ID.fullmatch(peer_id):
        raise ValueError("Invalid Syncthing device ID")
    cfg = api(home, "config")
    my_id = api(home, "system/status")["myID"]
    if peer_id == my_id:
        raise ValueError("Peer cannot be this device")
    if peer_id and not any(d["deviceID"] == peer_id for d in cfg["devices"]):
        device = copy.deepcopy(cfg["defaults"]["device"])
        device.update(deviceID=peer_id, name="WM-" + next(r for r in ROLES if r != role), autoAcceptFolders=False)
        cfg["devices"].append(device)
    for device in cfg["devices"]:
        if device["deviceID"] == my_id:
            device["name"] = self_name or "WM-" + role
    for folder_id, label, path, kind in folder_plan(role, exchange_root, data_root):
        folder = next((copy.deepcopy(f) for f in cfg["folders"] if f["id"] == folder_id), copy.deepcopy(cfg["defaults"]["folder"]))
        path.mkdir(parents=True, exist_ok=True)
        members = {d["deviceID"] for d in folder.get("devices", [])} | {my_id}
        if peer_id:
            members.add(peer_id)
        # Syncthing v2 keeps an encryptionPassword field when it is omitted; clear it explicitly (AFDA lesson).
        folder.update(id=folder_id, label=label, path=str(path.resolve()), type=kind,
                      devices=[{"deviceID": d, "encryptionPassword": ""} for d in sorted(members)],
                      rescanIntervalS=3600 if folder_id == DATA_FOLDER else 60, fsWatcherEnabled=True)
        if kind == "receiveonly":
            # The empty simple-versioner path can resolve to the folder root on Windows v2 (AFDA lesson).
            folder.pop("versioning", None)
        cfg["folders"] = [f for f in cfg["folders"] if f["id"] != folder_id] + [folder]
        ignore = path / ".stignore"
        lines = data_ignore_lines(role, skip_train_videos) if folder_id == DATA_FOLDER else ["*.partial", ".staging-*"]
        if not ignore.exists() or ignore.read_text(encoding="utf-8").splitlines() != lines:
            ignore.write_text("\n".join(lines) + "\n", encoding="utf-8")
    cfg["options"].update(startBrowser=False, natEnabled=False, urAccepted=-1, autoUpgradeIntervalH=0,
                          listenAddresses=[f"tcp://0.0.0.0:{listen_port}", f"quic://0.0.0.0:{listen_port}", "dynamic+https://relays.syncthing.net/endpoint"])
    api(home, "config", cfg, "PUT")
    return {"device_id": my_id, "role": role, "peer_id": peer_id,
            "restart_required": api(home, "config/restart-required")["requiresRestart"]}


def status(home, peer_id=None):
    state = api(home, "system/status")
    connections = api(home, "system/connections")["connections"]
    folders = {}
    for folder in api(home, "config")["folders"]:
        try:
            info = api(home, "db/status?folder=" + folder["id"])
            folders[folder["id"]] = {k: info.get(k) for k in ("state", "globalBytes", "localBytes", "needBytes", "errors")}
        except Exception as error:  # status is best effort
            folders[folder["id"]] = {"error": str(error)}
    return {"device_id": state["myID"], "peer_device_id": peer_id,
            "peer_connected": bool(connections.get(peer_id, {}).get("connected")) if peer_id else None,
            "pending_devices": api(home, "cluster/pending/devices"), "folders": folders}

"""python -m harness {doctor,execute} --profile <role>. Each run writes immutable evidence under runs/."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("torch", "torchvision", "numpy", "polars", "pyarrow", "opencv-python", "av", "timm", "transformers", "diffusers",
            "accelerate", "peft", "psutil")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def versions():
    found = {}
    for name in PACKAGES:
        try:
            found[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            found[name] = None
    return found


def doctor(args, config, run):
    import psutil
    import torch
    errors = []
    installed = versions()
    errors += [f"missing package: {k}" for k, v in installed.items() if v is None]
    disk = shutil.disk_usage(ROOT)
    if disk.free < config["min_free_disk_gib"] * 1024**3:
        errors.append(f"free disk below {config['min_free_disk_gib']} GiB")
    cuda = {"available": torch.cuda.is_available(), "runtime": torch.version.cuda}
    if config["device"] == "cuda":
        if not cuda["available"]:
            errors.append("CUDA required for the Ultra profile")
        else:
            props = torch.cuda.get_device_properties(0)
            cuda.update(name=props.name, capability=list(torch.cuda.get_device_capability()),
                        total_memory_gib=round(props.total_memory / 1024**3, 2))
            value = torch.ones((64, 64), device="cuda")
            assert (value @ value)[0, 0].item() == 64
            torch.cuda.synchronize()
            cuda["matrix_test"] = "passed"
    data = ROOT / "open" / "data"  # in an agent worktree, scripts/link_data.ps1 makes ./open point at the data
    eval_images = len(list((data / "eval" / "images").glob("*.png"))) if data.exists() else 0
    data_status = {"data_root": str(data), "present": data.exists(), "eval_images": eval_images,
                   "train_datasets": len(list((data / "train").glob("*/*/meta/info.json"))) if data.exists() else 0}
    if config.get("requires_data", True) and eval_images != 216:
        errors.append(f"expected 216 eval images, found {eval_images}")
    result = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "host": platform.node(),
              "packages": installed, "cuda": cuda, "data": data_status, "disk_free_gib": round(disk.free / 1024**3, 1),
              "ram_total_gib": round(psutil.virtual_memory().total / 1024**3, 1), "errors": errors}
    write_json(run / "environment.json", result)
    frozen = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    (run / "pip-freeze.txt").write_text(frozen.stdout, encoding="utf-8")
    if errors:
        raise ValueError("; ".join(errors))
    return result


def gpu_lock_path():
    """One GPU lock per PC, shared by every checkout (main folder and agent worktrees)."""
    if os.environ.get("WM_GPU_LOCK"):
        return Path(os.environ["WM_GPU_LOCK"])
    shared = Path("C:/Dacon/WM_Runtime")
    return shared / "gpu.lock" if shared.is_dir() else ROOT / "work" / "gpu.lock"


def acquire_gpu_lock(lock, run):
    """Create the lock file; a lock left by a supervisor that no longer exists (crash, TDR) is replaced."""
    import psutil
    lock.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        try:
            with lock.open("x", encoding="utf-8") as handle:
                json.dump({"supervisor_pid": os.getpid(), "host": platform.node(), "run": str(run)}, handle)
            return
        except FileExistsError:
            try:
                holder = json.loads(lock.read_text(encoding="utf-8")).get("supervisor_pid")
            except (OSError, ValueError):
                holder = None
            if holder and psutil.pid_exists(holder):
                raise RuntimeError(f"GPU busy: {lock} is held by pid {holder}") from None
            lock.unlink(missing_ok=True)
    raise RuntimeError(f"Could not take the GPU lock {lock}")


def execute(args, config, run):
    """Supervise one process tree with a timeout and a RAM budget; GPU jobs hold the PC-wide GPU lock."""
    import psutil
    command = list(args.child_command)
    if command and command[0] == "--":
        command.pop(0)
    if not command or args.timeout <= 0:
        raise ValueError("Provide a command after -- and a positive timeout")
    if args.kind == "gpu" and config["device"] != "cuda":
        raise ValueError("GPU jobs belong on Ultra")
    lock = gpu_lock_path()
    acquired = False
    child = None
    try:
        if args.kind == "gpu":
            acquire_gpu_lock(lock, run)
            acquired = True
        environment = os.environ.copy()
        environment.update(OMP_NUM_THREADS=str(config["torch_threads"]), MKL_NUM_THREADS=str(config["torch_threads"]),
                           PYTHONHASHSEED=str(config["seed"]), WM_PROFILE=args.profile, WM_RUN_DIR=str(run),
                           PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        if config["device"] == "cpu":
            environment["CUDA_VISIBLE_DEVICES"] = ""
        started = time.monotonic()
        peak_rss = 0
        failure = None
        with (run / "stdout.log").open("w", encoding="utf-8") as out, (run / "stderr.log").open("w", encoding="utf-8") as err:
            child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=out, stderr=err)
            process = psutil.Process(child.pid)
            while child.poll() is None:
                try:
                    family = [process, *process.children(recursive=True)]
                    rss = sum(p.memory_info().rss for p in family if p.is_running())
                    peak_rss = max(peak_rss, rss)
                    if rss > config["ram_budget_gib"] * 1024**3:
                        failure = "Process tree exceeded the configured RAM budget"
                    if time.monotonic() - started > args.timeout:
                        failure = "Process tree exceeded its timeout"
                    if failure:
                        for p in reversed(family):
                            try:
                                p.kill()
                            except psutil.NoSuchProcess:
                                pass
                        child.wait(timeout=15)
                        break
                except psutil.NoSuchProcess:
                    pass
                time.sleep(0.5)
        result = {"argv": command, "returncode": child.returncode, "peak_process_tree_rss_gib": round(peak_rss / 1024**3, 2),
                  "seconds": round(time.monotonic() - started, 1), "failure": failure}
        write_json(run / "process.json", result)
        if failure or child.returncode:
            raise RuntimeError(failure or f"Child exited with code {child.returncode}; see stderr.log")
        return result
    finally:
        if child is not None and child.poll() is None:
            try:
                process = psutil.Process(child.pid)
                for p in reversed([process, *process.children(recursive=True)]):
                    try:
                        p.kill()
                    except psutil.NoSuchProcess:
                        pass
                child.wait(timeout=15)
            except psutil.NoSuchProcess:
                pass
        if acquired:
            lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("doctor", "execute"):
        p = sub.add_parser(name)
        p.add_argument("--profile", choices=["pro360", "ultra5060"], required=True)
        if name == "execute":
            p.add_argument("--kind", choices=["cpu", "gpu"], required=True)
            p.add_argument("--timeout", type=float, default=3600)
            p.add_argument("--name", default="job")
            p.add_argument("child_command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    config = json.loads((ROOT / "configs" / (args.profile + ".json")).read_text(encoding="utf-8"))
    label = args.command if args.command == "doctor" else "".join(c if c.isalnum() or c in "-_" else "-" for c in args.name)[:40]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + label + "_" + uuid.uuid4().hex[:6]
    run = ROOT / "runs" / run_id
    run.mkdir(parents=True)
    try:
        git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:  # no git on PATH (the Pro may use a portable MinGit)
        git_head = None
    metadata = {"run_id": run_id, "started_utc": datetime.now(timezone.utc).isoformat(), "host": platform.node(),
                "command": sys.argv, "profile": config, "git_head": git_head, "python": sys.executable, "packages": versions()}
    started = time.monotonic()
    try:
        metadata["result"] = globals()[args.command](args, config, run)
        metadata["status"] = "passed"
        status = 0
    except Exception as error:
        metadata["status"] = "failed"
        metadata["error"] = f"{type(error).__name__}: {error}"
        (run / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        status = 1
    metadata["elapsed_s"] = round(time.monotonic() - started, 1)
    write_json(run / "run.json", metadata)
    print(json.dumps({"status": metadata["status"], "run": str(run), "result": metadata.get("result"),
                      "error": metadata.get("error")}, ensure_ascii=False, indent=2))
    return status


if __name__ == "__main__":
    raise SystemExit(main())

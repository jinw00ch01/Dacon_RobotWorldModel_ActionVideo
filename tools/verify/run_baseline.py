"""Run the official baseline (copied out of open/baseline) on an eval-shaped folder, for score calibration only.

The copy lives in --baseline (default C:\\Dacon\\WM_Shared\\baseline_run, with checkpoints/backbone.ckpt =
DynamiCrafter_512 and checkpoints/baseline_diffusion.ckpt) and runs in its own venv. open/ is never written.

  python tools/verify/run_baseline.py --inputs C:/Dacon/WM_Shared/holdout_v1 --out C:/Dacon/WM_Shared/holdout_v1/pred/baseline
"""
import argparse, os, subprocess, sys

CREATE_NO_WINDOW = 0x08000000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=r"C:\Dacon\WM_Shared\baseline_run")
    ap.add_argument("--python", default=r"C:\Dacon\WM_Runtime\venv-baseline\Scripts\python.exe")
    ap.add_argument("--inputs", default=r"C:\Dacon\WM_Shared\holdout_v1")
    ap.add_argument("--out", default=r"C:\Dacon\WM_Shared\holdout_v1\pred\baseline")
    ap.add_argument("--stats", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "open", "data",
                                                    "train", "so100_action_statistics.json"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cmd = [args.python, "scripts/inference/generate_baseline_videos.py",
           "--checkpoint", "../checkpoints/baseline_diffusion.ckpt",
           "--challenge-root", args.inputs, "--prediction-root", args.out,
           "--action-stats-path", os.path.abspath(args.stats), "--seed", str(args.seed)]
    print(" ".join(cmd), flush=True)
    env = dict(os.environ, PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python", PYTHONUNBUFFERED="1")
    return subprocess.run(cmd, cwd=os.path.join(args.baseline, "challenge_kit"), env=env,
                          creationflags=CREATE_NO_WINDOW if os.name == "nt" else 0).returncode


if __name__ == "__main__":
    sys.exit(main())

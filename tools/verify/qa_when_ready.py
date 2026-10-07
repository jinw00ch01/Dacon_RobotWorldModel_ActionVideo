"""Wait until a prediction folder is complete, then run qa_sheet and action_direction on it.

Meant to run as a runner CPU job so it survives the agent session parking:
  python tools/verify/qa_when_ready.py --pred <dir of mp4> --holdout <eval-shaped dir> --name <tag> [--pred ... --holdout ... --name ...]
Outputs: C:/Dacon/WM_Shared/verify_qa/<name>/ (sheets, qa.csv) and verify_qa/<name>_action_direction.json.
For eval submission folders pass --format-only (contact sheet + format checks only): eval actions are inference
input, never a metric to tune or select against.
"""
import argparse, glob, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
CREATE_NO_WINDOW = 0x08000000


def wait_complete(pred, n, timeout, settle=60):
    t0, last, since = time.time(), -1, time.time()
    while time.time() - t0 < timeout:
        k = len(glob.glob(os.path.join(pred, "*.mp4")))
        if k != last:
            last, since = k, time.time()
        if k >= n and time.time() - since >= settle:
            return True
        time.sleep(30)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", action="append", required=True)
    ap.add_argument("--holdout", action="append", required=True)
    ap.add_argument("--name", action="append", required=True)
    ap.add_argument("--qa-root", default=r"C:\Dacon\WM_Shared\verify_qa")
    ap.add_argument("--timeout", type=float, default=6 * 3600)
    ap.add_argument("--score", action="store_true", help="also run feature_scores on CPU with --idm")
    ap.add_argument("--idm", default=r"C:\Dacon\WM_Shared\idm\idm_v2.pt")
    ap.add_argument("--score-root", default=r"C:\Dacon\WM_Shared\verify_scores")
    ap.add_argument("--format-only", action="store_true",
                    help="only the contact sheet and format checks; use for eval outputs (no metrics against eval actions)")
    args = ap.parse_args()
    if args.format_only and args.score:
        ap.error("--format-only and --score are exclusive")
    t0, failed = time.time(), 0
    for pred, hold, name in zip(args.pred, args.holdout, args.name):
        n = len(glob.glob(os.path.join(hold, "images", "*.png")))
        if not wait_complete(pred, n, args.timeout - (time.time() - t0)):
            print(f"{name}: timed out waiting for {n} mp4 in {pred}", flush=True)
            failed += 1
            continue
        flags = CREATE_NO_WINDOW if os.name == "nt" else 0
        cmds = [[sys.executable, os.path.join(HERE, "qa_sheet.py"), "--pred", pred, "--images", os.path.join(hold, "images"),
                     "--out", os.path.join(args.qa_root, name)],
                    [sys.executable, os.path.join(HERE, "action_direction.py"), "--pred", pred, "--holdout", hold,
                     "--out", os.path.join(args.qa_root, name + "_action_direction.json")]]
        if args.format_only:
            cmds = cmds[:1]
        if args.score:
            cmds.append([sys.executable, os.path.join(HERE, "feature_scores.py"), "--ref", os.path.join(hold, "gt_videos"),
                         "--pred", pred, "--out", os.path.join(args.score_root, f"{name}_idm_v2_cpu.csv"), "--idm", args.idm,
                         "--device", "cpu"])
        for cmd in cmds:
            r = subprocess.run(cmd, capture_output=True, text=True, creationflags=flags)
            print(name, r.stdout.strip()[-2000:], r.stderr.strip()[-500:] if r.returncode else "", flush=True)
            failed += r.returncode != 0
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

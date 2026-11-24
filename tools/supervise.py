"""Keep a long training run alive: rerun it with --resume until it finishes.

    python tools/supervise.py --log data/build/train_base.log -- tools/train_parser_tf.py --model ... --resume

The trainer checkpoints mid-epoch (--ckpt-min) and at every epoch end, so a run killed by
low memory, a crash or a reboot-free hiccup loses at most a few minutes of work. This
watchdog restarts it after a short pause, up to --tries times, and stops as soon as the
trainer exits cleanly. Launch it detached (tools/train_detached.ps1) so it does not belong
to the terminal or editor session that started it.
"""
import argparse
import datetime
import os
import subprocess
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--tries", type=int, default=20)
    ap.add_argument("--pause", type=float, default=60.0, help="seconds to wait before a restart")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if "--resume" not in cmd:
        cmd.append("--resume")
    env = dict(os.environ, PYTHONUNBUFFERED="1", HF_HUB_DISABLE_SYMLINKS_WARNING="1")
    for attempt in range(1, a.tries + 1):
        with open(a.log, "a", encoding="utf-8") as log:
            log.write(f"\n=== supervise: attempt {attempt} at {datetime.datetime.now():%Y-%m-%d %H:%M:%S} ===\n")
            log.flush()
            rc = subprocess.call([sys.executable] + cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
            log.write(f"=== supervise: exit code {rc} ===\n")
        if rc == 0:
            return 0
        time.sleep(a.pause)
    return 1


if __name__ == "__main__":
    sys.exit(main())

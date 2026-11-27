"""Restart every unfinished corpus build (and the progress monitor) in a folder.

    python tools/resume_builds.py data/build/corpus-v2 [--device cuda]
    python tools/resume_builds.py data/build/corpus-v2 --install-startup   # also after a reboot
    python tools/resume_builds.py data/build/corpus-v2 --status

Each build folder's run.json names its input files; a build is finished when
its progress.json has read every document.  Unfinished builds that are not
already running are started with --resume (tools/build_corpus.py appends after
the last finished document, so nothing is lost or written twice).

--install-startup puts a small script in the Windows Startup folder that runs
this at login, so builds continue after an unexpected restart (power cut,
crash).  The script removes itself once every build has finished.
"""
import argparse
import json
import re
import os
import subprocess
import time
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
STARTUP = os.path.join(os.environ.get("APPDATA", ""), "Microsoft", "Windows", "Start Menu", "Programs",
                       "Startup", "ugl-resume-corpus.cmd")
DETACHED = 0x00000008 | 0x00000200 | 0x08000000   # DETACHED_PROCESS | NEW_PROCESS_GROUP | CREATE_NO_WINDOW


def out_folder(command):
    """The --out folder of a build_corpus command line (relative to the repo), or None."""
    m = re.search(r'--out\s+("([^"]+)"|(\S+))', command)
    return None if not m else os.path.join(ROOT, m.group(2) or m.group(3))


def running_commands():
    out = subprocess.run(["powershell", "-NoProfile", "-c",
                          "Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | "
                          "ForEach-Object { $_.CommandLine }"], capture_output=True, text=True).stdout
    return out.splitlines()


def total_documents(files):
    from build_corpus import document_units
    return sum(1 for _ in document_units(files))


def read_progress(path, tries=10):
    """Documents finished according to a build's progress.json (0 if none).
    Builds started before checkpoints were written atomically rewrite the file
    in place, so a read can land mid-write: wait and read again."""
    for _ in range(tries):
        if not os.path.exists(path):
            return 0
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)["docs"]
        except (ValueError, KeyError, OSError):
            time.sleep(0.5)
    raise RuntimeError(f"{path} stayed unreadable")


def state(folder):
    """[(name, files, docs done, docs total)] for every build folder."""
    out = []
    for name in sorted(os.listdir(folder)):
        d = os.path.join(folder, name)
        if not os.path.exists(os.path.join(d, "run.json")):
            continue
        with open(os.path.join(d, "run.json"), encoding="utf-8") as f:
            run = json.load(f)
        done = read_progress(os.path.join(d, "progress.json"))
        if run.get("finished"):
            # finished builds say so: counting their documents again would give a
            # different total whenever document_units changes (long documents now
            # go in pieces), and the build would be resumed and write a tail twice
            out.append((name, run["files"], done, done))
            continue
        out.append((name, run["files"], done, total_documents(run["files"])))
    return out


def start(args, log):
    env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2",
               PYTHONIOENCODING="utf-8")
    with open(log + ".log", "a") as o, open(log + ".err", "a") as e:
        return subprocess.Popen([PY, *args], cwd=ROOT, stdout=o, stderr=e, env=env, creationflags=DETACHED).pid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--install-startup", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--max", type=int, default=None,
                    help="keep at most this many builds running (more GPU processes than fit in its memory crash)")
    ap.add_argument("--watch", type=int, default=None, metavar="SECONDS",
                    help="keep checking: whenever fewer than --max run, resume the next unfinished build")
    a = ap.parse_args()
    while True:
        try:
            unfinished = check(a)
        except Exception:  # noqa: BLE001  one failed pass must not end an unattended watch
            if not a.watch:
                raise
            import traceback
            traceback.print_exc()
            unfinished = True
        if not a.watch or not unfinished:
            break
        time.sleep(a.watch)


def check(a):
    """One pass: report every build, resume stopped ones (up to --max running).
    -> the unfinished builds."""
    folder = os.path.abspath(a.folder)
    builds = state(folder)
    procs = running_commands()
    unfinished = [b for b in builds if b[2] < b[3]]
    # by the exact --out folder: a name test ("tinystories-8000" in the command line)
    # took a running tinystories-8000-b for tinystories-8000, which then never resumed
    outs = {os.path.normcase(os.path.abspath(m)) for c in procs if "build_corpus" in c
            for m in [out_folder(c)] if m}
    live = {name for name, *_ in builds if os.path.normcase(os.path.join(folder, name)) in outs}
    slots = (a.max - len(live)) if a.max else len(builds)
    print(time.strftime("%H:%M"), flush=True)
    # TinyStories first when a slot opens: the corpus is meant to be mostly stories
    for name, files, done, total in sorted(builds, key=lambda b: not b[0].startswith("tinystories")):
        print(f"{name:26} {done:6}/{total:<6} documents  "
              f"{'running' if name in live else 'finished' if done >= total else 'stopped'}", flush=True)
        if a.status or done >= total or name in live or slots <= 0:
            continue
        slots -= 1
        pid = start(["tools/build_corpus.py", *files, "--out", os.path.join(folder, name), "--resume",
                     "--device", a.device], os.path.join(folder, name))
        print(f"  resumed, pid {pid}")
    if not a.status and unfinished and not any("corpus_monitor" in c for c in procs):
        pid = start(["tools/corpus_monitor.py", folder], folder + "-monitor")
        print(f"monitor started, pid {pid}")
    if a.install_startup and unfinished:
        with open(STARTUP, "w") as f:
            f.write(f'@echo off\r\ncd /d "{ROOT}"\r\n"{PY}" tools\\resume_builds.py "{folder}" --device {a.device}\r\n')
        print("startup entry:", STARTUP)
    if not unfinished and os.path.exists(STARTUP):
        os.remove(STARTUP)                  # every build is done: nothing to resume after a reboot
        print("all builds finished; startup entry removed")
    return unfinished


if __name__ == "__main__":
    main()

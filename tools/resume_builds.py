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
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
STARTUP = os.path.join(os.environ.get("APPDATA", ""), "Microsoft", "Windows", "Start Menu", "Programs",
                       "Startup", "ugl-resume-corpus.cmd")
DETACHED = 0x00000008 | 0x00000200 | 0x08000000   # DETACHED_PROCESS | NEW_PROCESS_GROUP | CREATE_NO_WINDOW


def running_commands():
    out = subprocess.run(["powershell", "-NoProfile", "-c",
                          "Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | "
                          "ForEach-Object { $_.CommandLine }"], capture_output=True, text=True).stdout
    return out.splitlines()


def total_documents(files):
    from build_corpus import document_units
    return sum(1 for _ in document_units(files))


def state(folder):
    """[(name, files, docs done, docs total)] for every build folder."""
    out = []
    for name in sorted(os.listdir(folder)):
        d = os.path.join(folder, name)
        if not os.path.exists(os.path.join(d, "run.json")):
            continue
        with open(os.path.join(d, "run.json"), encoding="utf-8") as f:
            run = json.load(f)
        done = 0
        if os.path.exists(os.path.join(d, "progress.json")):
            with open(os.path.join(d, "progress.json"), encoding="utf-8") as f:
                done = json.load(f)["docs"]
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
    a = ap.parse_args()
    folder = os.path.abspath(a.folder)
    builds = state(folder)
    procs = running_commands()
    unfinished = [b for b in builds if b[2] < b[3]]
    for name, files, done, total in builds:
        live = any(os.path.join(folder, name) in c or f"\\{name}" in c for c in procs if "build_corpus" in c)
        print(f"{name:26} {done:6}/{total:<6} documents  {'running' if live else 'finished' if done >= total else 'stopped'}")
        if a.status or done >= total or live:
            continue
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


if __name__ == "__main__":
    main()

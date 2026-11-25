"""Publish corpus-build progress for the playground's Corpus page.

    python tools/corpus_monitor.py data/build/corpus-v2            # every 10 minutes, forever
    python tools/corpus_monitor.py data/build/corpus-v2 --once     # one update

Every interval it reads each build folder (tools/build_corpus.py output),
counts kept sentence pairs, English words and UGL tokens, estimates time left
from run.json (the build's input files), takes the latest kept passages with
the decoder's English for their UGL, and publishes stats.json on the
`corpus-stats` branch (one commit, replaced each time, so the branch never
grows).  site/corpus.html reads it from raw.githubusercontent.com.

Nothing but these numbers and sample sentences (from public datasets) leaves
the computer.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

BRANCH = "corpus-stats"
WORKTREE = os.path.join(ROOT, "data", "build", "stats-site")


def git(*args, cwd=WORKTREE, check=True):
    return subprocess.run(["git", *args], cwd=cwd, check=check, capture_output=True, text=True)


def ensure_worktree():
    """A worktree on the orphan stats branch, outside the main checkout's files."""
    if os.path.exists(os.path.join(WORKTREE, ".git")):
        return
    git("worktree", "add", "--detach", WORKTREE, cwd=ROOT)
    git("checkout", "--orphan", BRANCH)
    git("rm", "-rf", "--quiet", ".", check=False)


def count_lines_words(path):
    lines = words = 0
    if os.path.exists(path):
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                lines += 1
                words += len(line.split())
    return lines, words


_TOTALS = {}


def total_sentences(files):
    """Sentences the build will read in all (as tools/build_corpus.py splits them)."""
    key = tuple(files)
    if key not in _TOTALS:
        from ugl.corpus import documents, paragraphs
        from ugl.parser.tokenize import split_sentences
        n = 0
        for _, text in documents(files):
            paras = [p for p in paragraphs(text) if len(p) >= 40 or re.search(r"[.!?][\"'”’)]*$", p)]
            n += sum(len(split_sentences(p)) for p in paras)
        _TOTALS[key] = n
    return _TOTALS[key]


def last_passages(path, k):
    """The last k passages of a passages.jsonl (read from the end: the file grows)."""
    if not os.path.exists(path):
        return []
    with open(path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - 200_000))
        tail = f.read().decode("utf-8", errors="replace").splitlines()[1:]
    out = []
    for line in tail[-k:]:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def snapshot(folder, u, history):
    now = time.time()
    builds, recent = [], []
    for name in sorted(os.listdir(folder)):
        d = os.path.join(folder, name)
        if not os.path.isdir(d) or not os.path.exists(os.path.join(d, "english.txt")):
            continue
        kept, en_words = count_lines_words(os.path.join(d, "english.txt"))
        _, ugl_tokens = count_lines_words(os.path.join(d, "ugl.txt"))
        rejected, _ = count_lines_words(os.path.join(d, "rejected.tsv"))
        seen = kept + rejected
        run = {}
        if os.path.exists(os.path.join(d, "run.json")):
            with open(os.path.join(d, "run.json"), encoding="utf-8") as f:
                run = json.load(f)
        total = run.get("max_sents") or (total_sentences(run["files"]) if run.get("files") else None)
        started = run.get("started")
        last_write = max(os.path.getmtime(os.path.join(d, x)) for x in ("english.txt", "rejected.tsv"))
        running = now - last_write < 900 and (total is None or seen < total)
        # sentences / hour since the build last started or resumed
        done_here = seen - run.get("seen_at_start", 0)
        rate = done_here / ((now - started) / 3600) if started and now > started and done_here > 0 else None
        eta = (total - seen) / rate if running and rate and total else None
        builds.append({"name": name, "kept": kept, "seen": seen, "english_words": en_words,
                       "ugl_tokens": ugl_tokens, "total": total, "running": running,
                       "rate_per_hour": round(rate) if rate else None, "device": run.get("device") or "cpu",
                       "eta_hours": round(eta, 2) if eta is not None else None})
        for p in last_passages(os.path.join(d, "passages.jsonl"), 2):
            try:
                back = [u.decode(x) for x in p["ugl"]]
            except Exception:  # noqa: BLE001
                back = []
            recent.append({"build": name, "english": p["english"], "ugl": p["ugl"], "decoded": back})
    tot = {k: sum(b[k] for b in builds) for k in ("kept", "seen", "english_words", "ugl_tokens")}
    history.append({"t": round(now), "kept": tot["kept"], "english_words": tot["english_words"]})
    # sentences/hour and words/hour over the last hour of history
    hour = [h for h in history if now - h["t"] <= 3700]
    span = (hour[-1]["t"] - hour[0]["t"]) / 3600 if len(hour) > 1 else 0
    return {"updated": round(now), "folder": os.path.relpath(folder, ROOT).replace(os.sep, "/"),
            "totals": tot,
            "kept_per_hour": round((hour[-1]["kept"] - hour[0]["kept"]) / span) if span else None,
            "words_per_hour": round((hour[-1]["english_words"] - hour[0]["english_words"]) / span) if span else None,
            "running": sum(b["running"] for b in builds),
            "eta_hours": max((b["eta_hours"] for b in builds if b["eta_hours"] is not None), default=None),
            "builds": builds, "recent": recent[-12:], "history": history[-2000:]}


def publish(stats):
    with open(os.path.join(WORKTREE, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False)
    git("add", "stats.json")
    # one commit, replaced each time: the branch holds only the latest numbers
    has_head = git("rev-parse", "--verify", "HEAD", check=False).returncode == 0
    git("commit", "--quiet", *(["--amend"] if has_head else []), "-m", "corpus build progress")
    git("push", "--quiet", "--force", "origin", f"HEAD:{BRANCH}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--interval", type=int, default=600, help="seconds between updates")
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    import ugl
    u = ugl.UGL(wsd=False)
    ensure_worktree()
    history = []
    old = os.path.join(WORKTREE, "stats.json")
    if os.path.exists(old):
        with open(old, encoding="utf-8") as f:
            history = json.load(f).get("history", [])
    while True:
        try:
            stats = snapshot(os.path.abspath(a.folder), u, history)
            publish(stats)
            t = stats["totals"]
            print(time.strftime("%H:%M"), f"{t['kept']} pairs, {t['english_words']} words, "
                  f"{stats['running']} builds running", flush=True)
        except Exception as e:  # noqa: BLE001  (keep publishing; a bad cycle is not fatal)
            print(time.strftime("%H:%M"), "update failed:", e, flush=True)
        if a.once:
            break
        time.sleep(a.interval)


if __name__ == "__main__":
    main()

"""Reading raw text corpora: one place that knows the file formats.

    .jsonl  one article per line: {"title": ..., "text": ...}   (scripts/fetch_wikipedia.py)
    .txt    plain text, paragraphs separated by newlines; no titles or
            article boundaries (the older format)

documents(paths) yields (title, text) pairs; title is None for .txt files,
whose whole content counts as one document.
"""
import glob
import json
import os


def raw_files(folder):
    return sorted(glob.glob(os.path.join(folder, "**", "*.jsonl"), recursive=True)
                  + glob.glob(os.path.join(folder, "**", "*.txt"), recursive=True))


def documents(paths):
    for path in paths:
        with open(path, encoding="utf-8") as f:
            if path.endswith(".jsonl"):
                for line in f:
                    line = line.strip()
                    if line:
                        d = json.loads(line)
                        yield d.get("title") or None, d.get("text", "")
            else:
                yield None, f.read()


def paragraphs(text):
    return [p.strip() for p in text.split("\n") if p.strip()]

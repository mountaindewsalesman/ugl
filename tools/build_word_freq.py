"""Count how often English words are used, for the decoder's word choice.

    python tools/build_word_freq.py      # -> ugl/dictionary/wordfreq.json.gz

A UGL concept can be said with any of the words that mean it (girl / lass,
big / large, mom / mummy).  WordNet lists them in no particular order of use,
so the decoder picks the one English uses most (ugl/lexicon.py surface()).
Counts come from two halves weighted equally, so neither register dominates:
stories (the TinyStories text in data/raw) and general text (Wikipedia,
FineWeb-Edu, news, Gutenberg, Simple Wikipedia).  Stored: lowercase word
form -> uses per million words, for the 200,000 commonest forms.  Independent
of the dictionary version (it is about English, not about UGL numbers).
"""
import glob
import gzip
import json
import os
import re
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "ugl", "dictionary", "wordfreq.json.gz")
WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")


def texts(paths, limit_words):
    n = 0
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if path.endswith(".jsonl"):
                    try:
                        line = json.loads(line).get("text", "")
                    except ValueError:
                        continue
                yield line
                n += line.count(" ")
                if n >= limit_words:
                    return


def count(paths, limit_words):
    c = Counter()
    for t in texts(paths, limit_words):
        c.update(WORD.findall(t.lower()))
    return c


def main():
    stories = sorted(glob.glob(os.path.join(RAW, "tinystories*", "*.jsonl")))
    general = sorted(glob.glob(os.path.join(RAW, "dump-*.txt")) + glob.glob(os.path.join(RAW, "api-*.jsonl"))
                     + [p for s in ("fineweb-edu", "news", "gutenberg", "simplewiki")
                        for p in glob.glob(os.path.join(RAW, s, "*.jsonl"))])
    halves = [count(stories, 20_000_000), count(general, 20_000_000)]
    per_million = Counter()
    for c in halves:
        total = sum(c.values())
        for w, k in c.items():
            per_million[w] += 0.5 * k * 1e6 / total
    top = {w: round(f, 3) for w, f in per_million.most_common(200_000)}
    with gzip.open(OUT, "wt", encoding="utf-8") as f:
        json.dump({"about": "English word form -> uses per million words (half stories, half general text); "
                            "tools/build_word_freq.py", "words": [sum(c.values()) for c in halves], "freq": top}, f)
    print(f"wrote {OUT}: {len(top):,} forms from {sum(sum(c.values()) for c in halves):,} words")


if __name__ == "__main__":
    main()

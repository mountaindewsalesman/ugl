"""Sample sentences per rejection reason for grading by hand.

    python tools/review_sample.py data/raw/*/*-0000.jsonl --per-file 250 --per-bucket 20 --out review.jsonl

Encodes the text the way tools/build_corpus.py does (base parser + agreement,
round trip), puts every sentence in one bucket by its exact set of issue kinds
(kept, round trip only, uncertain parse only (split by the lowest confidence
among the flagged words), ...), and writes a random sample of each bucket with
the source English, the UGL and the decoder's English for the UGL.

A reviewer then grades each line:  ok (the UGL says what the English says),
minor (a small loss: a dropped nuance word, a slightly different sense), or
wrong (a different meaning).  The share of ok+minor per bucket is what each
filter costs in usable sentences; see tools/review_score.py.
"""
import argparse
import collections
import json
import os
import pickle
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ugl  # noqa: E402
from ugl.corpus import documents, paragraphs  # noqa: E402
from ugl.parser.model import ParserModel  # noqa: E402
from build_corpus import kind, BASE, DEFAULT_PATH  # noqa: E402
from roundtrip_report import cached  # noqa: E402


def bucket(kinds, issues):
    if not kinds:
        return "kept"
    if kinds == {"uncertain parse"}:
        conf = min(int(m) for m in re.findall(r"\((\d+)% sure", " ".join(issues))) / 100
        return "uncertain parse 0.6-0.8" if conf >= 0.6 else "uncertain parse <0.6"
    if len(kinds) == 1:
        return next(iter(kinds)) + " only"
    return "several"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--per-file", type=int, default=250, help="sentences read from each file")
    ap.add_argument("--per-bucket", type=int, default=20)
    ap.add_argument("--cache", default=None, help="parser score cache (tools/roundtrip_report.py --cache)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    u = ugl.UGL()
    u.encoder.unsure = 0.8
    u.encoder.parser = ParserModel(BASE)
    u.encoder.second_parser = ParserModel(DEFAULT_PATH)
    cache = {}
    if a.cache and os.path.exists(a.cache):
        with open(a.cache, "rb") as f:
            cache = pickle.load(f)
    for m in (u.encoder.parser, u.encoder.second_parser):
        m.score = cached(m, cache)
    buckets = collections.defaultdict(list)
    for path in a.files:
        source = os.path.basename(os.path.dirname(os.path.abspath(path)))
        n = 0
        for title, text in documents([path]):
            if n >= a.per_file:
                break
            for para in paragraphs(text):
                if len(para) < 40 or n >= a.per_file:
                    continue
                by_src = collections.OrderedDict()
                for line in u.encoder.encode_lines(para, title):
                    by_src.setdefault(line.source, []).append(line)
                for src, lines in by_src.items():
                    if n >= a.per_file:
                        break
                    n += 1
                    issues = [i for l in lines for i in l.issues]
                    ugl_lines = [l.ugl for l in lines]
                    try:
                        eng = u.decode("\n".join(ugl_lines))
                        if not u.same_reading(ugl_lines, [l.ugl for l in u.encoder.encode_lines(eng)]):
                            issues.append("round trip changed the UGL")
                    except Exception as e:  # noqa: BLE001
                        eng = f"<{type(e).__name__}>"
                        issues.append("round trip failed")
                    kinds = {kind(i) for i in issues}
                    buckets[bucket(kinds, issues)].append(
                        {"source": source, "src": src, "ugl": ugl_lines, "eng": eng, "issues": issues,
                         "bucket": bucket(kinds, issues)})
    if a.cache:
        with open(a.cache, "wb") as f:
            pickle.dump(cache, f)
    rng = random.Random(a.seed)
    total = sum(len(v) for v in buckets.values())
    with open(a.out, "w", encoding="utf-8") as f:
        for b, items in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
            print(f"{b:32} {len(items):5}  ({len(items) / total:.1%})")
            for it in rng.sample(items, min(a.per_bucket, len(items))):
                it["bucket_share"] = len(items) / total
                f.write(json.dumps(it, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

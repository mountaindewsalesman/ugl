"""Where does the corpus pipeline lose sentences that the parser got right?

    python tools/roundtrip_report.py data/raw/api-20261124-102108-0000.jsonl --n 400 --out report.txt

For each English sentence (paragraph by paragraph, as tools/build_corpus.py
reads them) it records every issue the encoder raises plus the round trip.
Prints:
  kept          sentences with no issue at all (what the corpus keeps)
  ceiling       sentences the parser is sure of (no uncertain parse, no parser
                disagreement): the best the rest of the pipeline could keep
  per kind      how many of the ceiling sentences each other issue costs
and writes every round-trip failure inside the ceiling to --out as
SRC / UGL / ENG / UGL2 blocks, the material for fixing encoder and decoder bugs.
"""
import argparse
import collections
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ugl  # noqa: E402
from ugl.corpus import documents, paragraphs  # noqa: E402
from ugl.parser.model import ParserModel  # noqa: E402
from build_corpus import kind, BASE, DEFAULT_PATH  # noqa: E402

PARSER_KINDS = {"uncertain parse", "parsers disagree"}


def cached(model, cache):
    score = model.score
    tag = os.path.basename(model.path)

    def run(words):
        key = (tag, tuple(words))
        if key not in cache:
            cache[key] = score(words)
        return cache[key]
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--skip", type=int, default=0, help="sentences to skip first (a held-out slice)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--min-confidence", type=float, default=0.8)
    ap.add_argument("--base", action="store_true", help="base parser + agreement (slow); default small parser only")
    ap.add_argument("--cache", default=None,
                    help="pickle of parser scores by sentence, reused across runs (the parse of a "
                         "sentence does not change while the encoder and decoder are being fixed)")
    a = ap.parse_args()
    u = ugl.UGL()
    u.encoder.unsure = a.min_confidence
    if a.base:
        u.encoder.parser = ParserModel(BASE)
        u.encoder.second_parser = ParserModel(DEFAULT_PATH)
    cache = {}
    if a.cache and os.path.exists(a.cache):
        with open(a.cache, "rb") as f:
            cache = pickle.load(f)
    for m in (u.encoder.parser, getattr(u.encoder, "second_parser", None)):
        if m is not None:
            m.score = cached(m, cache)
    out = open(a.out, "w", encoding="utf-8") if a.out else None
    seen = kept = ceiling = ceiling_kept = 0
    cost = collections.Counter()
    for title, text in documents(a.files):
        if seen >= a.n:
            break
        for para in paragraphs(text):
            if len(para) < 40 or seen >= a.n:
                continue
            by_src = collections.OrderedDict()
            for line in u.encoder.encode_lines(para, title):
                by_src.setdefault(line.source, []).append(line)
            for src, lines in by_src.items():
                if a.skip > 0:
                    a.skip -= 1
                    continue
                if seen >= a.n:
                    break
                seen += 1
                issues = [i for l in lines for i in l.issues]
                ugl_lines = [l.ugl for l in lines]
                try:
                    eng = u.decode("\n".join(ugl_lines))
                    again = [l.ugl for l in u.encoder.encode_lines(eng)]
                    if not u.same_reading(ugl_lines, again):
                        issues.append("round trip changed the UGL")
                except Exception as e:  # noqa: BLE001
                    eng, again = f"<{type(e).__name__}: {e}>", []
                    issues.append("round trip failed")
                kinds = {kind(i) for i in issues}
                kept += not kinds
                if kinds & PARSER_KINDS:
                    continue
                ceiling += 1
                ceiling_kept += not kinds
                cost.update(kinds)
                if out and kinds:
                    out.write(f"SRC  {src}\nISS  {' | '.join(issues)}\nUGL  {' / '.join(ugl_lines)}\n"
                              f"ENG  {eng}\nUGL2 {' / '.join(again)}\n\n")
    if a.cache:
        with open(a.cache, "wb") as f:
            pickle.dump(cache, f)
    print(f"{seen} sentences: kept {kept / max(1, seen):.1%}; parser-sure ceiling {ceiling / max(1, seen):.1%}; "
          f"kept inside ceiling {ceiling_kept / max(1, ceiling):.1%}")
    print("issues among the parser-sure sentences:")
    for k, n in cost.most_common():
        print(f"  {k:22} {n:5}  ({n / max(1, ceiling):.1%})")


if __name__ == "__main__":
    main()

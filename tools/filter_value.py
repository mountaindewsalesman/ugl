"""What does each corpus filter buy?  Measured against hand-annotated trees.

    python tools/filter_value.py --n 1000 [--base]

For EWT test sentences the UGL written from the gold tree is taken as the right
answer.  The sentence is also encoded the way tools/build_corpus.py does it
(parser, flags, round trip).  For each filter stage: the share of sentences
kept, and the share of kept sentences whose UGL equals the gold-tree UGL.  A
filter is worth having when it raises that share; one that throws away right
sentences as often as wrong ones only costs data.

Sentences whose own tokenization differs from the treebank's are skipped, so
both sides see the same words.
"""
import argparse
import collections
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import ugl  # noqa: E402
from ugl.parser.conllu import read_conllu  # noqa: E402
from ugl.parser.model import ParserModel  # noqa: E402
from ugl.parser.tokenize import tokenize  # noqa: E402
from build_corpus import BASE, DEFAULT_PATH, kind  # noqa: E402
from compare_encoders import encode_tree  # noqa: E402

PARSER_KINDS = {"uncertain parse", "parsers disagree"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--base", action="store_true", help="base parser + agreement, as the corpus builder")
    ap.add_argument("--min-confidence", type=float, default=0.8)
    a = ap.parse_args()
    u = ugl.UGL()
    u.encoder.unsure = a.min_confidence
    if a.base:
        u.encoder.parser = ParserModel(BASE)
        u.encoder.second_parser = ParserModel(DEFAULT_PATH)
    sents = list(read_conllu(os.path.join(ROOT, "data", "ud", "en_ewt-ud-test.conllu")))
    random.Random(0).shuffle(sents)
    rows = []          # (kinds without round trip, round trip ok, exact)
    for s in sents:
        if len(rows) >= a.n:
            break
        words = [t["form"] for t in s]
        if not 4 <= len(words) <= 40 or tokenize(" ".join(words)) != words:
            continue
        gold, err = encode_tree(u.lexicon, [{k: v for k, v in t.items() if k != "lemma"} for t in s],
                                words, words[-1] == "?")
        if err:
            continue
        lines = u.encoder.encode_lines(" ".join(words))
        if len(lines) == 0 or len({l.source for l in lines}) != 1:
            continue                                  # split into two sentences: not comparable
        ugl_lines = [l.ugl for l in lines]
        kinds = {kind(i) for l in lines for i in l.issues}
        try:
            again = [l.ugl for l in u.encoder.encode_lines(u.decode("\n".join(ugl_lines)))]
            rt = u.same_reading(ugl_lines, again)
        except Exception:  # noqa: BLE001
            rt = False
        rows.append((kinds, rt, "\n".join(ugl_lines) == gold))

    def line(name, keep):
        kept = [r for r in rows if keep(r)]
        right = sum(r[2] for r in kept)
        print(f"  {name:46} kept {len(kept) / len(rows):6.1%}   right {right / max(1, len(kept)):6.1%}   "
              f"({len(kept)} sentences)")

    print(f"{len(rows)} EWT test sentences; 'right' = same UGL as from the hand-annotated tree")
    line("no filter", lambda r: True)
    line("parser checks", lambda r: not r[0] & PARSER_KINDS)
    line("parser checks + other flags", lambda r: not r[0])
    line("parser checks + other flags + round trip", lambda r: not r[0] and r[1])
    line("  (failed the round trip only)", lambda r: not r[0] and not r[1])
    for k in sorted({k for r in rows for k in r[0]} - PARSER_KINDS):
        line(f"  parser-sure, flagged '{k}'", lambda r, k=k: not r[0] & PARSER_KINDS and k in r[0])
    c = collections.Counter(k for r in rows for k in r[0])
    print("flags:", dict(c.most_common()))


if __name__ == "__main__":
    main()

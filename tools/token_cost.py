"""Token cost of UGL vs English on Wikipedia sentences, per dictionary version.

    python tools/token_cost.py --n 150 ugl-dict-0.2.0.json.gz ugl-dict-0.3.0.json.gz

Counts tokenizer ids (what a model trained on UGL would actually see; names
and unknown words not in the dictionary are spelled out character by
character) per English word.
"""
import argparse
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import ugl  # noqa: E402
from compare_encoders import wiki_sentences  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dictionaries", nargs="+")
    ap.add_argument("--n", type=int, default=150)
    a = ap.parse_args()
    sents = wiki_sentences(a.n)
    words = sum(len(re.findall(r"\w+", s)) for s in sents)
    for d in a.dictionaries:
        u = ugl.load(d, wsd=False)
        ids = lit = 0
        for s in sents:
            lines = u.encode(s)
            ids += len(u.tokenizer.encode("\n".join(lines)))
            lit += sum(1 for l in lines for t in l.split() if t.startswith("'") or t[1:2] == "?")
        print(f"{d:28} {ids / words:.2f} ids/word   {lit} literal tokens (spelled out)")


if __name__ == "__main__":
    main()

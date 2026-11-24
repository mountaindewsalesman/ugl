"""Score the numpy parser (exact tree decoding) on a UD file.

    python tools/eval_parser.py [--model PATH] [--split test|dev] [--n N]
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ugl.parser.conllu import read_conllu
from ugl.parser.model import ParserModel, DEFAULT_PATH

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_PATH)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--decoder", default="eisner", choices=["eisner", "cle"])
    a = ap.parse_args()
    m = ParserModel(a.model, decoder=a.decoder)
    sents = list(read_conllu(os.path.join(ROOT, "data", "ud", f"en_ewt-ud-{a.split}.conllu")))[: a.n]
    tot = up = xp = ua = la = 0
    t0 = time.time()
    for s in sents:
        out = m.parse([t["form"] for t in s])
        for g, p in zip(s, out):
            tot += 1
            up += g["upos"] == p["upos"]
            xp += g["xpos"] == p["xpos"]
            ua += g["head"] == p["head"]
            la += g["head"] == p["head"] and g["deprel"] == p["deprel"]
    dt = time.time() - t0
    print(f"{len(sents)} sentences ({tot} tokens) in {dt:.1f}s = {len(sents)/dt:.0f} sent/s")
    print(f"UPOS {up/tot:.4f}  XPOS {xp/tot:.4f}  UAS {ua/tot:.4f}  LAS {la/tot:.4f}")


if __name__ == "__main__":
    main()

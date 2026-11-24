"""Compare encoders end to end.

    python tools/compare_encoders.py --gold 500          # analyzer on gold UD trees vs predicted trees
    python tools/compare_encoders.py --wiki 150          # old (spaCy) vs new (UGL parser) on Wikipedia
    python tools/compare_encoders.py --wiki 150 --dump compare.md

--gold: for EWT test sentences, encode from the HAND-ANNOTATED tree and from
        the parser's predicted tree.  "exact" = identical UGL; "token F1" =
        overlap of the token multisets.  Gold-tree output is the upper bound the
        parser can reach; failures on gold trees are analyzer/serializer bugs.
--wiki: same random Wikipedia sentences through both encoders: decode
        failures, unparsed sentences, unknown-word tokens, tokens per word.
"""
import argparse
import collections
import glob
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import ugl  # noqa: E402
from ugl.analyze import Analyzer  # noqa: E402
from ugl.coref import resolve  # noqa: E402
from ugl.encoder import Encoder  # noqa: E402
from ugl.parser.conllu import read_conllu  # noqa: E402
from ugl.serialize import Serializer  # noqa: E402
from ugl.simplify import split  # noqa: E402


def f1(a, b):
    ca, cb = collections.Counter(a.split()), collections.Counter(b.split())
    inter = sum((ca & cb).values())
    if not ca and not cb:
        return 1.0
    p = inter / max(1, sum(ca.values()))
    r = inter / max(1, sum(cb.values()))
    return 0 if p + r == 0 else 2 * p * r / (p + r)


def encode_tree(lex, tree, words, question):
    an, se = Analyzer(lex), Serializer(lex)
    try:
        ev = an.sentence(tree, question=question)
        resolve([ev])
        return chr(10).join(se.sentence(part, words) for part in split(ev)), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def gold_eval(u, n, parser_path):
    from ugl.parser.model import ParserModel
    parser = ParserModel(parser_path) if parser_path else ParserModel()
    sents = list(read_conllu(os.path.join(ROOT, "data", "ud", "en_ewt-ud-test.conllu")))
    random.Random(0).shuffle(sents)
    sents = [s for s in sents if 4 <= len(s) <= 30][:n]
    exact = f1s = gold_err = pred_err = dec_err = 0
    errors = collections.Counter()
    for s in sents:
        words = [t["form"] for t in s]
        q = words[-1] == "?"
        # same lemmatizer on both sides, so differences are tree/tag differences only
        gold = [{k: v for k, v in t.items() if k != "lemma"} for t in s]
        g, ge = encode_tree(u.lexicon, gold, words, q)
        pred = parser.parse(words)
        p, pe = encode_tree(u.lexicon, pred, words, q)
        if ge:
            gold_err += 1
            errors[ge.split(":")[0]] += 1
            continue
        try:
            u.decode(g)
        except Exception:
            dec_err += 1
        if pe:
            pred_err += 1
            continue
        exact += g == p
        f1s += f1(g, p)
    n_ok = len(sents) - gold_err
    print(f"{len(sents)} EWT test sentences (4-30 tokens), parser {os.path.basename(parser_path or 'default')}")
    print(f"  analyzer/serializer crashes on GOLD trees : {gold_err}  {dict(errors)}")
    print(f"  gold-tree UGL that fails to decode        : {dec_err}")
    print(f"  predicted-tree crashes                    : {pred_err}")
    print(f"  predicted == gold UGL (exact)             : {exact / n_ok:.1%}")
    print(f"  predicted vs gold token F1                : {f1s / n_ok:.3f}")


def wiki_sentences(n, seed=1):
    from ugl.parser.tokenize import split_sentences
    rng = random.Random(seed)
    files = sorted(glob.glob(os.path.join(ROOT, "data", "raw", "*.txt")))
    out = []
    while len(out) < n:
        with open(rng.choice(files), encoding="utf-8") as f:
            paras = [p for p in f.read().split("\n") if len(p) > 80]
        cand = [s for s in split_sentences(rng.choice(paras)) if 6 <= len(s.split()) <= 30]
        if cand:
            out.append(rng.choice(cand))
    return out


def stats(name, u, enc, sents):
    import re
    words = toks = unk = unparsed = dec_fail = 0
    outs = []
    t0 = time.time()
    for s in sents:
        if hasattr(enc, "memory"):
            enc.memory = []
        lines = enc.encode(s)
        line = "\n".join(lines)
        outs.append(lines)
        words += len(re.findall(r"\w+", s))
        toks += len(line.split())
        unk += len(re.findall(r"(?:^|\s)[nva]\?", line))
        unparsed += line.count("[unparsed") + line.count("[unserializable")
        for l in lines:
            try:
                u.decode(l)
            except Exception:
                dec_fail += 1
    dt = time.time() - t0
    print(f"  {name:8} tokens/word {toks / words:.2f}   unknown-word tokens {unk / max(1, toks):.1%}   "
          f"unparsed {unparsed}   decode failures {dec_fail}   {len(sents) / dt:.1f} sent/s")
    return outs


def wiki_eval(u, n, dump):
    sents = wiki_sentences(n)
    from ugl.encoder_spacy import SpacyEncoder
    old = SpacyEncoder(u.lexicon)
    new = Encoder(u.lexicon)
    print(f"{n} Wikipedia sentences")
    o = stats("old", u, old, sents)
    w = stats("new", u, new, sents)
    if dump:
        with open(dump, "w", encoding="utf-8") as f:
            f.write("# Old (spaCy) vs new (UGL parser) encoder\n\n")
            for s, a, b in zip(sents, o, w):
                f.write(f"**{s}**\n\n")
                for tag, lines in (("old", a), ("new", b)):
                    for l in lines:
                        try:
                            back = u.decode(l)
                        except Exception as e:
                            back = f"<decode error: {e}>"
                        f.write(f"- {tag}: `{l}`\n  - read: {u.humanize(l)}\n  - back: {back}\n")
                f.write("\n")
        print("wrote", dump)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", type=int, default=0)
    ap.add_argument("--wiki", type=int, default=0)
    ap.add_argument("--parser", default=None)
    ap.add_argument("--dump", default=None)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    u = ugl.load(wsd=False)
    if a.gold:
        gold_eval(u, a.gold, a.parser)
    if a.wiki:
        wiki_eval(u, a.wiki, a.dump)


if __name__ == "__main__":
    main()

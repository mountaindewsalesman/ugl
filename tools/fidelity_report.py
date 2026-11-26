"""How much of each source sentence survives English -> UGL -> English?

    python tools/fidelity_report.py data/build/corpus-v2 --n 2000

The corpus filter's round trip only checks that UGL is stable (decode, encode
again, same UGL).  A sentence translated wrongly the same way twice passes.
This report compares the decoded English with the SOURCE English instead,
word by word in meaning: the content words of each side (nouns, names, verbs,
adjectives, adverbs, numbers; lemmatized), matched as

  same / synonym   the same word, or words sharing a WordNet meaning (big ~ large)
  referent         a name or noun from the passage standing where the source
                   had a pronoun (UGL writes out who is meant: expected)
  changed          a source word replaced by a word with another meaning (cry -> call)
  lost             a source word with nothing for it in the output
  added            an output word from nowhere

A sentence is faithful when nothing is changed, lost or added.  Prints the
faithful share and the most frequent changes, losses and additions (the bug
families), with examples; writes data/build/fidelity.tsv.
"""
import argparse
import json
import os
import random
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ugl.fidelity import Meaning, content_words, normalize_source  # noqa: E402  (the corpus filter's own check)


class Words:
    def __init__(self):
        from ugl.parser.model import ParserModel
        from ugl.parser.tokenize import tokenize
        self.parser, self.tokenize = ParserModel(), tokenize
        import ugl
        self.freq = ugl.load(wsd=False).lexicon.word_freq

    def __call__(self, text):
        """-> [(lemma, upos)] content words of an English sentence."""
        w = self.tokenize(text)
        if not w:
            return []
        st = self.parser.score(w)
        return content_words(self.parser.tree(st, self.parser.decode_tree(st["S"])), self.freq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--n", type=int, default=2000, help="sentences to check")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "build", "fidelity.tsv"))
    ap.add_argument("--reencode", action="store_true",
                    help="encode the sampled passages again with the current encoder (with its sense model), "
                         "instead of decoding the UGL stored in the corpus: measures encoder changes too")
    a = ap.parse_args()
    import glob
    import ugl
    u = ugl.load(wsd=a.reencode)
    rng = random.Random(a.seed)
    paths = sorted(glob.glob(os.path.join(a.corpus, "*", "passages.jsonl")))
    picks = []
    for path in paths:                        # a few thousand passages from each build, by position
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            for _ in range(max(1, a.n // len(paths) // 3)):
                f.seek(rng.randrange(size))
                f.readline()
                line = f.readline()
                try:
                    picks.append(json.loads(line))
                except ValueError:
                    pass
    words = Words()
    check = Meaning(u.lexicon)
    rows = []
    for p in picks:
        context_src = " ".join(p["english"])
        ctx_words = {w.lower() for w, _ in words(context_src)}
        pairs = list(zip(p["english"], p["ugl"]))
        if a.reencode:
            # the passage as one text (pronouns resolve across its sentences), each
            # source sentence with the UGL lines made from it
            lines = u.encode_lines("\n".join(p["english"]))
            by_src = {}
            for l in lines:
                by_src.setdefault(l.source.strip(), []).append(l.ugl)
            pairs = [(e, "\n".join(by_src.get(e.strip(), []))) for e in p["english"]]
            pairs = [(e, g) for e, g in pairs if g]
        for e, g in pairs:
            try:
                d = u.decode(g)
            except Exception as ex:  # noqa: BLE001
                d = f"<decode failed: {ex}>"
            src, out = words(normalize_source(e)), words(d)
            r = check.compare(src, out, ctx_words)
            m, ch, lo, ad, rf = r["matched"], r["changed"], r["lost"], r["added"], r["referents"]
            rows.append({"source": p.get("source", "?"), "english": e, "decoded": d, "matched": m,
                         "content": len(src), "changed": ch, "lost": lo, "added": ad, "referents": rf})
            if len(rows) >= a.n:
                break
        if len(rows) >= a.n:
            break
    n = len(rows)
    faithful = sum(not (r["changed"] or r["lost"] or r["added"]) for r in rows)
    words_in = sum(r["content"] for r in rows)
    print(f"{n} kept sentences: {faithful / n:.1%} faithful (nothing changed, lost or added); "
          f"{sum(r['matched'] for r in rows) / max(1, words_in):.1%} of source content words kept")
    print(f"per sentence: {sum(bool(r['changed']) for r in rows) / n:.1%} have a changed word, "
          f"{sum(bool(r['lost']) for r in rows) / n:.1%} a lost word, {sum(bool(r['added']) for r in rows) / n:.1%} an added word")
    for title, key, fmt in (("changed", "changed", lambda x: f"{x[0][0]} -> {x[1][0]}"),
                            ("lost", "lost", lambda x: f"{x[0]} ({x[1]})"),
                            ("added", "added", lambda x: f"{x[0]} ({x[1]})")):
        c = Counter(fmt(x) for r in rows for x in r[key])
        print(f"\nmost often {title} ({sum(c.values())} in all):")
        print("  " + ", ".join(f"{k} {v}" for k, v in c.most_common(30)))
    by_upos = Counter(x[1] for r in rows for x in r["lost"])
    print("\nlost by word class:", dict(by_upos.most_common()))
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("faithful\tchanged\tlost\tadded\tsource\tenglish\tdecoded\n")
        for r in rows:
            f.write("\t".join([str(int(not (r["changed"] or r["lost"] or r["added"]))),
                               ", ".join(f"{s[0]}->{o[0]}" for s, o in r["changed"]),
                               ", ".join(x[0] for x in r["lost"]), ", ".join(x[0] for x in r["added"]),
                               r["source"], r["english"], r["decoded"]]) + "\n")
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()

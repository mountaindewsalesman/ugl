"""Build a UGL training corpus from English text, keeping only sentences the
encoder has no doubts about.

    python tools/build_corpus.py data/raw/*.txt --out corpus/ --max-sents 50000

Writes:
    corpus/ugl.txt        one clean UGL line per line (training data)
    corpus/english.txt    the English sentence each clean line came from
    corpus/rejected.tsv   flagged sentences: English \t issues   (to inspect / fix)
and prints how many sentences were kept and why the others were rejected.

A sentence is rejected when any of its lines has an issue (see
ugl.encoder.Encoder.encode_lines): an uncertain parse, a pronoun with no
antecedent, a word not in the dictionary, a dropped or guessed piece.
Garbage in, garbage out: a smaller clean corpus beats a larger dirty one.
"""
import argparse
import collections
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ugl  # noqa: E402

KIND = [("uncertain grammar", "uncertain parse"), ("refers to nothing", "unresolved pronoun"),
        ("not in the dictionary", "unknown word"), ("no clear grammatical role", "dropped word"),
        ("not in the relation table", "unknown preposition"), ("failed", "crash")]


def kind(issue):
    return next((k for pat, k in KIND if pat in issue), "other")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default="corpus")
    ap.add_argument("--max-sents", type=int, default=None)
    ap.add_argument("--allow", nargs="*", default=[], help="issue kinds to tolerate, e.g. 'unknown word'")
    a = ap.parse_args()
    u = ugl.UGL()
    os.makedirs(a.out, exist_ok=True)
    fu = open(os.path.join(a.out, "ugl.txt"), "w", encoding="utf-8")
    fe = open(os.path.join(a.out, "english.txt"), "w", encoding="utf-8")
    fr = open(os.path.join(a.out, "rejected.tsv"), "w", encoding="utf-8")
    kept = seen = 0
    why = collections.Counter()
    for path in a.files:
        with open(path, encoding="utf-8") as f:
            for para in f:
                para = para.strip()
                if len(para) < 40:
                    continue
                # one paragraph at a time: pronouns can refer back within it
                by_src = collections.OrderedDict()
                for line in u.encode_lines(para):
                    by_src.setdefault(line.source, []).append(line)
                for src, lines in by_src.items():
                    seen += 1
                    issues = [i for l in lines for i in l.issues if kind(i) not in a.allow]
                    if issues:
                        why.update({kind(i) for i in issues})
                        fr.write(src.replace("\t", " ") + "\t" + " | ".join(issues) + "\n")
                    else:
                        kept += 1
                        fu.write("\n".join(l.ugl for l in lines) + "\n")
                        fe.write(re.sub(r"\s+", " ", src) + "\n")
                    if a.max_sents and seen >= a.max_sents:
                        break
                if a.max_sents and seen >= a.max_sents:
                    break
        if a.max_sents and seen >= a.max_sents:
            break
    print(f"{seen} English sentences: {kept} kept ({kept / max(1, seen):.1%}), {seen - kept} rejected")
    for k, n in why.most_common():
        print(f"  {k:22} {n:6}  ({n / max(1, seen):.1%} of sentences)")


if __name__ == "__main__":
    main()

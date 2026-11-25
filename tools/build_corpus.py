"""Build a UGL training corpus from English text, keeping only sentences the
encoder has no doubts about.

    python tools/build_corpus.py data/raw/*.jsonl --out corpus/ --max-sents 50000

Writes:
    corpus/ugl.txt        one clean UGL line per line (training data)
    corpus/english.txt    the English sentence each clean line came from
    corpus/rejected.tsv   flagged sentences: English \t issues   (to inspect / fix)
    corpus/passages.jsonl runs of consecutive kept sentences from one paragraph,
                          {"source", "title", "english": [...], "ugl": [...]}:
                          connected text for language-model training (a rejected
                          sentence ends a run, so no passage jumps over a gap)
and prints how many sentences were kept and why the others were rejected.

Defaults (measured with tools/calibrate_parser.py on the EWT test set): the
ELECTRA-base parser reads the text, the shipped small parser reads it again,
and a sentence needs every word >= 0.6 head confidence (only where the
runner-up head would change the UGL) and both parsers to agree wherever it
changes the UGL.  Words not in the dictionary are allowed (kept as literals).
Why 0.6 and not 0.8, and why unknown words pass: hand grading of 181 sentences
from 7 sources (data/build/review-grades-2026-11-25.jsonl) found sentences
flagged only for a 60-80%-sure head (70% usable) or only for an unknown word
(71%) as good as the fully clean ones (60%); below 60% sure, 54%.

A sentence is rejected when any of its lines has an issue (see
ugl.encoder.Encoder.encode_lines): an uncertain parse, a pronoun with no
antecedent, a word not in the dictionary, a dropped or guessed piece.
Garbage in, garbage out: a smaller clean corpus beats a larger dirty one.
"""
import argparse
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ugl  # noqa: E402
from ugl.corpus import documents, paragraphs  # noqa: E402
from ugl.parser.model import DEFAULT_PATH, ParserModel  # noqa: E402

# the larger corpus parser (tools/train_parser_tf.py --model base); too big to ship
BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "build", "parser-en-base.npz")

KIND = [("uncertain grammar", "uncertain parse"), ("refers to nothing", "unresolved pronoun"),
        ("not in the dictionary", "unknown word"), ("no clear grammatical role", "unattached word"),
        ("was dropped", "dropped content word"), ("parsers disagree", "parsers disagree"), ("round trip", "round trip unstable"),
        ("not in the relation table", "unknown preposition"), ("does not parse back", "undecodable"),
        ("not a sentence", "fragment"),
        ("failed", "crash")]


def kind(issue):
    return next((k for pat, k in KIND if pat in issue), "other")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default="corpus")
    ap.add_argument("--max-sents", type=int, default=None)
    ap.add_argument("--min-confidence", type=float, default=0.6,
                    help="calibrated head probability every word needs (see tools/calibrate_parser.py)")
    ap.add_argument("--parser", default=BASE if os.path.exists(BASE) else None,
                    help="parser weights (default: the larger corpus parser when built, else the shipped one)")
    ap.add_argument("--second-parser", default=DEFAULT_PATH,
                    help="another parser's weights: sentences where the two give different UGL are rejected")
    ap.add_argument("--single-parser", action="store_true", help="skip the two-parser agreement check")
    ap.add_argument("--no-roundtrip", action="store_true", help="skip the encode-decode-encode check")
    ap.add_argument("--allow", nargs="*", default=["unknown word"],
                    help="issue kinds to tolerate (default: 'unknown word'; pass --allow alone for none)")
    a = ap.parse_args()
    u = ugl.UGL()
    u.encoder.unsure = a.min_confidence
    if a.parser:
        u.encoder.parser = ParserModel(a.parser)
    if a.second_parser and not a.single_parser and os.path.abspath(a.second_parser) != os.path.abspath(a.parser or DEFAULT_PATH):
        u.encoder.second_parser = ParserModel(a.second_parser)
    os.makedirs(a.out, exist_ok=True)
    fu = open(os.path.join(a.out, "ugl.txt"), "w", encoding="utf-8")
    fe = open(os.path.join(a.out, "english.txt"), "w", encoding="utf-8")
    fr = open(os.path.join(a.out, "rejected.tsv"), "w", encoding="utf-8")
    fp = open(os.path.join(a.out, "passages.jsonl"), "w", encoding="utf-8")
    passages = kept = seen = 0
    why = collections.Counter()

    def paragraph_units():
        """(source, title, paragraph) for every paragraph worth encoding; the
        source is the folder the file sits in (scripts/fetch_sources.py)."""
        for path in a.files:
            source = os.path.basename(os.path.dirname(os.path.abspath(path)))
            for title, text in documents([path]):
                for para in paragraphs(text):
                    if len(para) >= 40:
                        yield source, title, para

    for source, title, para in paragraph_units():
        # one paragraph at a time: pronouns refer back within it, or to the title
        by_src = collections.OrderedDict()
        for line in u.encode_lines(para, topic=title, roundtrip=not a.no_roundtrip):
            by_src.setdefault(line.source, []).append(line)
        runs, run = [], []
        for src, lines in by_src.items():
            seen += 1
            issues = [i for l in lines for i in l.issues if kind(i) not in a.allow]
            if issues:
                why.update({kind(i) for i in issues})
                fr.write(src.replace("\t", " ") + "\t" + " | ".join(issues) + "\n")
                runs.append(run)
                run = []
            else:
                kept += 1
                eng, code = re.sub(r"\s+", " ", src), "\n".join(l.ugl for l in lines)
                fu.write(code + "\n")
                fe.write(eng + "\n")
                run.append((eng, code))
            if a.max_sents and seen >= a.max_sents:
                break
        for r in runs + [run]:
            if r:
                fp.write(json.dumps({"source": source, "title": title, "english": [e for e, _ in r],
                                     "ugl": [c for _, c in r]}, ensure_ascii=False) + "\n")
                passages += 1
        if a.max_sents and seen >= a.max_sents:
            break
    print(f"{seen} English sentences: {kept} kept ({kept / max(1, seen):.1%}), {seen - kept} rejected; "
          f"{passages} passages ({kept / max(1, passages):.1f} sentences each)")
    for k, n in why.most_common():
        print(f"  {k:22} {n:6}  ({n / max(1, seen):.1%} of sentences)")


if __name__ == "__main__":
    main()

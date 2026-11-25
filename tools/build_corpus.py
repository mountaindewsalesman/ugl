"""Build a UGL training corpus from English text, keeping only sentences the
encoder has no doubts about.

    python tools/build_corpus.py data/raw/*.jsonl --out corpus/ --max-sents 50000

Writes:
    corpus/ugl.txt        one clean UGL line per line (training data)
    corpus/english.txt    the English sentence each clean line came from
    corpus/rejected.tsv   flagged sentences: English \t issues   (to inspect / fix)
    corpus/passages.jsonl runs of consecutive kept sentences from one document,
                          {"source", "title", "english": [...], "ugl": [...],
                           "paragraph": [paragraph number of each sentence]}:
                          connected text for language-model training (a rejected
                          sentence ends a run, so no passage jumps over a gap)
and prints how many sentences were kept and why the others were rejected.

Defaults (measured with tools/calibrate_parser.py on the EWT test set): the
ELECTRA-base parser reads the text, the shipped small parser reads it again,
and a sentence needs every word >= 0.6 head confidence (only where the
runner-up head would change the UGL) and both parsers to agree wherever it
changes the UGL.  Words not in the dictionary are allowed (kept as literals).
Why 0.6 and not 0.8, and why unknown words pass: hand grading of 181 sentences
from 7 sources (data/review/grades-2026-11-25.jsonl) found sentences
flagged only for a 60-80%-sure head (70% usable) or only for an unknown word
(71%) as good as the fully clean ones (60%); below 60% sure, 54%.

Two ways to decide (see ugl.encoder.Encoder.encode_lines for the issues):

  rules  (default)  a sentence is rejected when any of
         its lines has an issue: an uncertain parse, parsers disagreeing, an
         unstable round trip, a pronoun with no antecedent, a dropped or guessed piece.
  grader (--grader [folder]; trained by tools/train_quality.py)
         sentences with a broken piece (HARD: dropped word, fragment, crash, ...)
         are rejected; every other sentence is scored by the grader (English vs the
         decoder's English for the UGL, plus the flags) and kept when the score is
         above the cut that kept --keep of the held-out test sentences.  On EWT test:
         rules keep 26% at 84% right; the grader keeps 40% at 87% right.
         The score is written with each kept sentence (scores.txt, passages.jsonl).
         NOT the default: it does not transfer beyond EWT-like text.  On other
         sources it kept half as many sentences as the rules (FineWeb-Edu 6.7% vs
         13.3%, Simple Wikipedia 16% vs 32%: scores shift down on unfamiliar text,
         so the EWT-calibrated cut is too strict), and among rule-rejected review
         sentences it was sure of (score >= 0.9), 11 of 20 were wrong by hand.
         Its labels come from EWT-style treebanks; it needs labels from the
         corpus's own kinds of text before it can decide for them.

Garbage in, garbage out: a smaller clean corpus beats a larger dirty one.
"""
import argparse
import bisect
import collections
import itertools
import json
import os
import re
import sys
import time

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


# issues that mean a piece of the sentence is missing or broken: rejected even
# when the grader is used (hand grading: ~21% / ~15% usable)
HARD = {"dropped content word", "fragment", "crash", "undecodable", "unattached word", "unknown preposition", "other"}


def document_units(files):
    """(source, title, paragraphs) per document; the source is the folder the
    file sits in (scripts/fetch_sources.py).  A paragraph is kept when it ends
    like a sentence or is long (short unpunctuated lines are headings):
    '"Yes!" said Tom.' is a sentence, "Early life" is not."""
    for path in files:
        source = os.path.basename(os.path.dirname(os.path.abspath(path)))
        for title, text in documents([path]):
            paras = [p for p in paragraphs(text) if len(p) >= 40 or re.search(r"[.!?][\"'”’)]*$", p)]
            if paras:
                yield source, title, paras


def _lines(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read().splitlines()


def resume_point(out, units):
    """(documents done, sentences read, kept, passages) of a stopped build.
    From progress.json when the build wrote one; otherwise from the files: a
    document's sentences are written only after the whole document is encoded,
    so the document holding the last written sentence is the last one done.
    Short sentences repeat across documents ("They were happy."), so of the
    documents holding it, the one nearest the position implied by the number
    of sentences read is taken."""
    english = _lines(os.path.join(out, "english.txt"))
    rejected = _lines(os.path.join(out, "rejected.tsv"))
    passages = len(_lines(os.path.join(out, "passages.jsonl")))
    seen, kept = len(english) + len(rejected), len(english)
    prog = os.path.join(out, "progress.json")
    if os.path.exists(prog):
        with open(prog, encoding="utf-8") as f:
            p = json.load(f)
        return p["docs"], p["seen"], p["kept"], p["passages"]
    if not seen:
        return 0, 0, 0, 0
    files = [(os.path.getmtime(os.path.join(out, n)), rows) for n, rows in
             (("english.txt", english), ("rejected.tsv", rejected)) if rows]
    last = max(files, key=lambda x: x[0])[1][-1].split("\t")[0]
    norm = lambda x: re.sub(r"\s+", " ", x).strip()
    last = norm(last)
    from ugl.parser.tokenize import split_sentences
    best, cum = None, 0
    for i, (_, _, paras) in enumerate(units):
        text = "\n".join(paras)
        cum += len(split_sentences(text))
        if last in norm(text) and (best is None or abs(cum - seen) < best[0]):
            best = (abs(cum - seen), i)
        if cum > seen * 1.1 + 2000:
            break
    if best is None:
        raise SystemExit(f"cannot find where {out} stopped: no document holds {last!r}")
    return best[1] + 1, seen, kept, passages


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
    ap.add_argument("--grader", nargs="?", const="", default=None,
                    help="decide judgment calls with the quality grader (folder; default data/build/quality-grader)")
    ap.add_argument("--resume", action="store_true",
                    help="continue a stopped build in --out: append, starting after the last finished document")
    ap.add_argument("--device", default=None,
                    help="run the transformer passes on this torch device (e.g. cuda); default numpy on the CPU")
    ap.add_argument("--keep", type=float, default=0.4,
                    help="grader cut: the share of held-out test sentences it kept (0.4: 87%% right there)")
    a = ap.parse_args()
    if a.device:
        from ugl.transformer import use_device
        use_device(a.device)
    u = ugl.UGL()
    grader = None
    if a.grader is not None:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from quality import DEFAULT_PATH as GRADER, QualityGrader
        path = a.grader or GRADER
        if os.path.exists(os.path.join(path, "grader.pt")):
            grader = QualityGrader(path)
            cut = grader.threshold(a.keep)
            print(f"grader {path}: keeping scores >= {cut:.3f}", file=sys.stderr)
    # the grader was trained on flags raised at 0.8 head confidence; it, not the
    # threshold, decides those sentences
    u.encoder.unsure = 0.8 if grader else a.min_confidence
    if a.parser:
        u.encoder.parser = ParserModel(a.parser)
    if a.second_parser and not a.single_parser and os.path.abspath(a.second_parser) != os.path.abspath(a.parser or DEFAULT_PATH):
        u.encoder.second_parser = ParserModel(a.second_parser)
    os.makedirs(a.out, exist_ok=True)
    passages = kept = seen = done_docs = 0
    if a.resume:
        done_docs, seen, kept, passages = resume_point(a.out, document_units(a.files))
        print(f"resuming after document {done_docs} ({seen} sentences read, {kept} kept)", file=sys.stderr)
    with open(os.path.join(a.out, "run.json"), "w", encoding="utf-8") as f:
        # what this build reads; tools/corpus_monitor.py estimates time left from it
        # (the rate counts from the last start or resume)
        json.dump({"files": [os.path.abspath(p) for p in a.files], "max_sents": a.max_sents,
                   "started": time.time(), "seen_at_start": seen, "device": a.device}, f, indent=1)
    mode = "a" if a.resume else "w"
    fu = open(os.path.join(a.out, "ugl.txt"), mode, encoding="utf-8", buffering=1)
    fe = open(os.path.join(a.out, "english.txt"), mode, encoding="utf-8", buffering=1)
    fr = open(os.path.join(a.out, "rejected.tsv"), mode, encoding="utf-8", buffering=1)
    fp = open(os.path.join(a.out, "passages.jsonl"), mode, encoding="utf-8", buffering=1)
    fs = open(os.path.join(a.out, "scores.txt"), mode, encoding="utf-8", buffering=1) if grader else None
    why = collections.Counter()

    for doc_no, (source, title, paras) in enumerate(document_units(a.files)):
        if doc_no < done_docs:
            continue
        # the whole document at once, so pronouns refer back across paragraphs
        # ("Lily went to the park.\nShe saw a dog."); every paragraph break ends a sentence
        text = "\n".join(paras)
        starts = [0]
        for p in paras[:-1]:
            starts.append(starts[-1] + len(p) + 1)
        # lines of one sentence are consecutive; group by position, not by text, so
        # two identical sentences ("Yes!") stay two
        items, cursor = [], 0
        for src, group in itertools.groupby(u.encode_lines(text, topic=title, roundtrip=not a.no_roundtrip),
                                            key=lambda l: l.source):
            at = text.find(src, cursor)
            if at >= 0:
                cursor = at + len(src)
            items.append((src, list(group), bisect.bisect_right(starts, max(at, cursor - 1)) - 1))
        if a.max_sents:
            items = items[: max(0, a.max_sents - seen)]
        scores = [None] * len(items)
        if grader:
            # one batch per document: every sentence without a broken piece
            rows, at = [], []
            for n, (src, lines, _) in enumerate(items):
                kinds = sorted({kind(i) for l in lines for i in l.issues})
                if set(kinds) & HARD:
                    continue
                try:
                    back = u.decode("\n".join(l.ugl for l in lines))
                except Exception:  # noqa: BLE001
                    continue
                rows.append({"src": src, "eng": back, "kinds": kinds,
                             "roundtrip": "round trip unstable" not in kinds})
                at.append(n)
            for n, sc in zip(at, grader.score(rows) if rows else []):
                scores[n] = sc
        runs, run = [], []
        for (src, lines, para), sc in zip(items, scores):
            seen += 1
            issues = [i for l in lines for i in l.issues if kind(i) not in a.allow]
            if grader:
                ok = sc is not None and sc >= cut
                if not ok and sc is not None:
                    issues = issues + [f"grader score {sc:.2f} < {cut:.2f}"]
            else:
                ok = not issues
            if not ok:
                why.update({"grader: low score"} if sc is not None else {kind(i) for i in issues})
                fr.write(src.replace("\t", " ") + "\t" + " | ".join(issues) + "\n")
                runs.append(run)
                run = []
            else:
                kept += 1
                eng, code = re.sub(r"\s+", " ", src), "\n".join(l.ugl for l in lines)
                fu.write(code + "\n")
                fe.write(eng + "\n")
                if fs:
                    fs.write(f"{sc:.4f}\n")
                run.append((eng, code, sc, para))
        for r in runs + [run]:
            if r:
                p = {"source": source, "title": title, "english": [x[0] for x in r], "ugl": [x[1] for x in r],
                     "paragraph": [x[3] for x in r]}     # paragraph number of each sentence in its document
                if grader:
                    p["scores"] = [round(x[2], 4) for x in r]
                fp.write(json.dumps(p, ensure_ascii=False) + "\n")
                passages += 1
        # checkpoint: every document up to this one is fully written
        with open(os.path.join(a.out, "progress.json"), "w", encoding="utf-8") as f:
            json.dump({"docs": doc_no + 1, "seen": seen, "kept": kept, "passages": passages}, f)
        if a.max_sents and seen >= a.max_sents:
            break
    print(f"{seen} English sentences: {kept} kept ({kept / max(1, seen):.1%}), {seen - kept} rejected; "
          f"{passages} passages ({kept / max(1, passages):.1f} sentences each)")
    for k, n in why.most_common():
        print(f"  {k:22} {n:6}  ({n / max(1, seen):.1%} of sentences)")


if __name__ == "__main__":
    main()

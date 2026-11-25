"""Labelled examples for the sentence-quality grader (tools/train_quality.py).

    python tools/make_quality_labels.py --out data/build/quality-labels.jsonl

For treebank sentences the parser never trained on, the sentence is encoded
the way tools/build_corpus.py does it (base parser + agreement + round trip)
and also from its hand-annotated tree.  label = 1 when both give the same UGL.
Each row also keeps what a grader may look at: the English, the UGL, the
decoder's English for it, the issue kinds, the round-trip result and the
parser's head confidences.

What these labels can and cannot teach: they mark sentences the parser read
wrongly.  An encoder mistake made from the correct tree too ("without" ->
"with") gives the same UGL on both sides and is labelled good; those are
found by hand review (tools/review_sample.py) and fixed in the encoder.

Splits: "test" = EWT test (never used for training the grader); "train" =
dev splits of the parser's treebanks + treebanks it never saw (PUD, Little
Prince, CTeTex, CHILDES).
"""
import argparse
import json
import multiprocessing
import os
import random
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

UD = os.path.join(ROOT, "data", "ud")
SOURCES = {   # file -> (split, max sentences)
    "en_ewt-ud-test.conllu": ("test", None),
    "en_ewt-ud-dev.conllu": ("train", None),
    "en_gum-ud-dev.conllu": ("train", None),
    "en_lines-ud-dev.conllu": ("train", None),
    "en_partut-ud-dev.conllu": ("train", None),
    "en_atis-ud-dev.conllu": ("train", 300),
    "en_pud-ud-test.conllu": ("train", None),
    "en_littleprince-ud-test.conllu": ("train", None),
    "en_ctetex-ud-test.conllu": ("train", None),
    "en_childes-ud-test.conllu": ("train", 2000),
}

_u = None


def _init():
    global _u
    import ugl
    from ugl.parser.model import ParserModel
    from build_corpus import BASE, DEFAULT_PATH
    _u = ugl.UGL()
    _u.encoder.unsure = 0.8
    _u.encoder.parser = ParserModel(BASE)
    _u.encoder.second_parser = ParserModel(DEFAULT_PATH)


def label(item):
    from build_corpus import kind
    from compare_encoders import encode_tree
    treebank, split, sent = item
    u = _u
    words = [t["form"] for t in sent]
    gold, err = encode_tree(u.lexicon, [{k: v for k, v in t.items() if k != "lemma"} for t in sent],
                            words, words[-1] == "?")
    if err:
        return None
    seen = []
    parse = u.encoder.parse
    u.encoder.parse = lambda text: seen.append(parse(text)) or seen[-1]
    try:
        lines = u.encoder.encode_lines(" ".join(words))
    except Exception:  # noqa: BLE001
        return None
    finally:
        u.encoder.parse = parse
    if not lines or len({l.source for l in lines}) != 1:
        return None                                   # split into two sentences: not comparable
    tree = seen[0][0][1]
    confs = [t.get("conf", 1.0) for t in tree if t["upos"] != "PUNCT"]
    ugl_lines = [l.ugl for l in lines]
    try:
        eng = u.decode("\n".join(ugl_lines))
        rt = u.same_reading(ugl_lines, [l.ugl for l in u.encoder.encode_lines(eng)])
    except Exception:  # noqa: BLE001
        eng, rt = "", False
    return {"treebank": treebank, "split": split, "src": " ".join(words), "ugl": ugl_lines, "eng": eng,
            "gold": gold, "label": int("\n".join(ugl_lines) == gold),
            "kinds": sorted({kind(i) for l in lines for i in l.issues}), "roundtrip": rt,
            "min_conf": min(confs) if confs else 1.0,
            "mean_conf": sum(confs) / len(confs) if confs else 1.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "build", "quality-labels.jsonl"))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--limit", type=int, default=None, help="sentences per treebank (quick runs)")
    a = ap.parse_args()
    from ugl.parser.conllu import read_conllu
    from ugl.parser.tokenize import tokenize
    items = []
    for f, (split, cap) in SOURCES.items():
        path = os.path.join(UD, f)
        if not os.path.exists(path):
            print("missing", f, file=sys.stderr)
            continue
        sents = [s for s in read_conllu(path)
                 if 4 <= len(s) <= 40 and tokenize(" ".join(t["form"] for t in s)) == [t["form"] for t in s]]
        random.Random(0).shuffle(sents)
        cap = min(x for x in (cap, a.limit, len(sents)) if x is not None)
        items += [(f.split("-ud-")[0], split, s) for s in sents[:cap]]
    print(f"{len(items)} sentences, {a.workers} workers", flush=True)
    n = good = 0
    with open(a.out, "w", encoding="utf-8") as out, multiprocessing.Pool(a.workers, _init) as pool:
        for r in pool.imap_unordered(label, items, chunksize=8):
            if r is None:
                continue
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
            good += r["label"]
            if n % 500 == 0:
                print(f"  {n} labelled, {good / n:.1%} right", flush=True)
    print(f"done: {n} labelled, {good / max(1, n):.1%} right -> {a.out}")


if __name__ == "__main__":
    main()

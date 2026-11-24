"""Evaluate sense choice on SemCor (hand-annotated WordNet senses).

    python tools/eval_wsd.py --sents 400

Compares 'most frequent sense' (MFS) with ugl.wsd.VectorWSD at several
weights.  Note WordNet's sense order was itself derived from SemCor, so MFS is
flattered here; any gain over it is a conservative estimate.

Scores accuracy on the UGL *token* level as well: two senses that the
dictionary merges into the same tokens count as correct, which is what matters
for UGL.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import nltk
import spacy
from nltk.corpus import semcor
from nltk.tree import Tree
from spacy.tokens import Doc

from ugl.lexicon import Lexicon
from ugl.wsd import VectorWSD

CLS = {"n": "n", "v": "v", "a": "a", "s": "a", "r": "r"}


def items(n_sents, offset):
    sents = semcor.tagged_sents(tag="sem")
    for sent in sents[offset: offset + n_sents]:
        words, targets = [], []
        for chunk in sent:
            if isinstance(chunk, Tree) and hasattr(chunk.label(), "synset"):
                lab = chunk.label()
                toks = chunk.leaves()
                try:
                    ss = lab.synset()
                except Exception:
                    ss = None
                if ss is not None:
                    targets.append((len(words), lab.name().lower(), ss.name(), CLS[ss.pos()]))
                words += toks
            else:
                words += chunk.leaves() if isinstance(chunk, Tree) else list(chunk)
        yield words, targets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sents", type=int, default=400)
    ap.add_argument("--offset", type=int, default=20000)
    ap.add_argument("--context", default=None, help="path to bert-semcor.npz: evaluate the contextual model")
    ap.add_argument("--gloss", default=None, help="path to a parser .npz with a sense head (ugl.wsd_gloss)")
    a = ap.parse_args()
    nltk.download("semcor", quiet=True)
    nlp = spacy.load("en_core_web_md")
    lex = Lexicon()
    if a.gloss:
        from ugl.parser.model import ParserModel
        from ugl.wsd_gloss import GlossWSD
        parser = ParserModel(a.gloss)
        weights = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 99.0]   # prior weight in logit units; 99 = most frequent
        models = {w: GlossWSD(lex, parser, prior_weight=w) for w in weights}
        for m in models.values():
            m._gloss = models[0.0]._gloss              # share the gloss-vector cache
    elif a.context:
        from ugl.wsd_context import ContextWSD
        base = ContextWSD(lex, a.context)
        weights = [0.0, 0.02, 0.05, 0.1, 0.2, 99.0]   # 99 = pure most-frequent-sense
        models = {w: ContextWSD(lex, a.context, prior_weight=w, embedder=base.emb) for w in weights}
        for m in models.values():
            m._gloss = base._gloss
    else:
        weights = [0, 1, 2, 4, 6, 8, 12]
        models = {w: VectorWSD(lex, nlp, weight=w) for w in weights}
    hits = {w: [0, 0] for w in weights}
    n = 0
    for words, targets in items(a.sents, a.offset):
        doc = nlp(Doc(nlp.vocab, words=[w.replace(" ", "_") for w in words]))
        for i, lemma, gold, cls in targets:
            cands = lex.senses(lemma, cls)
            if len(cands) < 2 or gold not in cands:
                continue
            n += 1
            gold_t = lex.senses_[gold]["tokens"]
            for w, m in models.items():
                pick = m.choose(lemma, cands, (doc, i))
                hits[w][0] += pick == gold
                hits[w][1] += lex.senses_[pick]["tokens"] == gold_t
    print(f"{n} ambiguous words from {a.sents} SemCor sentences")
    print("weight  sense-acc  token-acc   (" + ("prior weight; 99 = most frequent sense" if (a.context or a.gloss) else "weight 0 = most frequent sense") + ")")
    for w in weights:
        print(f"{w:6}  {hits[w][0]/n:9.3f}  {hits[w][1]/n:9.3f}")


if __name__ == "__main__":
    main()

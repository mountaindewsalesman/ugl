"""Sense choice with the parser's own transformer (numpy; runs in the browser).

Trained jointly with the parser (tools/train_parser_tf.py): the transformer
reads the sentence, and separately reads each candidate sense's dictionary
gloss ("bank : sloping land beside a body of water").  Two small projections
map the word-in-context and the gloss into one space; the sense whose gloss
lands nearest wins, nudged by the frequency prior:

    score(sense) = ctx . gloss(sense) - prior_weight * log(rank + 1)

Every sense has a gloss, so senses never seen in training still compete.

Gloss vectors depend only on the (frozen) dictionary and parser, so they are
computed once by tools/build_gloss_table.py and shipped next to the parser
weights (8-bit, one scale per row): choosing a sense is then one dot product
per candidate.  Without a matching table they are computed on first use (one
transformer pass per gloss: ~1 s per sentence in the browser instead of ~10).
"""
import json
import math
import os

import numpy as np

MAX_GLOSS_PIECES = 64


def table_path(parser_path, dict_version):
    return os.path.splitext(parser_path)[0] + f".glosses-{dict_version}.npz"


def table_meta(parser_path, dict_version):
    """What a table must match: the dictionary version and the exact parser file."""
    return {"dictionary": dict_version, "parser": os.path.basename(parser_path),
            "parser_bytes": os.path.getsize(parser_path)}


def load_table(parser_path, dict_version):
    """sense -> row, int8 rows, row scales; None when no matching table exists."""
    path = table_path(parser_path, dict_version)
    if not os.path.exists(path):
        return None
    z = np.load(path, allow_pickle=False)
    if json.loads(str(z["meta"])) != table_meta(parser_path, dict_version):
        return None                                   # stale: built for other weights
    names = str(z["senses"]).split("\n")
    return {s: i for i, s in enumerate(names)}, z["q"], z["scale"]


def gloss_words(lexicon, sense):
    """The text the gloss encoder reads; training and inference both use this."""
    lemma = sense.rsplit(".", 2)[0].replace("_", " ")
    return lemma.split() + [":"] + lexicon.gloss(sense).split()


class GlossWSD:
    def __init__(self, lexicon, parser, prior_weight=None):
        if "wsd.mix" not in parser.w:
            raise ValueError("this parser has no sense head")
        self.lex, self.p = lexicon, parser
        self.mix = parser.w["wsd.mix"]
        self.prior_weight = prior_weight if prior_weight is not None else parser.meta.get("wsd_prior", 0.05)
        self._gloss = {}
        self.table = None
        if getattr(parser, "path", None) and os.path.exists(parser.path) and getattr(lexicon, "version", None):
            self.table = load_table(parser.path, lexicon.version)

    def _lin(self, name, x):
        return x @ self.p.w[name + ".weight"].T + self.p.w[name + ".bias"]

    def context(self, words, i):
        st = self.p.layer_states(list(words))                    # (L+1, n, d)
        return self._lin("wsd_ctx", np.tensordot(self.mix, st[:, i], axes=1))

    def gloss_vec(self, sense):
        v = self._gloss.get(sense)
        if v is None:
            row = self.table[0].get(sense) if self.table else None
            v = (self.table[1][row].astype(np.float32) * self.table[2][row] if row is not None
                 else self.compute_gloss_vec(sense))
            self._gloss[sense] = v
        return v

    def compute_gloss_vec(self, sense):
        ids, _ = self.p.wp.encode_words(gloss_words(self.lex, sense))
        ids = ids[: MAX_GLOSS_PIECES - 1] + [ids[-1]]
        hs = self.p.tf.hidden_states(ids)
        return self._lin("wsd_gloss", sum(m * h[0] for m, h in zip(self.mix, hs)))

    def choose(self, lemma, cands, sent_and_index):
        sent, i = sent_and_index
        if isinstance(sent, (list, tuple)):
            words = list(sent)
        else:                                                   # spaCy span / doc
            words = [t.text for t in sent]
            i = i - getattr(sent, "start", 0)
        c = self.context(words, i)
        best, best_s = None, -1e18
        for rank, s in enumerate(cands[:12]):                   # the head was trained on the first 12
            sc = float(c @ self.gloss_vec(s)) - self.prior_weight * math.log(rank + 1)
            if sc > best_s:
                best, best_s = s, sc
        return best

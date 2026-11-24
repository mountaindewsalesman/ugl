"""Word-sense disambiguation (choosing which meaning of a word is meant).

VectorWSD is fully local and needs no training: it compares the sentence
context with each candidate sense's dictionary gloss using spaCy's word
vectors, and combines that with the most-frequent-sense prior.

    score(sense) = log prior(rank) + weight * cos(context, gloss)

`python tools/eval_wsd.py` measures it against SemCor (hand-tagged senses).
"""
import math

import numpy as np

DEFAULT_WEIGHT = 4.0
PRIOR_EXP = 1.0
WINDOW = 8


class VectorWSD:
    def __init__(self, lexicon, nlp, weight=DEFAULT_WEIGHT, prior_exp=PRIOR_EXP, window=WINDOW):
        self.lex, self.nlp = lexicon, nlp
        self.weight, self.prior_exp, self.window = weight, prior_exp, window
        self._gloss = {}
        self.stops = nlp.Defaults.stop_words

    def _vec(self, words):
        vs = []
        for w in words:
            lex = self.nlp.vocab[w]
            if lex.has_vector and w.lower() not in self.stops and w.isalpha():
                vs.append(lex.vector / (lex.vector_norm or 1.0))
        if not vs:
            return None
        v = np.mean(vs, axis=0)
        n = np.linalg.norm(v)
        return v / n if n else None

    def gloss_vec(self, sense):
        if sense not in self._gloss:
            e = self.lex.senses_[sense]
            words = e["gloss"].replace(";", " ").replace(",", " ").split()
            # the sense's own primitive lemma (if any) is a strong cue too
            tok = e["tokens"][0]
            ent = self.lex.entry(tok)
            if ent:
                words += ent["lemma"].split()
            self._gloss[sense] = self._vec(words)
        return self._gloss[sense]

    def context_vec(self, sent, i):
        words = []
        for t in sent:
            if t.i != i and abs(t.i - i) <= self.window and t.is_alpha and not t.is_stop:
                words.append(t.text)
        return self._vec(words)

    def scores(self, lemma, cands, sent_and_index):
        sent, i = sent_and_index
        ctx = self.context_vec(sent, i)
        out = []
        for rank, s in enumerate(cands):
            sc = -self.prior_exp * math.log(rank + 1)
            g = self.gloss_vec(s)
            if ctx is not None and g is not None:
                sc += self.weight * float(ctx @ g)
            out.append(sc)
        return out

    def choose(self, lemma, cands, sent_and_index):
        sc = self.scores(lemma, cands, sent_and_index)
        return cands[int(np.argmax(sc))]


def attach(lexicon, nlp, **kw):
    lexicon.wsd = VectorWSD(lexicon, nlp, **kw)
    return lexicon.wsd

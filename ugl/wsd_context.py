"""Contextual word-sense disambiguation (local, no cloud).

A BERT model turns each word in its sentence into a context vector.  Every
sense has a centroid vector: the average of its hand-tagged SemCor
occurrences (built by tools/train_wsd.py), or, for senses SemCor never tags,
the vector of the lemma inside its own dictionary gloss (computed lazily).
The chosen sense is the nearest centroid, nudged by the frequency prior:

    score(sense) = cos(context, centroid) - prior_weight * log(rank + 1)
"""
import math
import os

import numpy as np

DEFAULT_MODEL = "bert-base-uncased"
DEFAULT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "wsd", "bert-semcor.npz")


class ContextEmbedder:
    def __init__(self, model_name=DEFAULT_MODEL, layers=4):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True).eval()
        self.layers = layers
        torch.set_grad_enabled(False)

    def embed_batch(self, sents):
        """sents: list of word lists -> list of (n_words x dim) float32 arrays."""
        enc = self.tok(sents, is_split_into_words=True, return_tensors="pt", padding=True,
                       truncation=True, max_length=256)
        out = self.model(**enc)
        h = sum(out.hidden_states[-self.layers:]).numpy()       # (batch, seq, dim)
        res = []
        for b, words in enumerate(sents):
            ids = enc.word_ids(b)
            dim = h.shape[-1]
            acc = np.zeros((len(words), dim), np.float32)
            cnt = np.zeros(len(words), np.float32)
            for j, w in enumerate(ids):
                if w is not None:
                    acc[w] += h[b, j]
                    cnt[w] += 1
            cnt[cnt == 0] = 1
            v = acc / cnt[:, None]
            v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-8
            res.append(v)
        return res


class ContextWSD:
    def __init__(self, lexicon, path=DEFAULT_PATH, prior_weight=0.05, embedder=None):
        data = np.load(path, allow_pickle=False)
        self.lex = lexicon
        self.names = [str(x) for x in data["senses"]]
        self.row = {s: i for i, s in enumerate(self.names)}
        self.mat = data["vectors"].astype(np.float32)
        self.mat /= np.linalg.norm(self.mat, axis=1, keepdims=True) + 1e-8
        self.model_name = str(data["model"])
        self.emb = embedder or ContextEmbedder(self.model_name)
        self.prior_weight = prior_weight
        self._cache_key, self._cache_val = None, None
        self._gloss = {}

    def word_vectors(self, words):
        key = tuple(words)
        e = self.emb          # cache lives on the embedder so instances can share it
        if getattr(e, "_cache_key", None) != key:
            e._cache_key, e._cache_val = key, e.embed_batch([list(words)])[0]
        return e._cache_val

    def sense_vec(self, sense):
        if sense in self.row:
            return self.mat[self.row[sense]]
        if sense not in self._gloss:
            lemma = sense.rsplit(".", 2)[0].replace("_", " ")
            words = lemma.split() + [":"] + self.lex.gloss(sense).split()
            v = self.emb.embed_batch([words])[0]
            self._gloss[sense] = v[: len(lemma.split())].mean(axis=0)
        return self._gloss[sense]

    def choose(self, lemma, cands, sent_and_index):
        sent, i = sent_and_index
        words = [t.text for t in sent]
        i = i - getattr(sent, "start", 0)
        ctx = self.word_vectors(words)[i]
        best, best_s = None, -1e9
        for rank, s in enumerate(cands):
            sc = float(ctx @ self.sense_vec(s)) - self.prior_weight * math.log(rank + 1)
            if sc > best_s:
                best, best_s = s, sc
        return best


def attach(lexicon, path=DEFAULT_PATH, **kw):
    lexicon.wsd = ContextWSD(lexicon, path, **kw)
    return lexicon.wsd

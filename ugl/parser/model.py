"""Neural tagger + dependency parser, inference in pure numpy.

Architecture (trained by tools/train_parser.py, PyTorch):
    token features  = word emb (trained) + pretrained vector (frozen, PCA 128)
                      + prefix/suffix embs + shape emb
    encoder         = 2-layer BiLSTM
    taggers         = linear heads for UPOS, XPOS (Penn tag), FEATS
    arcs            = biaffine scorer over MLP(dep) x MLP(head), decoded to a
                      tree with Chu-Liu/Edmonds (one root)
    labels          = biaffine label scorer on the chosen arcs
Weights live in one .npz; this file is all that is needed to run it.
"""
import json
import os

import numpy as np

from .features import AFFIX_KEYS, affixes, norm, shape
from .mst import chu_liu_edmonds

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "parser-en.npz")


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def _relu(x):
    return np.maximum(x, 0)


class ParserModel:
    def __init__(self, path=DEFAULT_PATH, data=None):
        self.path = path
        z = np.load(path if data is None else data, allow_pickle=False)
        self.w = {k: z[k].astype(np.float32) for k in z.files if k != "meta"}
        meta = json.loads(str(z["meta"]))
        self.meta = meta
        self.vocab = {k: {s: i for i, s in enumerate(v)} for k, v in meta["vocabs"].items()}
        self.labels = {k: meta["vocabs"][k] for k in ("upos", "xpos", "feats", "deprel")}
        self.layers = meta["lstm_layers"]

    # -------------------------------------------------------------- features
    def _ids(self, vocab, keys):
        v = self.vocab[vocab]
        return np.array([v.get(k, 1) for k in keys])      # 1 = <unk>, 0 = <pad>

    def embed(self, words):
        w = self.w
        n = [norm(x) for x in words]
        parts = [w["emb.word"][self._ids("word", n)],
                 w["emb.pre"][self._ids("pre", n)]]
        aff = [affixes(x) for x in words]
        for k in AFFIX_KEYS:
            parts.append(w[f"emb.{k}"][self._ids(k, [a[k] for a in aff])])
        parts.append(w["emb.shape"][self._ids("shape", [shape(x) for x in words])])
        return np.concatenate(parts, axis=1)

    def lstm_dir(self, x, layer, rev):
        sfx = f"l{layer}" + ("_reverse" if rev else "")
        Wih, Whh = self.w[f"lstm.weight_ih_{sfx}"], self.w[f"lstm.weight_hh_{sfx}"]
        b = self.w[f"lstm.bias_ih_{sfx}"] + self.w[f"lstm.bias_hh_{sfx}"]
        H = Whh.shape[1]
        xs = x @ Wih.T + b
        h = np.zeros(H, np.float32)
        c = np.zeros(H, np.float32)
        out = np.zeros((len(x), H), np.float32)
        order = range(len(x) - 1, -1, -1) if rev else range(len(x))
        for t in order:
            g = xs[t] + Whh @ h
            i, f, gg, o = g[:H], g[H:2 * H], g[2 * H:3 * H], g[3 * H:]
            c = _sig(f) * c + _sig(i) * np.tanh(gg)
            h = _sig(o) * np.tanh(c)
            out[t] = h
        return out

    def encode(self, words):
        x = self.embed(words)
        for l in range(self.layers):
            x = np.concatenate([self.lstm_dir(x, l, False), self.lstm_dir(x, l, True)], axis=1)
        return x

    def mlp(self, name, x):
        return _relu(x @ self.w[f"{name}.weight"].T + self.w[f"{name}.bias"])

    def lin(self, name, x):
        return x @ self.w[f"{name}.weight"].T + self.w[f"{name}.bias"]

    # -------------------------------------------------------------- predict
    def parse(self, words):
        """words: list of token strings -> list of dicts
        {form, upos, xpos, feats, head (1-based, 0=root), deprel}."""
        if not words:
            return []
        Hs = self.encode(words)                              # (n, d)
        H = np.concatenate([self.w["root"][None, :], Hs])    # (n+1, d), index 0 = ROOT
        upos = self.lin("upos", Hs).argmax(1)
        xpos = self.lin("xpos", Hs).argmax(1)
        feats = self.lin("feats", Hs).argmax(1)

        dep, head = self.mlp("arc_dep", H), self.mlp("arc_head", H)
        S = dep @ self.w["arc_W"] @ head.T + (head @ self.w["arc_u"])[None, :]   # S[i, j]: head j of dep i
        heads = chu_liu_edmonds(S)

        ld, lh = self.mlp("lab_dep", H), self.mlp("lab_head", H)
        W = self.w["lab_W"]                                   # (L, a, a)
        out = []
        for i in range(1, len(H)):
            j = heads[i]
            s = np.einsum("a,lab,b->l", ld[i], W, lh[j]) + self.lin("lab_lin", np.concatenate([ld[i], lh[j]]))
            out.append({"form": words[i - 1], "upos": self.labels["upos"][upos[i - 1]],
                        "xpos": self.labels["xpos"][xpos[i - 1]],
                        "feats": self.labels["feats"][feats[i - 1]],
                        "head": int(j), "deprel": self.labels["deprel"][int(s.argmax())]})
        return out

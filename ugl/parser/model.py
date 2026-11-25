"""Neural tagger + dependency parser, inference in pure numpy.

Two encoders share the heads below (meta["encoder"] in the .npz says which):
  "transformer" (v0.3, tools/train_parser_tf.py): a fine-tuned ELECTRA-small
                  (ugl/transformer.py); a word = learned mix of all layers at
                  its first word piece
  "bilstm"      (v0.2, tools/train_parser.py):
    token features  = word emb (trained) + pretrained vector (frozen, PCA 128)
                      + prefix/suffix embs + shape emb + character CNN
    encoder         = 3-layer BiLSTM
    taggers         = linear heads for UPOS, XPOS (Penn tag), FEATS
    arcs            = biaffine scorer over MLP(dep) x MLP(head), decoded to a
                      tree with Chu-Liu/Edmonds (one root)
    labels          = biaffine label scorer on the chosen arcs
Weights live in one .npz; this file is all that is needed to run it.
"""
import json
import os

import numpy as np

from .features import AFFIX_KEYS, affixes, chars, norm, shape
from .mst import chu_liu_edmonds, eisner

DECODERS = {"eisner": eisner, "cle": chu_liu_edmonds}

WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
# the shipped parser; the v0.2 BiLSTM weights (parser-en.npz) remain in git history
# and still load here: the .npz meta says which encoder it uses
DEFAULT_PATH = os.path.join(WEIGHTS, "parser-en-tf.npz")


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def _relu(x):
    return np.maximum(x, 0)


class ParserModel:
    def __init__(self, path=DEFAULT_PATH, data=None, decoder="eisner"):
        """decoder: "eisner" = best projective tree (default; English trees are
        projective), "cle" = Chu-Liu/Edmonds, best tree with crossing arcs allowed."""
        self.path = path
        self.decode_tree = DECODERS[decoder]
        z = np.load(path if data is None else data, allow_pickle=False)
        self.w = {k: z[k].astype(np.float32) for k in z.files if k != "meta"}
        meta = json.loads(str(z["meta"]))
        self.meta = meta
        self.vocab = {k: {s: i for i, s in enumerate(v)} for k, v in meta["vocabs"].items()}
        self.labels = {k: meta["vocabs"][k] for k in ("upos", "xpos", "feats", "deprel")}
        self.kind = meta.get("encoder", "bilstm")
        # arc-score temperature fitted on held-out data (tools/calibrate_parser.py --fit),
        # so a head probability of p means about p of such heads are right
        self.temperature = float(meta.get("arc_temperature", 1.0))
        self.cache = {}                   # words -> per-layer word vectors, last few sentences
        if self.kind == "transformer":
            from ..transformer import Transformer, WordPiece
            t = meta["transformer"]
            self.wp = WordPiece(meta["wordpiece"], meta.get("lowercase", True))
            self.tf = Transformer({k[3:]: v for k, v in self.w.items() if k.startswith("tf.")},
                                  t["layers"], t["heads"], t["eps"])
            for k in [k for k in self.w if k.startswith("tf.")]:
                del self.w[k]                    # held by self.tf only (it may move them to a GPU)
        else:
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
        if "emb.char" in w:
            parts.append(self.char_features(words))
        return np.concatenate(parts, axis=1)

    def char_features(self, words):
        """Character CNN (kernel 3, zero padding), ReLU, max over the word."""
        E = self.w["emb.char"]
        W = self.w["char_conv.weight"]                       # (F, E, 3)
        Wr = W.transpose(0, 2, 1).reshape(W.shape[0], -1)     # (F, 3E), window-major
        b = self.w["char_conv.bias"]
        v = self.vocab["char"]
        out = np.zeros((len(words), W.shape[0]), np.float32)
        for i, word in enumerate(words):
            x = E[[v.get(ch, 1) for ch in chars(word)]]       # (L, E)
            xp = np.concatenate([np.zeros((1, x.shape[1]), np.float32), x,
                                 np.zeros((1, x.shape[1]), np.float32)])
            win = np.concatenate([xp[:-2], xp[1:-1], xp[2:]], axis=1)   # (L, 3E)
            out[i] = np.maximum(win @ Wr.T + b, 0).max(axis=0)
        return out

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
        if self.kind == "transformer":
            return np.tensordot(self.w["mix"], self.layer_states(words), axes=1)
        x = self.embed(words)
        for l in range(self.layers):
            x = np.concatenate([self.lstm_dir(x, l, False), self.lstm_dir(x, l, True)], axis=1)
        return x

    def layer_states(self, words):
        """Transformer: (layers+1, n_words, d) states at each word's first piece.
        Cached for the last sentence, so the sense model can reuse the pass."""
        key = tuple(words)
        if key in self.cache:
            return self.cache[key]
        limit = self.tf.max_len() - 2
        chunks, cur, size = [], [], 0          # a sentence longer than the model's window is read in parts
        for w in words:
            n = len(self.wp.encode_words([w])[0]) - 2
            if cur and size + n > limit:
                chunks.append(cur)
                cur, size = [], 0
            cur.append(w)
            size += n
        chunks.append(cur)
        parts = []
        for c in chunks:
            ids, first = self.wp.encode_words(c)
            hs = self.tf.hidden_states(ids[: limit + 2])
            parts.append(np.stack([h[first] for h in hs]))
        out = np.concatenate(parts, axis=1)
        if len(self.cache) >= 16:
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = out
        return out

    def head_probs(self, S, temperature=None):
        """P[i, j]: calibrated probability that word i+1 is headed by j (0 = ROOT)."""
        T = self.temperature if temperature is None else temperature
        P = S[1:] / T
        np.fill_diagonal(P[:, 1:], -np.inf)                 # a word cannot head itself
        P = np.exp(P - P.max(1, keepdims=True))
        return P / P.sum(1, keepdims=True)

    def arc_scores(self, Hs):
        """S[i, j] = score of word j heading word i (index 0 = ROOT)."""
        H = np.concatenate([self.w["root"][None, :], Hs])
        dep, head = self.mlp("arc_dep", H), self.mlp("arc_head", H)
        return dep @ self.w["arc_W"] @ head.T + (head @ self.w["arc_u"])[None, :]

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
        state = self.score(words)
        return self.tree(state, self.decode_tree(state["S"]))

    def score(self, words):
        """Everything the parser computes for a sentence before choosing a tree."""
        Hs = self.encode(words)                              # (n, d)
        H = np.concatenate([self.w["root"][None, :], Hs])    # (n+1, d), index 0 = ROOT
        S = self.arc_scores(Hs)
        return {"words": words, "S": S, "P": self.head_probs(S),
                "upos": self.lin("upos", Hs).argmax(1), "xpos": self.lin("xpos", Hs).argmax(1),
                "feats": self.lin("feats", Hs).argmax(1),
                "ld": self.mlp("lab_dep", H), "lh": self.mlp("lab_head", H)}

    def tree(self, state, heads):
        """The token dicts for a given choice of heads (labels scored for those arcs)."""
        ld, lh, P, W = state["ld"], state["lh"], state["P"], self.w["lab_W"]
        out = []
        for i in range(1, len(ld)):
            j = heads[i]
            s = np.einsum("a,lab,b->l", ld[i], W, lh[j]) + self.lin("lab_lin", np.concatenate([ld[i], lh[j]]))
            out.append({"form": state["words"][i - 1], "upos": self.labels["upos"][state["upos"][i - 1]],
                        "xpos": self.labels["xpos"][state["xpos"][i - 1]],
                        "feats": self.labels["feats"][state["feats"][i - 1]],
                        "head": int(j), "deprel": self.labels["deprel"][int(s.argmax())],
                        "conf": float(P[i - 1, j])})       # probability of the chosen head
        return out

    def alternative(self, state, tree, i):
        """The best tree in which word i (0-based) does NOT take its chosen head:
        what the sentence would mean if that uncertain attachment were wrong."""
        S = state["S"].copy()
        S[i + 1, tree[i]["head"]] = -1e9
        return self.tree(state, self.decode_tree(S))

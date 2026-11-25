"""A BERT-family transformer encoder (ELECTRA / BERT) in pure numpy, plus its
WordPiece tokenizer in pure Python, so it runs anywhere numpy does (Pyodide
included).  Used by the v0.3 parser (ugl.parser.model) and the browser sense
model (ugl.wsd_context.SharedEmbedder).

Weights come from a PyTorch / Hugging Face checkpoint via export_names();
inference matches it to float16 rounding (checked by tools/train_parser_tf.py).
"""
import math
import unicodedata

import numpy as np


# ------------------------------------------------------------------ WordPiece
def _is_punct(ch):
    cp = ord(ch)
    if 33 <= cp <= 47 or 58 <= cp <= 64 or 91 <= cp <= 96 or 123 <= cp <= 126:
        return True
    return unicodedata.category(ch).startswith("P")


def _is_cjk(cp):
    return (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or 0x20000 <= cp <= 0x2A6DF
            or 0x2A700 <= cp <= 0x2B73F or 0x2B740 <= cp <= 0x2B81F or 0x2B820 <= cp <= 0x2CEAF
            or 0xF900 <= cp <= 0xFAFF or 0x2F800 <= cp <= 0x2FA1F)


class WordPiece:
    """BERT's BasicTokenizer + WordPiece (greedy longest match, '##' pieces)."""

    def __init__(self, vocab_list, lowercase=True):
        self.vocab_list = list(vocab_list)
        self.vocab = {t: i for i, t in enumerate(self.vocab_list)}
        self.lowercase = lowercase
        self.unk, self.cls, self.sep = (self.vocab[t] for t in ("[UNK]", "[CLS]", "[SEP]"))

    def basic(self, text):
        out = []
        for ch in text:
            cp = ord(ch)
            if cp == 0 or cp == 0xFFFD or (unicodedata.category(ch).startswith("C") and ch not in "\t\n\r"):
                continue
            out.append(" " if ch.isspace() else (f" {ch} " if _is_cjk(cp) else ch))
        words = []
        for w in "".join(out).split():
            if self.lowercase:
                w = "".join(c for c in unicodedata.normalize("NFD", w.lower()) if unicodedata.category(c) != "Mn")
            cur = ""
            for ch in w:
                if _is_punct(ch):
                    if cur:
                        words.append(cur)
                    words.append(ch)
                    cur = ""
                else:
                    cur += ch
            if cur:
                words.append(cur)
        return words

    def pieces(self, word):
        if len(word) > 100:
            return [self.unk]
        out, start = [], 0
        while start < len(word):
            end, cur = len(word), None
            while start < end:
                sub = word[start:end] if start == 0 else "##" + word[start:end]
                if sub in self.vocab:
                    cur = self.vocab[sub]
                    break
                end -= 1
            if cur is None:
                return [self.unk]
            out.append(cur)
            start = end
        return out

    def encode_words(self, words):
        """-> (piece ids with [CLS] ... [SEP], index of each word's first piece)."""
        ids, first = [self.cls], []
        for w in words:
            p = [i for b in self.basic(w) for i in self.pieces(b)] or [self.unk]
            first.append(len(ids))
            ids += p
        ids.append(self.sep)
        return ids, first


# ------------------------------------------------------------------ encoder
def export_names(hf_key):
    """Hugging Face state-dict key -> name stored in the .npz (None = skip)."""
    if hf_key.endswith("position_ids") or hf_key.endswith("token_type_ids"):
        return None
    return hf_key.replace("encoder.layer.", "L.")


def _erf(x):
    # Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7): exact-GELU without scipy
    s = np.sign(x)
    a = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * a)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-a * a)
    return s * y


def _gelu(x):
    return 0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)))


def _ln(x, g, b, eps):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * g + b


# Optional GPU for corpus building: use_device("cuda") runs every transformer
# pass in PyTorch (float32, TF32 off, so results match the numpy path to float
# rounding).  The default (None) is pure numpy: the browser never imports torch.
DEVICE = None


def use_device(device):
    global DEVICE
    DEVICE = device


class Transformer:
    """w: dict of float32 arrays named by export_names() with prefix stripped."""

    def __init__(self, w, layers, heads, eps=1e-12):
        self.w, self.layers, self.heads, self.eps = w, layers, heads, eps
        self._tw = None                     # the weights on DEVICE, made on first use

    def lin(self, name, x):
        return x @ self.w[name + ".weight"].T + self.w[name + ".bias"]

    def _torch_weights(self):
        if self._tw is None:
            import torch
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            self._tw = {k: torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32)).to(DEVICE)
                        for k, v in self.w.items() if k.startswith(("embeddings", "L."))}
            # the layers now live on the device; their numpy copies (most of the
            # model) are not read again in this process
            for k in [k for k in self.w if k.startswith("L.")]:
                del self.w[k]
        return self._tw

    def _hidden_states_torch(self, ids):
        """hidden_states() on DEVICE: the same computation, step for step."""
        import torch
        import torch.nn.functional as F
        w, eps, H = self._torch_weights(), self.eps, self.heads
        T = len(ids)
        with torch.no_grad():
            t = torch.tensor(ids, device=DEVICE)
            x = (w["embeddings.word_embeddings.weight"][t] + w["embeddings.position_embeddings.weight"][:T]
                 + w["embeddings.token_type_embeddings.weight"][0])
            x = F.layer_norm(x, (x.shape[-1],), w["embeddings.LayerNorm.weight"], w["embeddings.LayerNorm.bias"], eps)
            if "embeddings_project.weight" in w:
                x = F.linear(x, w["embeddings_project.weight"], w["embeddings_project.bias"])
            out = [x]
            for l in range(self.layers):
                p = f"L.{l}."
                d = x.shape[1]

                def lin(n, z):
                    return F.linear(z, w[p + n + ".weight"], w[p + n + ".bias"])
                q = lin("attention.self.query", x).view(T, H, d // H).transpose(0, 1)
                k = lin("attention.self.key", x).view(T, H, d // H).transpose(0, 1)
                v = lin("attention.self.value", x).view(T, H, d // H).transpose(0, 1)
                s = (q @ k.transpose(1, 2) / math.sqrt(d // H)).softmax(-1)
                ctx = (s @ v).transpose(0, 1).reshape(T, d)
                x = F.layer_norm(lin("attention.output.dense", ctx) + x, (d,),
                                 w[p + "attention.output.LayerNorm.weight"], w[p + "attention.output.LayerNorm.bias"], eps)
                h = F.gelu(lin("intermediate.dense", x))
                x = F.layer_norm(lin("output.dense", h) + x, (d,),
                                 w[p + "output.LayerNorm.weight"], w[p + "output.LayerNorm.bias"], eps)
                out.append(x)
            return list(torch.stack(out).cpu().numpy())

    def hidden_states_batch(self, seqs):
        """Many sequences in one padded pass on DEVICE (padding is masked out of
        attention, so each result equals hidden_states() of that sequence alone
        to float rounding) -> list of (layers+1, T_i, d) arrays."""
        import torch
        import torch.nn.functional as F
        w, eps, H = self._torch_weights(), self.eps, self.heads
        B, T = len(seqs), max(len(s) for s in seqs)
        with torch.no_grad():
            ids = torch.zeros((B, T), dtype=torch.long, device=DEVICE)
            keep = torch.zeros((B, T), dtype=torch.bool, device=DEVICE)
            for b, s in enumerate(seqs):
                ids[b, :len(s)] = torch.tensor(s, device=DEVICE)
                keep[b, :len(s)] = True
            mask = torch.zeros((B, 1, 1, T), device=DEVICE).masked_fill(~keep[:, None, None, :], float("-inf"))
            x = (w["embeddings.word_embeddings.weight"][ids] + w["embeddings.position_embeddings.weight"][:T][None]
                 + w["embeddings.token_type_embeddings.weight"][0])
            x = F.layer_norm(x, (x.shape[-1],), w["embeddings.LayerNorm.weight"], w["embeddings.LayerNorm.bias"], eps)
            if "embeddings_project.weight" in w:
                x = F.linear(x, w["embeddings_project.weight"], w["embeddings_project.bias"])
            out = [x]
            for l in range(self.layers):
                p = f"L.{l}."
                d = x.shape[-1]

                def lin(n, z):
                    return F.linear(z, w[p + n + ".weight"], w[p + n + ".bias"])
                q = lin("attention.self.query", x).view(B, T, H, d // H).transpose(1, 2)
                k = lin("attention.self.key", x).view(B, T, H, d // H).transpose(1, 2)
                v = lin("attention.self.value", x).view(B, T, H, d // H).transpose(1, 2)
                s = (q @ k.transpose(2, 3) / math.sqrt(d // H) + mask).softmax(-1)
                ctx = (s @ v).transpose(1, 2).reshape(B, T, d)
                x = F.layer_norm(lin("attention.output.dense", ctx) + x, (d,),
                                 w[p + "attention.output.LayerNorm.weight"], w[p + "attention.output.LayerNorm.bias"], eps)
                h = F.gelu(lin("intermediate.dense", x))
                x = F.layer_norm(lin("output.dense", h) + x, (d,),
                                 w[p + "output.LayerNorm.weight"], w[p + "output.LayerNorm.bias"], eps)
                out.append(x)
            st = torch.stack(out, 1).cpu().numpy()          # (B, layers+1, T, d)
        return [st[b, :, :len(s)] for b, s in enumerate(seqs)]

    def hidden_states(self, ids):
        """ids: piece ids of one sequence -> list of (T, d) arrays, layer 0..L."""
        if DEVICE is not None:
            return self._hidden_states_torch(ids)
        w, eps = self.w, self.eps
        T = len(ids)
        x = (w["embeddings.word_embeddings.weight"][ids] + w["embeddings.position_embeddings.weight"][:T]
             + w["embeddings.token_type_embeddings.weight"][0])
        x = _ln(x, w["embeddings.LayerNorm.weight"], w["embeddings.LayerNorm.bias"], eps)
        if "embeddings_project.weight" in w:
            x = self.lin("embeddings_project", x)
        out = [x]
        H = self.heads
        for l in range(self.layers):
            p = f"L.{l}."
            d = x.shape[1]
            q = self.lin(p + "attention.self.query", x).reshape(T, H, d // H).transpose(1, 0, 2)
            k = self.lin(p + "attention.self.key", x).reshape(T, H, d // H).transpose(1, 0, 2)
            v = self.lin(p + "attention.self.value", x).reshape(T, H, d // H).transpose(1, 0, 2)
            s = q @ k.transpose(0, 2, 1) / math.sqrt(d // H)
            s = np.exp(s - s.max(-1, keepdims=True))
            s /= s.sum(-1, keepdims=True)
            ctx = (s @ v).transpose(1, 0, 2).reshape(T, d)
            x = _ln(self.lin(p + "attention.output.dense", ctx) + x,
                    w[p + "attention.output.LayerNorm.weight"], w[p + "attention.output.LayerNorm.bias"], eps)
            h = _gelu(self.lin(p + "intermediate.dense", x))
            x = _ln(self.lin(p + "output.dense", h) + x,
                    w[p + "output.LayerNorm.weight"], w[p + "output.LayerNorm.bias"], eps)
            out.append(x)
        return out

    def max_len(self):
        return self.w["embeddings.position_embeddings.weight"].shape[0]

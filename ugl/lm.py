"""Run a trained UGL or English story model: give it text, it writes what comes next.

    import ugl.lm
    m = ugl.lm.load("ugl-stories")            # downloaded once, then cached
    m.continue_text("Once upon a time, there was a little dog named Max.")
    # -> 'Max liked to play in the park. One day, ...'

    m = ugl.lm.load("path/to/model-folder")   # or a local folder

A model folder is the standard Hugging Face GPT-2 layout, so the same files
also load in `transformers` (GPT2LMHeadModel.from_pretrained) and convert
with the usual tools (llama.cpp's convert_hf_to_gguf.py, optimum's ONNX export):

    config.json         GPT-2 settings, plus a "ugl" section (language, dictionary version)
    model.safetensors   the weights (Hugging Face GPT-2 tensor names)
    tokenizer.json      Hugging Face tokenizer: byte-level BPE (English) or one
                        entry per UGL token (UGL: model id <-> UGL token text)

This module runs the model with numpy alone (no torch), on any machine.
An English model also needs `tokenizers` (pip install ugl[lm]).  A UGL model
uses UGL's own encoder and decoder: the prompt is encoded to UGL, the model
continues in UGL, and the continuation is decoded back to English.

Models are trained by tools/train_lm.py and packaged by tools/export_lm.py.
"""
import json
import math
import os
import struct
import urllib.request

import numpy as np

from .transformer import _gelu, _ln

PAD, EOS, BRK, UNK = 0, 1, 2, 3        # the same ids in every model (tools/train_lm.py)
FILES = ["config.json", "model.safetensors", "tokenizer.json"]
RELEASE = "https://github.com/mountaindewsalesman/ugl/releases/download/lm-{name}/{file}"


# ------------------------------------------------------------------ files
_DTYPES = {"F32": np.float32, "F16": np.float16, "I64": np.int64, "I32": np.int32}


def read_safetensors(path):
    """{name: float32 array} from a .safetensors file (numpy only)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        data = f.read()
    out = {}
    for name, t in header.items():
        if name == "__metadata__":
            continue
        b, e = t["data_offsets"]
        raw = data[b:e]
        if t["dtype"] == "BF16":
            a = (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32)
        else:
            a = np.frombuffer(raw, _DTYPES[t["dtype"]])
        a = a.reshape(t["shape"])
        out[name] = a.astype(np.float32) if a.dtype.kind == "f" else a
    return out


def write_safetensors(path, tensors, metadata=None):
    """Write {name: numpy array} as .safetensors (float16/float32/int)."""
    names = {np.dtype(v): k for k, v in _DTYPES.items()}
    header, chunks, off = {}, [], 0
    for name in sorted(tensors):
        a = np.ascontiguousarray(tensors[name])
        raw = a.tobytes()
        header[name] = {"dtype": names[a.dtype], "shape": list(a.shape), "data_offsets": [off, off + len(raw)]}
        chunks.append(raw)
        off += len(raw)
    if metadata:
        header["__metadata__"] = {k: str(v) for k, v in metadata.items()}
    h = json.dumps(header, separators=(",", ":")).encode("utf-8")
    h += b" " * (-len(h) % 8)              # the data starts 8-byte aligned
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h)
        for c in chunks:
            f.write(c)


def cache_dir():
    return os.environ.get("UGL_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "ugl")


def download(name, quiet=False):
    """A released model's folder in the local cache, downloaded the first time.
    Model `name` is published as GitHub release lm-<name>."""
    folder = os.path.join(cache_dir(), "models", name)
    os.makedirs(folder, exist_ok=True)
    for file in FILES:
        path = os.path.join(folder, file)
        if os.path.exists(path):
            continue
        url = RELEASE.format(name=name, file=file)
        if not quiet:
            print(f"downloading {url}", flush=True)
        try:
            urllib.request.urlretrieve(url, path + ".part")
        except Exception as e:
            raise OSError(f"could not download model {name!r} ({url}): {e}") from None
        os.replace(path + ".part", path)
    return folder


def load(name_or_folder):
    """A local model folder, or the name of a released model (downloaded once)."""
    folder = name_or_folder if os.path.isdir(name_or_folder) else download(name_or_folder)
    return LM(folder)


# ------------------------------------------------------------------ model
class GPT:
    """GPT-2 forward pass in numpy with a key/value cache, for generation."""

    def __init__(self, w, cfg):
        self.w = w
        self.layers, self.heads = cfg["n_layer"], cfg["n_head"]
        self.ctx = cfg["n_positions"]
        self.eps = cfg.get("layer_norm_epsilon", 1e-5)
        self.wte = w["transformer.wte.weight"]
        self.head = w.get("lm_head.weight", self.wte)      # tied unless stored separately

    def _ln(self, p, x):
        return _ln(x, self.w[p + ".weight"], self.w[p + ".bias"], self.eps)

    def _lin(self, p, x):
        return x @ self.w[p + ".weight"] + self.w[p + ".bias"]   # Hugging Face Conv1D: (in, out)

    def forward(self, ids, cache):
        """Logits (T, vocab) for new token ids, appending their keys/values to
        cache (a list per layer, [] to start)."""
        w = self.w
        start = cache[0][0].shape[1] if cache else 0
        T = len(ids)
        x = self.wte[ids] + w["transformer.wpe.weight"][start:start + T]
        d = x.shape[1]
        hd = d // self.heads
        if not cache:
            cache.extend([None, None] for _ in range(self.layers))
        mask = np.triu(np.full((T, start + T), -np.inf, np.float32), start + 1)
        for i in range(self.layers):
            p = f"transformer.h.{i}."
            q, k, v = self._lin(p + "attn.c_attn", self._ln(p + "ln_1", x)).reshape(T, 3, self.heads, hd).transpose(1, 2, 0, 3)
            if cache[i][0] is not None:
                k = np.concatenate([cache[i][0], k], 1)
                v = np.concatenate([cache[i][1], v], 1)
            cache[i] = [k, v]
            s = q @ k.transpose(0, 2, 1) / math.sqrt(hd) + mask
            s = np.exp(s - s.max(-1, keepdims=True))
            a = (s / s.sum(-1, keepdims=True)) @ v                   # (heads, T, hd)
            x = x + self._lin(p + "attn.c_proj", a.transpose(1, 0, 2).reshape(T, d))
            h = _gelu(self._lin(p + "mlp.c_fc", self._ln(p + "ln_2", x)))
            x = x + self._lin(p + "mlp.c_proj", h)
        return self._ln("transformer.ln_f", x) @ self.head.T

    def generate(self, ids, n, temperature=0.8, top_k=40, rng=None, stop=(EOS,), max_breaks=None):
        """Up to n new ids after ids; stops at a stop id or after max_breaks sentence breaks."""
        rng = rng or np.random.default_rng()
        seen = list(ids)[-self.ctx:]            # what the cache holds, in order
        cache = []
        logits = self.forward(seen, cache)[-1]
        out, breaks = [], 0
        for _ in range(n):
            if temperature <= 0:
                nxt = int(logits.argmax())
            else:
                z = logits.astype(np.float64) / temperature
                if top_k and top_k < len(z):
                    z[z < np.partition(z, -top_k)[-top_k]] = -np.inf
                pr = np.exp(z - z.max())
                nxt = int(rng.choice(len(pr), p=pr / pr.sum()))
            out.append(nxt)
            if nxt in stop:
                break
            breaks += nxt == BRK
            if max_breaks is not None and breaks >= max_breaks:
                break
            seen.append(nxt)
            if len(seen) >= self.ctx:
                # the window is full: keep its second half and recompute (positions restart at 0)
                seen = seen[-self.ctx // 2:]
                cache = []
                logits = self.forward(seen, cache)[-1]
            else:
                logits = self.forward([nxt], cache)[-1]
        return out


class LM:
    """A packaged model folder: tokenizer + GPT, English in, English out."""

    def __init__(self, folder):
        with open(os.path.join(folder, "config.json"), encoding="utf-8") as f:
            self.config = json.load(f)
        info = self.config.get("ugl", {})
        self.lang = info.get("lang", "english")
        self.model = GPT(read_safetensors(os.path.join(folder, "model.safetensors")), self.config)
        with open(os.path.join(folder, "tokenizer.json"), encoding="utf-8") as f:
            tj = f.read()
        if self.lang == "ugl":
            import ugl
            self.ugl = ugl.load(info.get("dictionary"))
            vocab = json.loads(tj)["model"]["vocab"]            # UGL token text -> model id
            tid = self.ugl.tokenizer.id
            self.to_model = {tid[s]: i for s, i in vocab.items() if s in tid}
            self.to_tok = {i: t for t, i in self.to_model.items()}
        else:
            try:
                from tokenizers import Tokenizer
            except ImportError:
                raise ImportError("an English model needs the tokenizers package: pip install ugl[lm]") from None
            self.bpe = Tokenizer.from_str(tj)

    # -------------------------------------------------- text <-> model ids
    def _sentences(self, text):
        from .parser.tokenize import split_sentences
        return [s.strip() for s in split_sentences(text) if s.strip()]

    def prompt_ids(self, text):
        """[EOS] + the prompt's sentences with a break after each (every model
        writes whole new sentences after a prompt)."""
        ids = [EOS]
        if self.lang == "ugl":
            for sent in self.ugl.encode(text):
                for line in sent.split("\n"):                 # as tools/train_lm.py tokenizes
                    ids += [self.to_model.get(t, UNK) for t in self.ugl.tokenizer.encode(line)]
                ids.append(BRK)
        else:
            for s in self._sentences(text):
                ids += self.bpe.encode(s).ids + [BRK]
        return ids

    def sentences(self, ids):
        """Generated model ids -> list of sentences ({"english", "ugl"})."""
        groups, cur = [], []
        for i in ids:
            if i == EOS:
                break
            if i == BRK:
                groups.append(cur)
                cur = []
            elif i > UNK:
                cur.append(i)
        groups.append(cur)
        out = []
        for g in (g for g in groups if g):
            if self.lang == "ugl":
                u = self.ugl.tokenizer.decode([self.to_tok[i] for i in g if i in self.to_tok])
                try:
                    en = self.ugl.decode(u)
                except Exception:  # noqa: BLE001  a malformed generated sentence spoils no other
                    en = None
                out.append({"english": en, "ugl": u})
            else:
                out.append({"english": self.bpe.decode(g).strip(), "ugl": None})
        return out

    # -------------------------------------------------- the public call
    def continue_text(self, prompt, sentences=5, max_tokens=400, temperature=0.8, top_k=40, seed=None,
                      details=False):
        """English prompt -> the next `sentences` sentences, as English text.
        details=True: the list of {"english", "ugl"} per sentence instead (a UGL
        sentence the decoder can't read has english None)."""
        ids = self.prompt_ids(prompt)
        new = self.model.generate(ids, max_tokens, temperature, top_k, np.random.default_rng(seed),
                                  max_breaks=sentences)
        got = self.sentences(new)
        if details:
            return got
        return " ".join(s["english"] for s in got if s["english"])

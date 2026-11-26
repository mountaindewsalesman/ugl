"""UGL text <-> integer ids, for training language models.

Every concept is exactly one id, and the ids follow the dictionary's frequency
order, so cutting the vocabulary to the first N concepts keeps the N most
frequent meanings.  Modifier letters are separate ids (there are only ~20), so
"n988st" is [n988, +s, +t].  Closing a reserved relation verb is one id.
Literals (rare names, numbers, unknown words) are spelled with the
dictionary's subword pieces (BPE), then characters, then UTF-8 bytes, between
a start marker and </lit>.

    tok = Tokenizer(lexicon)
    ids = tok.encode("n988st v9 n988 v4520 \\v9")
    tok.decode(ids)       # -> the same string
"""
import re

from .core import RESERVED_NOUNS, RESERVED_VERBS, WORD_RE, MOD_ORDER, parse_literal

# Token ids are a function of (dictionary version, SCHEME).  Bump SCHEME
# whenever the id layout changes; record both next to any tokenized data.
#   1: first layout
#   2: one-id closing tags for reserved verbs; literal pieces from the dictionary
#   3: modifier letters b (non-) and i (anti-)
#   4: back-references @1..@9 (rule 15)
SCHEME = 4

SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>", "<sep>"]
PIECE = "\u25b8"            # vocab prefix for a multi-character literal piece
STRUCT = ["<close>", "<nv>", "<num>", "<name>", "<unk_n>", "<unk_v>", "<unk_a>", "</lit>"]
CHARS = [chr(c) for c in range(33, 127)]          # printable ASCII, no space
BYTES = [f"<0x{b:02X}>" for b in range(256)]


class Tokenizer:
    def __init__(self, lexicon, max_concepts=None):
        """max_concepts: keep only the first N lexical concepts of each class
        (most frequent); the rest map to <unk>."""
        # closing a reserved relation/logic verb is one id (\\v20); closing an
        # ordinary verb is <close> + its id
        closes = [f"\\v{i}" for i in sorted(RESERVED_VERBS)]
        pieces = list(getattr(lexicon, "literal_pieces", []) or [])
        refs = [f"@{k}" for k in range(1, 10)]
        vocab = list(SPECIALS) + list(STRUCT) + refs + closes + [f"+{c}" for c in MOD_ORDER] + CHARS + BYTES
        vocab += [PIECE + p for p in pieces]
        vocab += [f"n{i}" for i in sorted(RESERVED_NOUNS)] + [f"v{i}" for i in sorted(RESERVED_VERBS)]
        from .pieces import Segmenter
        self.segment = Segmenter(pieces)
        by_class = {"n": [], "v": [], "a": []}
        for t in lexicon.ids:
            by_class[t[0]].append(t)
        for c in "nva":
            lst = sorted(by_class[c], key=lambda t: int(t[1:]))
            if max_concepts:
                lst = lst[:max_concepts]
            vocab += lst
        self.vocab = vocab
        self.id = {t: i for i, t in enumerate(vocab)}
        self.unk = self.id["<unk>"]
        self.version = lexicon.version
        self.scheme = SCHEME

    def __len__(self):
        return len(self.vocab)

    # ------------------------------------------------------------ encode
    def _literal(self, start, text):
        out = [self.id[start]]
        for piece in self.segment(text):
            if len(piece) > 1:
                out.append(self.id[PIECE + piece])
            elif piece in self.id:
                out.append(self.id[piece])
            else:
                out += [self.id[f"<0x{b:02X}>"] for b in piece.encode("utf-8")]
        out.append(self.id["</lit>"])
        return out

    def encode_token(self, t):
        if t.startswith("\\"):
            if t in self.id:
                return [self.id[t]]
            return [self.id["<close>"], self.id.get(t[1:], self.unk)]
        m = WORD_RE.match(t)
        if m:
            cls, idx, mods = m.groups()
            out = []
            if cls == "nv":
                out.append(self.id["<nv>"])
                cls = "v"
            out.append(self.id.get(f"{cls}{idx}", self.unk))
            return out + [self.id["+" + c] for c in mods]
        if t.startswith("@"):
            return [self.id[t[:2]]] + [self.id["+" + c] for c in t[2:]]
        if t.startswith("#"):
            return self._literal("<num>", t[1:])
        if t.startswith("'"):
            body, _, mods = t[1:].rpartition("'")
            return self._literal("<name>", body) + [self.id["+" + c] for c in mods]
        if len(t) > 2 and t[1] == "?":
            _, cls, text, mods = parse_literal(t)
            return self._literal(f"<unk_{cls}>", text.replace(" ", "_")) + [self.id["+" + c] for c in mods]
        return [self.unk]

    def encode(self, ugl, bos=False, eos=False):
        """UGL string (sentences separated by newlines) -> list of ids."""
        ids = [self.id["<bos>"]] if bos else []
        for k, line in enumerate(l for l in ugl.split("\n") if l.strip()):
            if k:
                ids.append(self.id["<sep>"])
            for t in line.split():
                ids += self.encode_token(t)
        if eos:
            ids.append(self.id["<eos>"])
        return ids

    # ------------------------------------------------------------ decode
    def decode(self, ids):
        v = self.vocab
        out, line, i = [], [], 0
        while i < len(ids):
            t = v[ids[i]]
            i += 1
            if t in ("<pad>", "<bos>", "<eos>"):
                continue
            if t == "<sep>":
                out.append(" ".join(line)); line = []
                continue
            if t.startswith("+") and len(t) == 2 and line:
                line[-1] += t[1]          # modifiers follow words, names and n?'..' literals alike
                continue
            if t.startswith("\\v"):
                line.append(t)
                continue
            if t == "<close>":
                line.append("\\" + (v[ids[i]] if i < len(ids) else "")); i += 1
                continue
            if t == "<nv>":
                nxt = v[ids[i]] if i < len(ids) else "v0"; i += 1
                line.append("nv" + nxt[1:])
                continue
            if t in ("<num>", "<name>", "<unk_n>", "<unk_v>", "<unk_a>"):
                buf = bytearray()
                while i < len(ids) and v[ids[i]] != "</lit>":
                    x = v[ids[i]]
                    if x.startswith(PIECE):
                        x = x[len(PIECE):]
                    buf += bytes([int(x[3:5], 16)]) if x.startswith("<0x") else x.encode("utf-8")
                    i += 1
                i += 1
                s = buf.decode("utf-8", errors="replace")
                line.append("#" + s if t == "<num>" else "'" + s + "'" if t == "<name>" else f"{t[5]}?'{s}'")
                continue
            line.append(t)
        if line:
            out.append(" ".join(line))
        return "\n".join(out)

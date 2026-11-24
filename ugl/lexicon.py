"""Runtime dictionary: English word sense <-> UGL token(s).

A dictionary is one self-contained JSON file (optionally gzipped), see
DICTIONARY_FORMAT.md.  The frozen default ships inside the package in
ugl/dictionary/ and is named in ugl/dictionary/default.txt.  Loading needs
nothing but the standard library.
"""
import gzip
import json
import os
from functools import lru_cache

from .core import RESERVED_NOUNS, RESERVED_VERBS, WORD_RE, add_mods

DICT_DIR = os.path.join(os.path.dirname(__file__), "dictionary")

# UGL class -> dictionary index classes searched, in order
SEARCH = {"n": ["n"], "v": ["v"], "a": ["a", "r"], "r": ["r", "a"]}


def default_path():
    with open(os.path.join(DICT_DIR, "default.txt"), encoding="utf-8") as f:
        return os.path.join(DICT_DIR, f.read().strip())


def available():
    """Dictionary files shipped with the package."""
    return sorted(f for f in os.listdir(DICT_DIR) if f.endswith((".json", ".json.gz")))


def read_dictionary(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        d = json.load(f)
    if d.get("format") != "ugl-dictionary":
        raise ValueError(f"{path} is not a UGL dictionary")
    if d.get("format_version") != 1:
        raise ValueError(f"unsupported dictionary format_version {d.get('format_version')}")
    return d


class Lexicon:
    def __init__(self, path=None):
        """path: a dictionary file; None = the packaged default; a bare file
        name = one of the packaged dictionaries (see ugl.lexicon.available())."""
        if path is None:
            path = default_path()
        elif not os.path.exists(path) and os.path.exists(os.path.join(DICT_DIR, path)):
            path = os.path.join(DICT_DIR, path)
        self.path = path
        d = read_dictionary(path)
        self.version = d["version"]
        self.data = d
        self.ids = d["primitives"]           # "n100" -> {"lemma","sense","freq","gloss"}
        self.senses_ = d["senses"]           # "dog.n.01" -> {"tokens","gloss","why"?}
        self.index = d["index"]              # class -> lemma -> [sense names]
        self.person = set(d["person_senses"])
        self.classes = {k: set(v) for k, v in d.get("classes", {"person": d["person_senses"]}).items()}
        self.names = d.get("names", {})      # exact proper name -> sense
        self.literal_pieces = d.get("literal_pieces", [])
        # unit symbols ("°C", "%") -> the senses they can mean (dictionary 0.6.0+)
        self.symbols = d.get("symbols", {})
        self.wsd = None                      # optional sense disambiguator (see ugl.wsd)

    # ------------------------------------------------------------ encode side
    def senses(self, lemma, cls):
        lemma = lemma.lower().replace("_", " ")
        out = []
        for c in SEARCH[cls]:
            out += [s for s in self.index[c].get(lemma, ()) if s in self.senses_]
        return list(dict.fromkeys(out))

    def name_tokens(self, text, mods=""):
        """Tokens for a frequent proper name, or None (then it stays a literal)."""
        s = self.names.get(text)
        if s is None and text.lower().startswith("the "):
            s = self.names.get(text[4:])
        if s is None or s not in self.senses_:
            return None
        toks = list(self.senses_[s]["tokens"])
        toks[0] = add_mods(toks[0], mods)
        return toks

    def is_proper(self, token):
        e = self.entry(token)
        return bool(e) and e.get("proper", False)

    def gloss(self, sense):
        return self.senses_[sense]["gloss"]

    def in_class(self, lemma, cls):
        """Is the most frequent noun sense of lemma in semantic class cls (person/time/place)?"""
        s = self.index["n"].get(lemma.lower())
        return bool(s) and s[0] in self.classes.get(cls, ())

    def name_is_person(self, text):
        """True / False when the dictionary knows whether this name is a person
        ("Simon" yes, "Paris" no); None for names it does not know."""
        s = self.names.get(text) or (self.names.get(text[4:]) if text.lower().startswith("the ") else None)
        if s is None or s not in self.senses_:
            return None
        return s in self.person

    def is_person(self, lemma):
        s = self.index["n"].get(lemma.lower())
        return bool(s) and s[0] in self.person

    def choose_sense(self, lemma, cls, context=(), sent=None, only=None):
        cands = self.senses(lemma, cls)
        if only:                                    # e.g. a symbol's possible senses
            cands = [s for s in only if s in self.senses_] or cands
        if not cands:
            return None
        # names are looked up by name_tokens(); a word reaching here is a common
        # word, so proper senses ("the Writings" for writings) go unless nothing else fits
        common = [s for s in cands if not self.is_proper(self.senses_[s]["tokens"][0])]
        cands = common or cands
        if len(cands) == 1:
            return cands[0]
        if self.wsd is not None and sent is not None:
            r = self.wsd.choose(lemma, cands, sent)
            if r:
                return r
        # fallback: most frequent sense (index order)
        return cands[0]

    def encode_word(self, lemma, cls, mods="", context=(), sent=None, only=None):
        """List of UGL tokens for a word (mods go on the head token), or None.
        only: restrict the choice to these senses."""
        s = self.choose_sense(lemma, cls, context, sent, only)
        if s is None:
            return None
        toks = list(self.senses_[s]["tokens"])
        toks[0] = add_mods(toks[0], mods)
        return toks

    # ------------------------------------------------------------ decode side
    @lru_cache(maxsize=None)
    def lemma_of(self, cls, idx):
        idx = int(idx)
        if cls == "n" and idx in RESERVED_NOUNS:
            return RESERVED_NOUNS[idx]
        if cls == "v" and idx in RESERVED_VERBS:
            return RESERVED_VERBS[idx][0]
        e = self.ids.get(f"{cls}{idx}")
        return e["lemma"] if e else f"<{cls}{idx}?>"

    def entry(self, token):
        m = WORD_RE.match(token)
        if not m:
            return None
        cls = "v" if m.group(1) == "nv" else m.group(1)
        return self.ids.get(f"{cls}{m.group(2)}")

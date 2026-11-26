"""Runtime dictionary: English word sense <-> UGL token(s).

A dictionary is one self-contained JSON file (optionally gzipped), see
DICTIONARY_FORMAT.md.  The frozen default ships inside the package in
ugl/dictionary/ and is named in ugl/dictionary/default.txt.  Loading needs
nothing but the standard library.
"""
import gzip
import re
import json
import os
from functools import lru_cache

from .core import PREFIXES, RESERVED_NOUNS, RESERVED_VERBS, WORD_RE, add_mods

DICT_DIR = os.path.join(os.path.dirname(__file__), "dictionary")

# closed-class English words: never chosen as the surface word of a concept by
# frequency (their counts are of their grammatical use)
FUNCTION_WORDS = set("""a an the this that these those and or but nor so yet for if then than as because
since while though although unless until when where why how what which who whom whose of in on at by to
from with without about above below under over into onto out up down off through across along around
between among against during before after near not no yes all any some each every both either neither
he she it they we you i me him her us them his its their our your my mine one
be do have will would shall should can could may might must""".split())   # auxiliaries: counted as such

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
        if not out and "-" in lemma:
            # one word, three spellings: "long-standing" = "longstanding", "check-in" = "check in"
            for variant in (lemma.replace("-", ""), lemma.replace("-", " ")):
                for c in SEARCH[cls]:
                    out += [s for s in self.index[c].get(variant, ()) if s in self.senses_]
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
        if s.startswith("name:"):
            # a frequent name numbered for its frequency alone ("Tim", "Max"): the
            # dictionary has no WordNet sense saying what it names, so it is unknown,
            # not "not a person" (which would stop "he" / "his" from meaning Tim)
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
            return self.prefixed(lemma, cls, mods, context, sent)
        toks = list(self.senses_[s]["tokens"])
        toks[0] = add_mods(toks[0], mods)
        return toks

    def prefixed(self, lemma, cls, mods="", context=(), sent=None):
        """A word the dictionary lacks, built from a grammatical prefix (core.PREFIXES)
        and a stem it has: "non-member" = member + b, "anti-inflammatory" =
        inflammatory + i.  An adjective whose stem is a noun is that noun used as
        an attribute: "anti-war" = war + j + i.  None when nothing fits."""
        low = lemma.lower()
        for pre, letters in PREFIXES.items():
            if not low.startswith(pre) or len(low) < len(pre) + 3:
                continue
            stem = low[len(pre):]
            if stem.startswith("-"):
                stem = stem[1:]
                if self.senses(pre + stem, cls):          # "non-toxic" spelled as the word "nontoxic"
                    return self.encode_word(pre + stem, cls, mods, context, sent)
            if letters.get(cls) and self.senses(stem, cls):
                return self.encode_word(stem, cls, mods + letters[cls], context, sent)
            if cls in ("a", "r") and self.senses(stem, "n"):
                return self.encode_word(stem, "n", "j" + letters["n"], context, sent)
        return None

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

    def _first_sense_owners(self):
        """(index class, token) -> [(number of senses, lemma)] for every lemma whose
        most frequent common sense is that one token: the words that, read
        without context, mean exactly this concept."""
        if getattr(self, "_owners", None) is None:
            own = {}
            for c, lemmas in self.index.items():
                for lemma, senses in lemmas.items():
                    common = [s for s in senses if s in self.senses_
                              and not self.is_proper(self.senses_[s]["tokens"][0])]
                    if common and len(self.senses_[common[0]]["tokens"]) == 1:
                        own.setdefault((c, self.senses_[common[0]]["tokens"][0]), []).append((len(common), lemma))
            self._owners = own
        return self._owners

    @lru_cache(maxsize=None)
    def surface(self, cls, idx, adverb=False):
        """The English word to write for a concept: its own lemma when that word
        mostly means it, else the least ambiguous word that does ("key" for the
        'of primary importance' sense whose lemma "central" mostly means
        something else; "currently" for the adverb of 'present').  Returns None
        when no word fits better than the lemma (the caller uses lemma_of)."""
        token = f"{cls}{int(idx)}"
        e = self.ids.get(token)
        if e is None:
            return None
        own = self._first_sense_owners()
        c = "r" if adverb else cls                # an adjective slot needs an adjective
        cands = own.get((c, token), [])
        mine = e["lemma"].replace("_", " ").lower()
        mine_ok = any(lemma == mine for _, lemma in cands)    # its own word mostly means it
        if adverb and any(lemma == mine + "ly" or lemma == mine[:-1] + "ily" for _, lemma in cands):
            return None                           # the ordinary -ly form means it: let the caller build it
        # one plain word only: "date stamp the manuscript" reads worse than "date"
        plain = [x for x in cands if x[1].isalpha()]
        if self.word_freq is not None:
            # a function word's count is its function-word use ("but" the conjunction,
            # not "but" = merely): it says nothing about it as a content word
            plain = [x for x in plain if x[1] not in FUNCTION_WORDS]
        if mine_ok:
            plain = [x for x in plain if x[1] != mine] + [(0, mine)]
        freq = self.word_freq
        if freq is not None:
            # the word English uses most for this concept ("big", not "large"; "mom",
            # not "mummy"), among the words that mean it first, so a reader (and the
            # encoder) reads the English back as the same concept
            if not plain:
                return None
            best = max(plain, key=lambda x: (self.lemma_freq(x[1], c), x[1] == mine, x[0], -len(x[1])))[1]
            if not mine_ok and not adverb and \
                    self.lemma_freq(best, c) < 0.1 * self.lemma_freq(mine, c):
                # its own word means something else first, but every word that means
                # this first is rare: "interactive", not "interactional"
                return None
        elif not plain or mine_ok:
            return None
        else:
            # no counts: the commonest word (more senses goes with more use), then the shorter
            best = max(plain, key=lambda x: (x[0], -len(x[1]), x[1]))[1]
        return None if best == mine else best

    def compound_word(self, head, adjs):
        """The one English word for a noun the dictionary spells as a head plus
        modifiers (girl = woman + young, puppy = dog + young, from its gloss), when
        `head` (a token) carries those modifiers among `adjs` (plain adjective
        tokens).  -> (word, the modifier tokens it uses) or None."""
        if getattr(self, "_compounds", None) is None:
            comp = {}
            for lemma, senses in self.index.get("n", {}).items():
                common = [s for s in senses if s in self.senses_ and not self.is_proper(self.senses_[s]["tokens"][0])]
                if not common or not lemma.isalpha():
                    continue
                toks = self.senses_[common[0]]["tokens"]
                if len(toks) > 1 and WORD_RE.match(toks[0]) and toks[0][0] == "n" \
                        and all(re.fullmatch(r"a\d+", t) for t in toks[1:]):
                    comp.setdefault(toks[0], []).append((frozenset(toks[1:]), lemma))
            self._compounds = comp
        have = set(adjs)
        fits = [(mods, w) for mods, w in self._compounds.get(head, []) if mods <= have]
        if not fits:
            return None
        freq = (lambda w: self.lemma_freq(w, "n")) if self.word_freq is not None else (lambda w: 0)
        # the most modifiers it accounts for, then the word English uses most
        mods, word = max(fits, key=lambda f: (len(f[0]), freq(f[1]), -len(f[1])))
        return word, mods

    @property
    def word_freq(self):
        """English word form -> uses per million (ugl/dictionary/wordfreq.json.gz,
        tools/build_word_freq.py), or None if the table is not there."""
        if not hasattr(self, "_word_freq"):
            path = os.path.join(DICT_DIR, "wordfreq.json.gz")
            self._word_freq = None
            if os.path.exists(path):
                with gzip.open(path, "rt", encoding="utf-8") as f:
                    self._word_freq = json.load(f)["freq"]
        return self._word_freq

    @lru_cache(maxsize=None)
    def lemma_freq(self, lemma, cls):
        """Uses per million of a lemma in all its forms (dog + dogs; go, went, gone ...).
        The counts are of word forms, so a form that is also a form of another
        word of the same class says nothing about this one: "found" is mostly
        the past of "find", so only "founded", "founding", "founds" count for
        the verb "found"."""
        from lemminflect import getAllInflections, getAllLemmas
        upos = {"n": "NOUN", "v": "VERB", "a": "ADJ", "r": "ADV"}.get(cls)
        forms = {lemma.lower()}
        if upos:
            for fs in getAllInflections(lemma, upos=upos).values():
                forms.update(f.lower() for f in fs)
        total = 0.0
        for f in forms:
            others = {l.lower() for l in getAllLemmas(f, upos=upos).get(upos, ())} - {lemma.lower()} if upos else set()
            if not others:
                total += self.word_freq.get(f, 0.0)
        return total

    def is_person_token(self, token):
        """Does this concept token name a kind of person ("woman", "singer")?"""
        e = self.entry(token)
        return bool(e) and e.get("sense") in self.person

    def entry(self, token):
        m = WORD_RE.match(token)
        if not m:
            return None
        cls = "v" if m.group(1) == "nv" else m.group(1)
        return self.ids.get(f"{cls}{m.group(2)}")

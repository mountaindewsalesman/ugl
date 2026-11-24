"""English -> UGL with UGL's own parser (pure numpy; runs in Pyodide too).

    text --tokenize--> words --parser--> UD tree --analyze--> Event graph
         --coref (whole document)--> --simplify--> one or more simpler Events
         --serialize--> UGL lines
"""
import re

from .analyze import Analyzer
from .coref import resolve
from .parser.model import ParserModel, DEFAULT_PATH
from .parser.tokenize import split_sentences, tokenize
from .serialize import Serializer
from .simplify import split

_ASIDE = re.compile(r"\s*[\(\[][^\(\)\[\]]*[\)\]]")


def strip_asides(text):
    """Drop (parenthetical) and [bracketed] asides, innermost first."""
    prev = None
    while prev != text:
        prev, text = text, _ASIDE.sub("", text)
    return text


# a word whose head the parser gives less than this probability is flagged
UNSURE = 0.6


def name_internal(t, tree):
    """A link inside a multiword name ("Biju Janata Dal"): UGL writes the name
    as one unit, so how its words attach to each other cannot change the output."""
    h = tree[t["head"] - 1] if t["head"] > 0 else None
    return (h is not None and t["upos"] == "PROPN" and h["upos"] == "PROPN"
            and t["deprel"] in ("flat", "compound", "flat:name"))


class Line(tuple):
    """One UGL line: (ugl, source English sentence, issues).  issues lists what
    may be wrong with it; empty = nothing the encoder knows of."""
    __slots__ = ()

    def __new__(cls, ugl, source, issues):
        return tuple.__new__(cls, (ugl, source, tuple(issues)))

    ugl = property(lambda s: s[0])
    source = property(lambda s: s[1])
    issues = property(lambda s: s[2])


class Encoder:
    def __init__(self, lexicon, parser=None, strip_parens=True, simplify=True):
        self.lex = lexicon
        self.parser = parser or ParserModel(DEFAULT_PATH)
        self.analyzer = Analyzer(lexicon)
        self.serializer = Serializer(lexicon)
        self.strip_parens = strip_parens
        self.simplify = simplify

    @property
    def unknown(self):
        return self.serializer.unknown

    def parse(self, text):
        """-> list of (words, UD token dicts) per sentence."""
        if self.strip_parens:
            text = strip_asides(text)
        out = []
        for s in split_sentences(text):
            words = tokenize(s)
            if any(w[:1].isalnum() for w in words):
                out.append((words, self.parser.parse(words), s.rstrip().endswith("?"), s.strip()))
        return out

    def encode(self, text, keep_memory=False, with_sources=False):
        lines = self.encode_lines(text)
        if with_sources:
            return [(l.ugl, l.source) for l in lines]
        return [l.ugl for l in lines]

    def encode_lines(self, text):
        """-> list of Line(ugl, source, issues).  For training data, keep only
        lines with no issues (see tools/build_corpus.py)."""
        parsed = self.parse(text)
        events, issues = [], []
        for words, tree, question, _ in parsed:
            found = [f"uncertain grammar at '{t['form']}' ({t['conf']:.0%} sure of its head)"
                     for t in tree if t.get("conf", 1.0) < UNSURE and t["upos"] != "PUNCT"
                     and not name_internal(t, tree)]
            try:
                events.append(self.analyzer.sentence(tree, question=question))
                found += self.analyzer.issues
            except Exception as e:          # keep going; report the sentence as unparsed
                events.append(e)
                found.append(f"analyzer failed: {type(e).__name__}")
            issues.append(found)
        unresolved = []
        resolve([e if not isinstance(e, Exception) else None for e in events], unresolved)
        for sno, w in unresolved:
            issues[sno].append(f"'{w}' refers to nothing earlier; written as a generic someone / thing")
        out = []
        for (words, _, _, src), ev, found in zip(parsed, events, issues):
            if ev is None or isinstance(ev, Exception):
                out.append(Line("n?[unparsed]", src, found))
                continue
            for part in (split(ev) if self.simplify else [ev]):
                before = set(self.serializer.unknown)
                try:
                    u = self.serializer.sentence(part, words)
                except Exception as e:
                    out.append(Line("n?[unserializable]", src, found + [f"serializer failed: {type(e).__name__}"]))
                    continue
                new = sorted(w for w, _ in set(self.serializer.unknown) - before)
                unk = [f"'{w}' is not in the dictionary; kept as a literal" for w in new]
                out.append(Line(u, src, found + unk))
        return out

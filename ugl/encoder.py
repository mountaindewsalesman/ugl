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
from .coverage import FOLDED, missing, positions
from .decoder import Parser as UGLParser

_ASIDE = re.compile(r"\s*[\(\[][^\(\)\[\]]*[\)\]]")


def strip_asides(text):
    """Drop (parenthetical) and [bracketed] asides, innermost first."""
    prev = None
    while prev != text:
        prev, text = text, _ASIDE.sub("", text)
    return text


# a word whose head the parser gives less than this (calibrated) probability is
# flagged.  On held-out EWT (tools/calibrate_parser.py): at 0.8, 3.4% of unflagged
# sentences still hold a wrong head; at 0.9, 2.1% (build_corpus.py uses 0.9).
UNSURE = 0.8


def name_internal(t, tree):
    """A link inside a multiword name ("Biju Janata Dal"): UGL writes the name
    as one unit, so how its words attach to each other cannot change the output."""
    h = tree[t["head"] - 1] if t["head"] > 0 else None
    return (h is not None and t["upos"] == "PROPN" and h["upos"] == "PROPN"
            and t["deprel"] in ("flat", "compound", "flat:name"))


class Line(tuple):
    """One UGL line: (ugl, source English sentence, issues, spans).  issues
    lists what may be wrong with it (empty = nothing the encoder knows of);
    spans are the (start, end) character ranges of the source this line says."""
    __slots__ = ()

    def __new__(cls, ugl, source, issues, spans=()):
        return tuple.__new__(cls, (ugl, source, tuple(issues), tuple(spans)))

    ugl = property(lambda s: s[0])
    source = property(lambda s: s[1])
    issues = property(lambda s: s[2])
    spans = property(lambda s: s[3])


def char_offsets(words, src):
    """(start, end) of each token in the sentence text (None if not found)."""
    out, cur = [], 0
    for w in words:
        i = src.find(w, cur)
        if i < 0:
            out.append(None)
            continue
        out.append((i, i + len(w)))
        cur = i + len(w)
    return out


def source_spans(part, toks, offsets, src):
    """Character ranges of the English that one simplified part expresses:
    its words plus the function words folded into them (the, of, will, not...),
    merged where only spaces or punctuation separate them."""
    idx = positions(part)
    for t in toks:
        if t.i in idx:
            idx |= {k.i for k in t.kids if k.deprel in FOLDED and k.deprel not in ("punct", "cc")}
    ranges = sorted(offsets[i] for i in idx if i < len(offsets) and offsets[i])
    merged = []
    for a, b in ranges:
        if merged and src[merged[-1][1]:a].strip(" ,;:'\"-") == "":
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


class Encoder:
    def __init__(self, lexicon, parser=None, strip_parens=True, simplify=True, unsure=UNSURE):
        self.lex = lexicon
        self.parser = parser or ParserModel(DEFAULT_PATH)
        self.analyzer = Analyzer(lexicon)
        self.serializer = Serializer(lexicon)
        self.strip_parens = strip_parens
        self.simplify = simplify
        self.unsure = unsure

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

    def encode_lines(self, text, topic=None):
        """-> list of Line(ugl, source, issues).  For training data, keep only
        lines with no issues (see tools/build_corpus.py)."""
        parsed = self.parse(text)
        events, issues, toks = [], [], []
        for words, tree, question, _ in parsed:
            found = [f"uncertain grammar at '{t['form']}' ({t['conf']:.0%} sure of its head)"
                     for t in tree if t.get("conf", 1.0) < self.unsure and t["upos"] != "PUNCT"
                     and not name_internal(t, tree)]
            try:
                ev = self.analyzer.sentence(tree, question=question)
                events.append(ev)
                toks.append(self.analyzer.toks)
                found += self.analyzer.issues
                found += [f"'{w}' was dropped: it is not in the UGL" for w in missing(self.analyzer.toks, ev)]
            except Exception as e:          # keep going; report the sentence as unparsed
                events.append(e)
                toks.append(None)
                found.append(f"analyzer failed: {type(e).__name__}")
            issues.append(found)
        unresolved = []
        resolve([e if not isinstance(e, Exception) else None for e in events], unresolved, topic)
        for sno, w in unresolved:
            issues[sno].append(f"'{w}' refers to nothing earlier; written as a generic someone / thing")
        out = []
        for (words, _, _, src), ev, found, tk in zip(parsed, events, issues, toks):
            if ev is None or isinstance(ev, Exception):
                out.append(Line("n?[unparsed]", src, found))
                continue
            offsets = char_offsets(words, src)
            parts = split(ev) if self.simplify else [ev]
            for part in parts:
                spans = source_spans(part, tk, offsets, src) if len(parts) > 1 else [(0, len(src))]
                before = set(self.serializer.unknown)
                try:
                    u = self.serializer.sentence(part, words)
                except Exception as e:
                    out.append(Line("n?[unserializable]", src, found + [f"serializer failed: {type(e).__name__}"]))
                    continue
                new = sorted(w for w, _ in set(self.serializer.unknown) - before)
                unk = [f"'{w}' is not in the dictionary; kept as a literal" for w in new]
                try:                                     # every line written must parse back
                    UGLParser(u.split()).parse()
                except Exception as e:
                    unk.append(f"the UGL line does not parse back ({type(e).__name__}: {e})")
                out.append(Line(u, src, found + unk, spans))
        return out

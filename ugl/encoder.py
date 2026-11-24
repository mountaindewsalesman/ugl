"""English -> UGL with UGL's own parser (pure numpy; runs in Pyodide too).

    text --tokenize--> words --parser--> UD tree --analyze--> Event graph
         --coref (whole document)--> --serialize--> UGL lines
"""
import re

from .analyze import Analyzer
from .coref import resolve
from .parser.model import ParserModel, DEFAULT_PATH
from .parser.tokenize import split_sentences, tokenize
from .serialize import Serializer

_ASIDE = re.compile(r"\s*[\(\[][^\(\)\[\]]*[\)\]]")


def strip_asides(text):
    """Drop (parenthetical) and [bracketed] asides, innermost first."""
    prev = None
    while prev != text:
        prev, text = text, _ASIDE.sub("", text)
    return text


class Encoder:
    def __init__(self, lexicon, parser=None, strip_parens=True):
        self.lex = lexicon
        self.parser = parser or ParserModel(DEFAULT_PATH)
        self.analyzer = Analyzer(lexicon)
        self.serializer = Serializer(lexicon)
        self.strip_parens = strip_parens

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
                out.append((words, self.parser.parse(words), s.rstrip().endswith("?")))
        return out

    def encode(self, text, keep_memory=False):
        parsed = self.parse(text)
        events = []
        for words, tree, question in parsed:
            try:
                events.append(self.analyzer.sentence(tree, question=question))
            except Exception as e:          # keep going; report the sentence as unparsed
                events.append(e)
        resolve([e if not isinstance(e, Exception) else None for e in events])
        out = []
        for (words, _, _), ev in zip(parsed, events):
            if ev is None or isinstance(ev, Exception):
                out.append("n?[unparsed]")
                continue
            try:
                out.append(self.serializer.sentence(ev, words))
            except Exception:
                out.append("n?[unserializable]")
        return out

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
        # separate instances for meaning checks, so checking never disturbs encoding
        self._check_an, self._check_se = Analyzer(lexicon), Serializer(lexicon)
        self.strip_parens = strip_parens
        self.simplify = simplify
        self.unsure = unsure
        self.second_parser = None        # optional: a second parser for ensemble disagreement checks
        # optional progress(done, total) callback: each sentence counts twice,
        # once when parsed and once when analyzed (the playground's progress bar)
        self.progress = None

    @property
    def unknown(self):
        return self.serializer.unknown

    def parse(self, text):
        """-> list of (words, UD token dicts, question?, source text, parser state) per sentence."""
        if self.strip_parens:
            text = strip_asides(text)
        out = []
        sents = split_sentences(text)
        for k, s in enumerate(sents):
            words = tokenize(s)
            if any(w[:1].isalnum() for w in words):
                state = self.parser.score(words)
                tree = self.parser.tree(state, self.parser.decode_tree(state["S"]))
                out.append((words, tree, s.rstrip().endswith("?"), s.strip(), state))
            if self.progress:
                self.progress(k + 1, 2 * len(sents))
        return out

    def meaning(self, tree, words, question):
        """The UGL a tree gives on its own (no cross-sentence coreference): what
        the uncertainty check compares."""
        try:
            ev = self._check_an.sentence(tree, question=question)
            resolve([ev])
            return "\n".join(self._check_se.sentence(p, words) for p in split(ev))
        except Exception as e:
            return f"<failed: {type(e).__name__}>"

    def uncertain(self, words, tree, question, state):
        """Words whose head the parser is unsure of AND whose next-best
        attachment would change the UGL.  Uncertainty that cannot change the
        meaning ("the" attached to either of two nouns that end up the same)
        is not an issue."""
        out, base = [], None
        for i, t in enumerate(tree):
            if t.get("conf", 1.0) >= self.unsure or t["upos"] == "PUNCT" or name_internal(t, tree):
                continue
            base = base if base is not None else self.meaning(tree, words, question)
            alt = self.parser.alternative(state, tree, i)
            if self.meaning(alt, words, question) != base:
                out.append(f"uncertain grammar at '{t['form']}' ({t['conf']:.0%} sure of its head; "
                           f"the other reading means something else)")
        if self.second_parser is not None:
            # ensemble check: an independently trained parser reads the sentence;
            # if its tree gives different UGL, the reading is in doubt
            other = self.second_parser.parse(words)
            base = base if base is not None else self.meaning(tree, words, question)
            if self.meaning(other, words, question) != base:
                diff = [t["form"] for t, o in zip(tree, other)
                        if (t["head"], t["deprel"]) != (o["head"], o["deprel"]) and t["upos"] != "PUNCT"]
                out.append("the two parsers disagree in a way that changes the UGL (at "
                           + ", ".join(f"'{w}'" for w in diff[:4]) + ")")
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
        for k, (words, tree, question, _, state) in enumerate(parsed):
            if self.progress:
                self.progress(len(parsed) + k, 2 * len(parsed))
            found = self.uncertain(words, tree, question, state)
            if not any(t["upos"] in ("VERB", "AUX") for t in tree):
                # "Pathétique Sonata – Beethoven", "Lift Me Up - 5:52": the UGL would
                # invent an "is"; such lines are not sentences to learn from
                found.append("no verb: a title, caption or list item, not a sentence")
            elif any(t["head"] == 0 and t["upos"] == "AUX" for t in tree):
                # "It will not.", "I can.", "Yes, it is.": the helper stands for a verb
                # said before; UGL would make the helper itself the event
                found.append("ellipsis: a helper verb stands for a verb left out, not a sentence on its own")
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
        for (words, _, _, src, _), ev, found, tk in zip(parsed, events, issues, toks):
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

"""UGL - a universal grammatical language: English <-> numbered-concept notation.

    import ugl
    ugl.encode("At nine, I ate the apple.")   # ['#9 v20 n1 v444p n2344t \\v20']
    ugl.decode("#9 v20 n1 v444p n2344t \\v20") # 'I ate the apple at 9.'
    ids = ugl.to_ids("Every dog barks.")       # integer ids for model training
    ugl.from_ids(ids)                          # back to UGL text

    u = ugl.load("ugl-dict-0.1.0.json.gz")     # pin an exact dictionary version
"""
__version__ = "0.3.0"

from .lexicon import Lexicon, available
from .decoder import Decoder
from .tokenizer import Tokenizer


class UGL:
    """A dictionary plus the encoder, decoder and tokenizer built on it.
    The parser is only loaded the first time you encode English."""

    def __init__(self, dictionary=None, wsd=True, engine="ugl", simplify=True):
        """wsd: True = contextual sense model if installed, False = most
        frequent sense, or a path to a centroid file.
        engine: "ugl" (own numpy parser, default) or "spacy" (legacy).
        simplify: split long sentences into simpler ones with the same meaning
        (top-level "and", comma-set-off clauses); see ugl/simplify.py."""
        self.engine = engine
        self.simplify = simplify
        self.wsd = None
        self.lexicon = Lexicon(dictionary)
        self.decoder = Decoder(self.lexicon)
        self.tokenizer = Tokenizer(self.lexicon)
        from .humanize import Humanizer
        self.humanizer = Humanizer(self.lexicon, self.decoder)
        self._encoder = None
        self._want_wsd = wsd

    @property
    def encoder(self):
        if self._encoder is None:
            if self.engine == "spacy":
                from .encoder_spacy import SpacyEncoder
                self._encoder = SpacyEncoder(self.lexicon)
            else:
                from .encoder import Encoder
                self._encoder = Encoder(self.lexicon, simplify=self.simplify)
            if self._want_wsd:
                self.wsd = _attach_context_wsd(self.lexicon, self._want_wsd,
                                               getattr(self._encoder, "parser", None))
        return self._encoder

    @property
    def version(self):
        return self.lexicon.version

    def encode(self, text, keep_memory=False):
        """English text -> list of UGL sentences (a long English sentence
        may become several simpler ones; see simplify)."""
        return self.encoder.encode(text, keep_memory=keep_memory)

    def encode_lines(self, text, topic=None, roundtrip=False):
        """-> list of Line(ugl, source, issues, spans): each UGL line with the
        English sentence it came from and anything the encoder is unsure of.
        topic: what the text is about (e.g. an article title), for pronouns.
        roundtrip: also decode each sentence's UGL to English, encode that
        again, and flag the sentence if the UGL changes (costs a second parse)."""
        lines = self.encoder.encode_lines(text, topic)
        if roundtrip:
            lines = self._roundtrip(lines)
        return lines

    def _roundtrip(self, lines):
        from .encoder import Line
        groups = {}
        for l in lines:
            groups.setdefault(l.source, []).append(l)
        bad = {}
        for src, ls in groups.items():
            ugl = [l.ugl for l in ls]
            try:
                english = self.decode("\n".join(ugl))
                again = [l.ugl for l in self.encoder.encode_lines(english)]
            except Exception as e:
                bad[src] = f"round trip failed: {type(e).__name__}"
                continue
            if again != ugl:
                bad[src] = f"round trip changed the UGL (decoded as: {english})"
        return [Line(l.ugl, l.source, l.issues + ((bad[l.source],) if l.source in bad else ()), l.spans)
                for l in lines]

    def encode_with_sources(self, text):
        """-> [(UGL line, the English sentence it came from)]; a long sentence
        split by simplify gives several lines with the same source."""
        return self.encoder.encode(text, with_sources=True)

    def decode(self, ugl_text):
        """UGL (one sentence per line) -> English."""
        return self.decoder.decode(ugl_text)

    def humanize(self, ugl_text):
        """UGL -> the same tokens as English words, e.g. 'someone has three apples \\has'."""
        return self.humanizer(ugl_text)

    def to_ids(self, text, bos=True, eos=True):
        """English text -> token ids."""
        return self.tokenizer.encode("\n".join(self.encode(text)), bos=bos, eos=eos)

    def from_ids(self, ids):
        """token ids -> UGL text."""
        return self.tokenizer.decode(ids)


_default = None


def _attach_context_wsd(lexicon, want, parser=None):
    """wsd=True: the parser's own sense head (ugl/wsd_gloss.py: the word in
    context against each sense's gloss, trained jointly with the parser; numpy,
    identical on desktop and in the browser; 73.5% on held-out SemCor vs 66.8%
    most-frequent-sense).  wsd="path.npz": a centroid file instead, e.g. BERT-base
    (needs torch).  Without either, most-frequent-sense (always works)."""
    import os
    if want is True and parser is not None and "wsd.mix" in getattr(parser, "w", {}):
        from .wsd_gloss import GlossWSD
        lexicon.wsd = GlossWSD(lexicon, parser)
        return lexicon.wsd
    from .wsd_context import DEFAULT_PATH
    path = want if isinstance(want, str) else DEFAULT_PATH
    if not os.path.exists(path):
        return None
    try:
        from .wsd_context import attach
        return attach(lexicon, path, parser=parser)
    except ImportError:
        return None


def load(dictionary=None, wsd=True, engine="ugl"):
    return UGL(dictionary, wsd, engine)


def default():
    global _default
    if _default is None:
        _default = UGL()
    return _default


def encode(text, keep_memory=False):
    return default().encode(text, keep_memory)


def decode(ugl_text):
    return default().decode(ugl_text)


def humanize(ugl_text):
    return default().humanize(ugl_text)


def to_ids(text, bos=True, eos=True):
    return default().to_ids(text, bos, eos)


def from_ids(ids):
    return default().from_ids(ids)

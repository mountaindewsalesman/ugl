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

    def encode_lines(self, text):
        """-> list of Line(ugl, source, issues): each UGL line with the English
        sentence it came from and anything the encoder is unsure of."""
        return self.encoder.encode_lines(text)

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
    """wsd=True: the sense centroids built on the parser's own transformer
    (numpy; identical on desktop and in the browser).  wsd="path.npz": another
    centroid file, e.g. BERT-base ones (needs torch).  If the file or its
    model is missing, most-frequent-sense is kept (always works)."""
    import os
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

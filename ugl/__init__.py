"""UGL - a universal grammatical language: English <-> numbered-concept notation.

    import ugl
    ugl.encode("At nine, I ate the apple.")   # ['#9 v20 n1 v444p n2344t \\v20']
    ugl.decode("#9 v20 n1 v444p n2344t \\v20") # 'I ate the apple at 9.'
    ids = ugl.to_ids("Every dog barks.")       # integer ids for model training
    ugl.from_ids(ids)                          # back to UGL text

    u = ugl.load("ugl-dict-0.1.0.json.gz")     # pin an exact dictionary version
"""
__version__ = "0.1.0"

from .lexicon import Lexicon, available
from .decoder import Decoder
from .tokenizer import Tokenizer


class UGL:
    """A dictionary plus the encoder, decoder and tokenizer built on it.
    The spaCy model is only loaded the first time you encode English."""

    def __init__(self, dictionary=None, wsd=True):
        self.lexicon = Lexicon(dictionary)
        self.decoder = Decoder(self.lexicon)
        self.tokenizer = Tokenizer(self.lexicon)
        self._encoder = None
        self._want_wsd = wsd

    @property
    def encoder(self):
        if self._encoder is None:
            from .encoder import Encoder
            self._encoder = Encoder(self.lexicon)
            if self._want_wsd:
                from . import wsd
                wsd.attach(self.lexicon, self._encoder.nlp)
        return self._encoder

    @property
    def version(self):
        return self.lexicon.version

    def encode(self, text, keep_memory=False):
        """English text -> list of UGL sentences."""
        return self.encoder.encode(text, keep_memory=keep_memory)

    def decode(self, ugl_text):
        """UGL (one sentence per line) -> English."""
        return self.decoder.decode(ugl_text)

    def to_ids(self, text, bos=True, eos=True):
        """English text -> token ids."""
        return self.tokenizer.encode("\n".join(self.encode(text)), bos=bos, eos=eos)

    def from_ids(self, ids):
        """token ids -> UGL text."""
        return self.tokenizer.decode(ids)


_default = None


def load(dictionary=None, wsd=True):
    return UGL(dictionary, wsd)


def default():
    global _default
    if _default is None:
        _default = UGL()
    return _default


def encode(text, keep_memory=False):
    return default().encode(text, keep_memory)


def decode(ugl_text):
    return default().decode(ugl_text)


def to_ids(text, bos=True, eos=True):
    return default().to_ids(text, bos, eos)


def from_ids(ids):
    return default().from_ids(ids)

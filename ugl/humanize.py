"""Readable view of UGL: the same tokens in the same order, each shown as an
English word with its modifiers applied.

    n4 v2 #3 n2344s #1 n5120 \\v2   ->   someone has three apples one pear \\has

Unlike decoding, nothing is reordered or dropped, so the structure (scopes,
relation order, repeated nouns) stays visible.  It reuses the decoder's
inflection so both agree on word forms.
"""
from .core import RESERVED_NOUNS, RESERVED_VERBS, WORD_RE
from .decoder import PRON, SPECIAL, Decoder, Tok, inflect, parse_tok

NUMBER_WORDS = ("zero one two three four five six seven eight nine ten eleven twelve thirteen "
                "fourteen fifteen sixteen seventeen eighteen nineteen twenty").split()


class Humanizer:
    def __init__(self, lexicon, decoder=None):
        self.lex = lexicon
        self.dec = decoder or Decoder(lexicon)

    def line(self, ugl_line):
        out, after_verb = [], False
        for t in ugl_line.split():
            out.append(self.token(t, obj=after_verb))
            m = WORD_RE.match(t)
            if m and m.group(1) == "v":
                # a real verb's next noun is its object ("saw me"); after a
                # relation / logic / quantifier word a new clause starts
                i = int(m.group(2))
                after_verb = i not in RESERVED_VERBS or RESERVED_VERBS[i][1] == "core"
            elif not (m and m.group(1) == "a"):
                after_verb = False
        return " ".join(out)

    def __call__(self, text):
        return "\n".join(self.line(l) for l in text.strip().split("\n") if l.strip())

    # ------------------------------------------------------------------ tokens
    def token(self, s, obj=False):
        if s.startswith("\\"):
            idx = s[2:]
            return "\\" + (self.verb(Tok(s, "word", "v", int(idx), "")) if idx.isdigit() else idx)
        try:
            t = parse_tok(s)
        except Exception:
            return s
        if t.kind == "num":
            n = t.raw[1:]
            return NUMBER_WORDS[int(n)] if n.isdigit() and int(n) < len(NUMBER_WORDS) else n.replace("_", " ")
        if t.kind == "name":
            name = self.dec.lemma(t)
            return name + ("'s" if "y" in t.mods else "")
        if t.kind == "unk":
            return "?" + t.raw[2:].replace("_", " ")
        if t.cls == "a":
            return self.dec.adj(t)
        if t.cls == "v":
            return self.verb(t)
        return self.noun(t, obj)

    def noun(self, t, obj=False):
        m = t.mods
        if t.cls == "n" and t.idx in RESERVED_NOUNS:
            key = t.base + ("s" if "s" in m else "")
            if key in PRON and not set(m) & set("xq"):
                word = PRON[key][2] if "y" in m else PRON[key][1 if obj else 0]
                return word
            sel = "".join(c for c in m if c in "qxdets")
            word = (SPECIAL.get((t.base, sel)) or SPECIAL.get((t.base, sel.replace("s", "")))
                    or RESERVED_NOUNS[t.idx])
            return word + ("'s" if "y" in m else "")
        word = self.dec.lemma(t)
        if t.cls == "nv":
            word = inflect(word, "VBG")
        elif "s" in m:
            word = inflect(word, "NNS")
        det = ("which " if "q" in m else "no " if "x" in m else
               ("these " if "s" in m else "this ") if "d" in m else
               ("those " if "s" in m else "that ") if "e" in m else
               "the " if "t" in m else "")
        if "y" in m:
            word += "'" if word.endswith("s") and "s" in m else "'s"
        return det + word

    def verb(self, t):
        if t.kind == "word" and t.idx in RESERVED_VERBS and RESERVED_VERBS[t.idx][1] not in ("core",):
            word = RESERVED_VERBS[t.idx][0]
            return ("not " + word) if "x" in t.mods else word
        before, after = self.dec.verb_group(t, None)
        word = " ".join(before + after)
        if "q" in t.mods:
            word += "?"
        if "o" in t.mods:
            word += "!"
        return word

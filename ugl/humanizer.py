"""Readable view of UGL: the same tokens in the same order, each shown as an
English word with its modifiers applied.

    n4 v2 n2344s #3        ->   someone has apples three
    #9 v20 n1 v444p n2344t \\v20   ->   nine time I ate the apple \\time

Unlike decoding, nothing is reordered or dropped, so the structure (scopes,
relation order, repeated nouns) stays visible.  It reuses the decoder's
inflection so both agree on word forms.
"""
from .core import RESERVED_NOUNS, RESERVED_VERBS, WORD_RE, number_text, parse_literal
from .decoder import NP, PRON, SPECIAL, Decoder, Tok, inflect, parse_tok, pron_key

NUMBER_WORDS = ("zero one two three four five six seven eight nine ten eleven twelve thirteen "
                "fourteen fifteen sixteen seventeen eighteen nineteen twenty").split()


class Humanizer:
    def __init__(self, lexicon, decoder=None):
        self.lex = lexicon
        self.dec = decoder or Decoder(lexicon)

    def line(self, ugl_line):
        return " ".join(self.words(ugl_line))

    def tree(self, ugl_line):
        """The readable view as an indented tree: every word that opens a scope
        (a relation, logic word, quantifier, "that", or a verb with clause
        complements) starts a block, and its closing tag(s) end it.
        -> list of {"depth", "text" (readable), "ugl"} lines in reading order.
        Which tag closes which word comes from the decoder's own parse."""
        from .decoder import Parser
        toks = ugl_line.split()
        words = self.words(ugl_line)
        p = Parser(toks)
        try:
            p.parse()
        except Exception:
            return [{"depth": 0, "text": " ".join(words), "ugl": ugl_line}]
        closers = {}
        for c, o in p.opener.items():
            closers.setdefault(o, []).append(c)
        last = {max(cs) for cs in closers.values()}
        out, cur, depth = [], [], 0

        def flush():
            if cur:
                out.append({"depth": depth, "text": " ".join(words[k] for k in cur),
                            "ugl": " ".join(toks[k] for k in cur)})
                cur.clear()
        for k in range(len(toks)):
            if k in p.opener:                       # a closing tag: own line, one level out
                flush()
                depth -= 1
                cur.append(k)
                flush()
                if k not in last:                   # clause form "R A close B close": B follows
                    depth += 1
                continue
            cur.append(k)
            if k in closers:                        # opens a block
                flush()
                depth += 1
        flush()
        return out

    def words(self, ugl_line):
        out, after_verb, subj = [], False, None
        for t in ugl_line.split():
            out.append(self.token(t, obj=after_verb, subj=subj))
            try:
                x = parse_tok(t) if not t.startswith("\\") else None
            except Exception:
                x = None
            if x is not None and (x.cls in ("n", "nv") or x.kind in ("name", "num")) and "y" not in x.mods:
                subj = NP(x)                       # the subject a following verb agrees with
            m = WORD_RE.match(t)
            if m and m.group(1) == "v":
                # a real verb's next noun is its object ("saw me"); after a
                # relation / logic / quantifier word a new clause starts
                i = int(m.group(2))
                after_verb = i not in RESERVED_VERBS or RESERVED_VERBS[i][1] == "core"
            elif not (m and m.group(1) == "a"):
                after_verb = False
        return out

    def __call__(self, text):
        lines = [l for l in text.strip().split("\n") if l.strip()]
        if "@" in text:                       # back-references read as what they point to
            from .refs import expand
            lines = expand(lines, self.lex.is_proper)
        return "\n".join(self.line(l) for l in lines)

    # ------------------------------------------------------------------ tokens
    def token(self, s, obj=False, subj=None):
        if s.startswith("\\"):
            idx = s[2:]
            return "\\" + (self.verb(Tok(s, "word", "v", int(idx), "")) if idx.isdigit() else idx)
        try:
            t = parse_tok(s)
        except Exception:
            return s
        if t.kind == "num":
            n = t.raw[1:]
            return NUMBER_WORDS[int(n)] if n.isdigit() and int(n) < len(NUMBER_WORDS) else number_text(t.raw)
        if t.kind == "name":
            name = self.dec.lemma(t)
            return name + ("'s" if "y" in t.mods else "")
        if t.kind == "unk":
            return "?" + parse_literal(t.raw)[2]
        if t.cls == "a":
            return self.dec.adj(t)
        if t.cls == "v":
            return self.verb(t, subj)
        return self.noun(t, obj)

    def noun(self, t, obj=False):
        m = t.mods
        if t.cls == "n" and t.idx in RESERVED_NOUNS:
            key = pron_key(t.base, m)
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

    def verb(self, t, subj=None):
        if t.kind == "word" and t.idx in RESERVED_VERBS and RESERVED_VERBS[t.idx][1] not in ("core",):
            word = RESERVED_VERBS[t.idx][0]
            return ("not " + word) if "x" in t.mods else word
        before, after = self.dec.verb_group(t, subj)
        word = " ".join(before + after)
        if "q" in t.mods:
            word += "?"
        if "o" in t.mods:
            word += "!"
        return word

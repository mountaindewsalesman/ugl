"""UGL -> English.  Deterministic: parse the token string into a tree, then
realise each node with English word order and inflection.  No neural net.

Grammar (whitespace-separated tokens)
  clause  := R clause \\R clause \\R           R in relation/logic verbs (clause form)
           | np R clause \\R                  R in relation/quantifier verbs (NP form)
           | np V adj* [np] (clause \\V)*      core clause
           | np                              bare noun phrase ("there is X")
  np      := head mod*
  head    := n# | nv# | #num | 'Name' | n?word
  mod     := a#           adjective
           | n#j | 'X'j   noun used as attribute
           | #num         count
           | v3 clause \\v3   relative clause (head noun repeated inside)
           | np-with-y    possessor (always last; its own mods follow it)
"""
import re
from dataclasses import dataclass, field
from typing import List, Optional

from .core import (parse_literal, WORD_RE, RESERVED_NOUNS, RESERVED_VERBS, VERB_KIND, WRAPPER_VERBS,
                   CLAUSE_FORM_VERBS, V, tokenize)

try:
    from lemminflect import getInflection
except ImportError:  # pragma: no cover
    getInflection = None


class ParseError(Exception):
    pass


# ------------------------------------------------------------------ tree
@dataclass
class Tok:
    raw: str
    kind: str            # word | num | name | unk
    cls: str = ""        # n v a nv
    idx: int = 0
    mods: str = ""

    @property
    def base(self):
        if self.kind == "word":
            return f"{'v' if self.cls == 'nv' else self.cls}{self.idx}"
        if self.kind == "name":
            return self.raw.rsplit("'", 1)[0] + "'"
        return self.raw


@dataclass
class NP:
    head: Tok
    adjs: List[Tok] = field(default_factory=list)
    comps: List["NP"] = field(default_factory=list)
    nums: List[Tok] = field(default_factory=list)
    rels: List[object] = field(default_factory=list)
    poss: Optional["NP"] = None


@dataclass
class Core:
    subj: NP
    verb: Tok
    advs: List[Tok]
    obj: Optional[NP]
    comps: List[object]


@dataclass
class Wrap:
    land: NP
    rel: Tok
    body: object


@dataclass
class Logic:
    rel: Tok
    a: object
    b: object


@dataclass
class Bare:
    np: NP


def parse_tok(s):
    m = WORD_RE.match(s)
    if m:
        return Tok(s, "word", m.group(1), int(m.group(2)), m.group(3))
    if s.startswith("#"):
        return Tok(s, "num")
    if s.startswith("'"):
        name, _, mods = s[1:].rpartition("'")
        return Tok(s, "name", "n", 0, mods)
    if len(s) > 2 and s[1] == "?":
        _, cls, _, mods = parse_literal(s)
        return Tok(s, "unk", cls, 0, mods)
    raise ParseError(s)


class Parser:
    def __init__(self, tokens):
        self.t = tokens
        self.n = len(tokens)

    def peek(self, i):
        return self.t[i] if i < self.n else None

    def is_close(self, i):
        return i < self.n and self.t[i].startswith("\\")

    def tokat(self, i):
        return parse_tok(self.t[i]) if i < self.n and not self.is_close(i) else None

    def starts_np(self, i):
        x = self.tokat(i)
        if x is None:
            return False
        return (x.kind == "word" and x.cls in ("n", "nv")) or x.kind in ("num", "name") or (x.kind == "unk" and x.cls == "n")

    def is_verb(self, x):
        return x is not None and ((x.kind == "word" and x.cls == "v") or (x.kind == "unk" and x.cls == "v"))

    def expect_close(self, i, vid):
        if self.peek(i) != f"\\v{vid}":
            raise ParseError(f"expected \\v{vid} at {i}, got {self.peek(i)}")
        return i + 1

    def clause(self, i):
        x = self.tokat(i)
        if x is None:
            raise ParseError(f"clause expected at {i}")
        if x.kind == "word" and x.cls == "v" and x.idx in CLAUSE_FORM_VERBS:
            a, i = self.clause(i + 1)
            i = self.expect_close(i, x.idx)
            b, i = self.clause(i)
            i = self.expect_close(i, x.idx)
            return Logic(x, a, b), i
        np, i = self.np(i)
        v = self.tokat(i)
        if self.is_verb(v):
            if v.kind == "word" and v.idx in WRAPPER_VERBS:
                body, i = self.clause(i + 1)
                i = self.expect_close(i, v.idx)
                return Wrap(np, v, body), i
            i += 1
            advs = []
            while True:
                a = self.tokat(i)
                if a is not None and a.cls == "a":
                    advs.append(a)
                    i += 1
                else:
                    break
            obj, comps = None, []
            vid = v.idx if v.kind == "word" else 0
            if self.starts_np(i) or self.clause_form_start(i):
                save = i
                try:
                    c, j = self.clause(i)
                    if not isinstance(c, Bare) and self.peek(j) == f"\\v{vid}":
                        comps.append(c)
                        i = j + 1
                    else:
                        raise ParseError("not a complement")
                except ParseError:
                    i = save
                    if self.starts_np(i):
                        obj, i = self.np(i)
                        if self.peek(i) == f"\\v{vid}":   # the NP was a bare complement
                            comps.append(Bare(obj))
                            obj, i = None, i + 1
            while self.starts_np(i) or self.clause_form_start(i):
                c, i = self.clause(i)
                i = self.expect_close(i, vid)
                comps.append(c)
            return Core(np, v, advs, obj, comps), i
        return Bare(np), i

    def clause_form_start(self, i):
        x = self.tokat(i)
        return x is not None and x.kind == "word" and x.cls == "v" and x.idx in CLAUSE_FORM_VERBS

    def np(self, i):
        if not self.starts_np(i):
            raise ParseError(f"noun phrase expected at {i}: {self.peek(i)}")
        np = NP(parse_tok(self.t[i]))
        i += 1
        while i < self.n and not self.is_close(i):
            x = parse_tok(self.t[i])
            if x.cls == "a":
                np.adjs.append(x); i += 1
            elif x.kind == "num":
                np.nums.append(x); i += 1
            elif "j" in x.mods and x.cls in ("n", "nv") or (x.kind == "name" and "j" in x.mods):
                np.comps.append(NP(x)); i += 1
            elif x.kind == "word" and x.cls == "v" and x.idx == V["that"]:
                c, i = self.clause(i + 1)
                i = self.expect_close(i, V["that"])
                np.rels.append(c)
            elif "y" in x.mods and x.cls in ("n", "nv"):
                np.poss, i = self.np(i)
                break
            else:
                break
        return np, i

    def parse(self):
        c, i = self.clause(0)
        if i != self.n:
            raise ParseError(f"trailing tokens from {i}: {' '.join(self.t[i:i+5])}")
        return c


# ------------------------------------------------------------- realiser
PRON = {"n1": ("I", "me", "my"), "n1s": ("we", "us", "our"), "n2": ("you", "you", "your"),
        "n2s": ("you", "you", "your"),
        # "the thing(s)" re-mentioned (an unresolved it / they)
        "n3t": ("it", "it", "its"), "n3ts": ("they", "them", "their")}


def pron_key(base, mods):
    """PRON table key for a reserved noun with its modifier letters."""
    return base + ("t" if base == "n3" and "t" in mods else "") + ("s" if "s" in mods else "")
SPECIAL = {  # reserved noun + modifier -> word
    ("n3", "q"): "what", ("n3", "x"): "nothing", ("n3", "d"): "this", ("n3", "e"): "that",
    ("n3", ""): "something", ("n3", "s"): "things", ("n3", "ds"): "these", ("n3", "es"): "those",
    ("n4", "q"): "who", ("n4", "x"): "nobody", ("n4", ""): "someone", ("n4", "s"): "people",
    ("n5", "q"): "where", ("n5", "d"): "here", ("n5", "e"): "there",
    ("n6", "q"): "when", ("n6", "d"): "now", ("n6", "e"): "then",
    ("n7", "q"): "how", ("n8", "q"): "why", ("n9", "q"): "how much",
    ("n3", "t"): "the thing", ("n3", "ts"): "the things",
    ("n4", "t"): "the person", ("n5", "t"): "the place", ("n6", "t"): "the time",
    ("n7", "t"): "the way", ("n8", "t"): "the reason", ("n9", "t"): "the amount",
}
ADVERBIAL = {("n5", "q"), ("n5", "d"), ("n5", "e"), ("n6", "q"), ("n6", "d"), ("n6", "e"),
             ("n7", "q"), ("n8", "q")}
REL_WORD = {V["time"]: "at", V["cause"]: "because of", V["of"]: "of", V["by"]: "by",
            V["in front of"]: "in front of", V["out of"]: "out of"}
QUANT_WORD = {V["all"]: "every", V["most"]: "most", V["many"]: "many", V["few"]: "few"}
RESERVED_NOUN_BASES = {f"n{i}" for i in RESERVED_NOUNS}
VOWEL = re.compile(r"^[aeiou]", re.I)


def inflect(lemma, tag):
    if " " in lemma:
        first, rest = lemma.split(" ", 1)
        return inflect(first, tag) + " " + rest
    if getInflection:
        r = getInflection(lemma, tag=tag)
        if r:
            return r[0]
    if tag in ("VBZ", "NNS"):
        return lemma + ("es" if lemma.endswith(("s", "x", "ch", "sh")) else "s")
    if tag in ("VBD", "VBN"):
        return lemma + ("d" if lemma.endswith("e") else "ed")
    if tag == "VBG":
        return (lemma[:-1] if lemma.endswith("e") else lemma) + "ing"
    return lemma


def adverb_form(adj):
    if adj.endswith("ly") or adj in ("well", "fast", "hard", "late", "early", "often", "soon", "never", "always", "again"):
        return adj
    if adj == "good":
        return "well"
    if adj.endswith("y"):
        return adj[:-1] + "ily"
    if adj.endswith("le"):
        return adj[:-1] + "y"
    if adj.endswith("ue"):                      # true -> truly, due -> duly
        return adj[:-1] + "ly"
    if adj.endswith("ic"):
        return adj + "ally"
    return adj + "ly"


class Ctx:
    def __init__(self):
        self.quant = {}      # base -> quant word (unused yet)
        self.gap = None      # base of relative-clause head
        self.gap_word = "that"


class Decoder:
    def __init__(self, lexicon):
        self.lex = lexicon

    def decode(self, text):
        """Decode one UGL sentence (or several separated by newlines)."""
        out = []
        for line in text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            tree = Parser(tokenize(line)).parse()
            s = self.clause(tree, Ctx())
            s = re.sub(r"\s+", " ", s).strip()
            q = getattr(self, "_q", False)
            self._q = False
            out.append((s[:1].upper() + s[1:]) + ("?" if q else "."))
        return " ".join(out)

    # -------------------------------------------------- words
    def lemma(self, t):
        if t.kind == "name":
            return t.raw.rsplit("'", 1)[0][1:].replace("_", " ")
        if t.kind == "num":
            return t.raw[1:].replace("_", " ")
        if t.kind == "unk":
            return parse_literal(t.raw)[2]
        cls = "v" if t.cls == "nv" else t.cls
        return self.lex.lemma_of(cls, t.idx)

    def is_adverb_sense(self, t):
        """The dictionary meaning is itself an adverb (twice, often): no -ly."""
        if t.kind != "word":
            return False
        e = self.lex.entry(t.raw)
        return bool(e) and ".r." in e.get("sense", "")

    def adj(self, t, adverb=False):
        w = self.lemma(t)
        m = t.mods
        if adverb and not self.is_adverb_sense(t):
            w = adverb_form(w)
        if "r" in m:
            f = inflect(w, "JJR") if not adverb else ""
            w = f if f and f != w and " " not in w else "more " + w
        if "z" in m:
            f = inflect(w, "JJS") if not adverb else ""
            w = f if f and f != w and " " not in w else "most " + w
        if "l" in m:
            w = "less " + w
        if "w" in m:
            w = "very " + w
        if "x" in m:
            w = "not " + w
        return w

    # -------------------------------------------------- noun phrases
    def matches(self, np, base):
        return base is not None and np is not None and np.head.base == base

    def np(self, np, ctx, case="subj"):
        h = np.head
        if h.kind == "word" and h.cls == "n" and h.idx in RESERVED_NOUNS:
            key = pron_key(h.base, h.mods)
            if key in PRON and not ("x" in h.mods or "q" in h.mods) and not np.rels and not np.adjs:
                p = PRON[key]
                return p[0] if case == "subj" else p[1] if case == "obj" else p[2]
            m = "".join(c for c in h.mods if c in "qxdets")
            word = SPECIAL.get((h.base, m)) or SPECIAL.get((h.base, m.replace("s", ""))) or RESERVED_NOUNS[h.idx]
            words = [word] + [self.adj(a) for a in np.adjs]
            for r in np.rels:                       # "someone who can help"
                sub = Ctx()
                sub.gap = h.base
                sub.gap_word = "who" if h.base == "n4" else "that"
                words.append(self.clause(r, sub))
            s = " ".join(w for w in words if w)
            if case == "poss":
                s += "'s"
            return s
        noun = self.lemma(h)
        plural = "s" in h.mods
        if h.kind == "word" and h.cls == "nv":
            noun = inflect(noun, "VBG")
        elif h.kind == "word" and plural:
            noun = inflect(noun, "NNS")
        pre = [self.adj(a) for a in np.adjs] + [self.lemma(c.head) for c in np.comps]
        nums = [self.lemma(n) for n in np.nums]
        det = ""
        of_part = ""
        quant = ctx.quant.pop(h.base, None) if h.base in ctx.quant else None
        if quant:
            if quant == "every" and (plural or np.poss is not None or "d" in h.mods or "e" in h.mods):
                quant = "all"                               # all friends / all my friends / all these
            det = quant
            if np.poss is not None:                         # "all my friends", "most of John's books"
                det += (" " if quant == "all" else " of ") + self.np(np.poss, ctx, "poss")
            elif "d" in h.mods or "e" in h.mods:            # "most of these wars"
                dem = ("these" if plural else "this") if "d" in h.mods else ("those" if plural else "that")
                det += (" " if quant == "all" else " of ") + dem
        elif np.poss is not None and (np.poss.head.kind == "name" or np.poss.head.base in RESERVED_NOUN_BASES
                                      or self.lex.is_proper(np.poss.head.raw)):
            det = self.np(np.poss, ctx, "poss")
        elif np.poss is not None:
            det = "the"
            of_part = "of " + self.np(np.poss, ctx, "obj")
        elif "q" in h.mods:
            det = "which"
        elif "x" in h.mods:
            det = "no"
        elif "d" in h.mods:
            det = "these" if plural else "this"
        elif "e" in h.mods:
            det = "those" if plural else "that"
        elif "t" in h.mods:
            det = "the"
        elif h.kind == "word" and not plural and not nums and not self.lex.is_proper(h.raw):
            first = (pre[0] if pre else noun)
            det = "an" if VOWEL.match(first) else "a"
        words = [det] + nums + pre + [noun, of_part]
        for r in np.rels:
            sub = Ctx()
            sub.gap = h.base
            sub.gap_word = "who" if h.kind == "name" or self.lex.is_proper(h.raw) else "that"
            words.append(self.clause(r, sub))
        s = " ".join(w for w in words if w)
        if case == "poss":
            s += "'" if s.endswith("s") and plural else "'s"
        return s

    # -------------------------------------------------- verbs
    def verb_group(self, v, subj, question=False):
        """Return (before_subject, after_subject) word lists."""
        m = v.mods
        lemma = self.lemma(v) if not (v.kind == "word" and v.idx in RESERVED_VERBS) else RESERVED_VERBS[v.idx][0]
        sbase = subj.head.base if subj else ""
        plural_subj = subj is not None and ("s" in subj.head.mods or sbase == "n2")
        first_person = sbase == "n1" and "s" not in (subj.head.mods if subj else "")
        third_sg = not plural_subj and sbase not in ("n1", "n2")
        past = "p" in m
        neg = "x" in m

        chain = []
        modal = None
        if "c" in m:
            modal = "could" if (past or "h" in m) else "can"
        elif "m" in m:
            modal = "might" if "h" in m else "may"
        elif "u" in m:
            modal = "should" if "h" in m else "must"
        elif "f" in m:
            modal = "would" if (past or "h" in m) else "will"
        elif "h" in m:
            modal = "would"
        if modal:
            chain.append(("modal", modal))
        if "k" in m:
            chain.append(("have", "have"))
        if "g" in m:
            chain.append(("be", "be"))
        chain.append(("main", lemma))

        words = []
        for i, (kind, w) in enumerate(chain):
            prev = chain[i - 1][0] if i else None
            if i == 0 and kind != "modal":
                words.append(self.finite(w, past, first_person, third_sg))
            elif kind == "modal":
                words.append(w)
            elif prev == "modal":
                words.append(w)
            elif prev == "have":
                words.append(inflect(w, "VBN") if w != "be" else "been")
            elif prev == "be":
                words.append(inflect(w, "VBG"))
        if "o" in m:
            words = [lemma]
            if neg:
                words = ["do", "not"] + words
            return [], words
        aux_first = len(chain) > 1 or lemma == "be"
        if neg or question:
            if not aux_first:
                do = self.finite("do", past, first_person, third_sg)
                words = [do, lemma]
            if neg:
                words.insert(1, "not")
        if question:
            return [words[0]], words[1:]
        return [], words

    def finite(self, w, past, first_person, third_sg):
        if w == "be":
            if past:
                return "was" if (first_person or third_sg) else "were"
            return "am" if first_person else "is" if third_sg else "are"
        if w == "have" and not past:
            return "has" if third_sg else "have"
        if past:
            return inflect(w, "VBD")
        return inflect(w, "VBZ") if third_sg else w

    # -------------------------------------------------- clauses
    def clause(self, c, ctx, question=False):
        if isinstance(c, Bare):
            if self.matches(c.np, ctx.gap):
                return ""
            many = "s" in c.np.head.mods or any(n.raw not in ("#1", "#one") for n in c.np.nums) \
                or (c.np.head.kind == "num" and c.np.head.raw != "#1")
            there = "there are " if many else "there is "
            return (there if not question else "") + self.np(c.np, ctx, "obj")
        if isinstance(c, Logic):
            return self.logic(c, ctx)
        if isinstance(c, Wrap):
            return self.wrap(c, ctx, question)
        return self.core(c, ctx, question)

    def logic(self, c, ctx):
        r, m = c.rel.idx, c.rel.mods
        a, b = self.clause(c.a, ctx), self.clause(c.b, ctx)
        name = RESERVED_VERBS[r][0]
        if r == V["and"]:
            return f"{a} and {b}"
        if r == V["or"]:
            return f"{a} or {b}"
        if r == V["but"]:
            return f"{a}, but {b}"
        if r == V["cause"]:
            return f"{b} because {a}"
        if r == V["if"]:
            return f"{'unless' if 'x' in m else 'if'} {a}, {b}"
        if r == V["time"]:
            return f"when {a}, {b}"
        if r == V["during"]:
            return f"while {a}, {b}"
        if isinstance(c.a, Core) and isinstance(c.b, Core) and c.a.subj.head.base == c.b.subj.head.base:
            a = self.core(c.a, ctx, gerund=True)
        prep = REL_WORD.get(r, name)
        return f"{b} {prep} {a}"

    def wrap(self, c, ctx, question=False):
        r = c.rel.idx
        if VERB_KIND.get(r) == "quant":
            ctx.quant[c.land.head.base] = QUANT_WORD[r]
            return self.clause(c.body, ctx, question)
        key = (c.land.head.base, "".join(x for x in c.land.head.mods if x in "qde"))
        if key in ADVERBIAL:
            word = SPECIAL[key]
            if "q" in key[1]:
                self._q = True
                return f"{word} {self.clause(c.body, ctx, question=True)}"
            return f"{self.clause(c.body, ctx, question)} {word}"
        if r == V["with"] and c.land.head.base == "n7" and c.land.adjs and not c.land.head.mods:
            advs = " ".join(self.adj(a, adverb=True) for a in c.land.adjs)
            return f"{advs}, {self.clause(c.body, ctx, question)}"
        prep = REL_WORD.get(r, RESERVED_VERBS[r][0])
        if "x" in c.rel.mods:
            prep = {"with": "without", "like": "unlike"}.get(prep, "not " + prep)
        if self.matches(c.land, ctx.gap):          # "the house in which I live"
            inner = self.clause(c.body, Ctx(), question)
            return f"{prep} which {inner}"
        body = self.clause(c.body, ctx, question)
        land = self.np(c.land, ctx, "obj")
        return f"{body} {prep} {land}".strip()

    def core(self, c, ctx, question=False, infinitive=False, gerund=False):
        v = c.verb
        if gerund:
            lemma = RESERVED_VERBS[v.idx][0] if v.kind == "word" and v.idx in RESERVED_VERBS else self.lemma(v)
            copula = v.kind == "word" and v.idx == V["be"]
            parts = (["not"] if "x" in v.mods else []) + [inflect(lemma, "VBG")]
            parts += [self.adj(a, adverb=not copula) for a in c.advs]
            if c.obj is not None:
                parts.append(self.np(c.obj, ctx, "obj"))
            parts += [self.complement(x, c, ctx) for x in c.comps]
            return " ".join(p for p in parts if p)
        if infinitive:
            lemma = RESERVED_VERBS[v.idx][0] if v.kind == "word" and v.idx in RESERVED_VERBS else self.lemma(v)
            copula = v.kind == "word" and v.idx == V["be"]
            parts = (["not"] if "x" in v.mods else []) + ["to", lemma]
            parts += [self.adj(a, adverb=not copula) for a in c.advs]
            if c.obj is not None:
                parts.append(self.np(c.obj, ctx, "obj"))
            parts += [self.complement(x, c, ctx) for x in c.comps]
            return " ".join(p for p in parts if p)
        q = question or "q" in v.mods
        if "q" in v.mods:
            self._q = True
        subj_gap = self.matches(c.subj, ctx.gap)
        obj_gap = self.matches(c.obj, ctx.gap)
        obj_wh = c.obj is not None and c.obj.head.base in ("n3", "n4") and "q" in c.obj.head.mods
        if obj_wh:
            q = True
            self._q = True
        before, after = self.verb_group(v, c.subj, question=q and not subj_gap)
        subj = "" if "o" in v.mods and c.subj.head.base == "n2" else self.np(c.subj, ctx, "subj")
        if subj_gap:
            subj = ctx.gap_word
        copula = v.kind == "word" and v.idx == V["be"]
        advs = [self.adj(a, adverb=not copula) for a in c.advs]
        obj = ""
        if c.obj is not None and not obj_gap and not obj_wh:
            obj = self.np(c.obj, ctx, "obj")
        comps = [self.complement(x, c, ctx) for x in c.comps]
        if copula and not advs and not obj and not comps:
            pass
        parts = before + [subj] + after + advs + [obj] + comps
        s = " ".join(p for p in parts if p)
        if obj_gap:
            s = f"{ctx.gap_word} {s}"
        if obj_wh:
            s = f"{self.np(c.obj, ctx)} {s}"
        return s

    def complement(self, comp, main, ctx):
        if isinstance(comp, Core):
            same = comp.subj.head.base in (main.subj.head.base, main.obj.head.base if main.obj else None)
            if same and not comp.verb.mods.replace("x", ""):
                return self.core(comp, ctx, infinitive=True)
        return "that " + self.clause(comp, ctx)

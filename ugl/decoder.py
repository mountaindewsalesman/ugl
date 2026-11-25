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
           | L mod+ \\L mod+ \\L   coordinated modifiers ("cell and car numbers"), L = v4 / v5
           | v3 clause \\v3   relative clause (head noun repeated inside)
           | np-with-y    possessor (always last; its own mods follow it)
"""
import re
from dataclasses import dataclass, field
from typing import List, Optional

from .core import (PREFIX_TEXT, number_text, parse_literal, WORD_RE, RESERVED_NOUNS, RESERVED_VERBS, VERB_KIND, WRAPPER_VERBS,
                   CLAUSE_FORM_VERBS, V, tokenize)

try:
    from lemminflect import getInflection
except ImportError:  # pragma: no cover
    getInflection = None


COORD_VERBS = {V["and"], V["or"]}          # these also coordinate noun phrases


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
    members: List["NP"] = field(default_factory=list)    # coordinated NP: head is the and/or token


@dataclass
class Group:
    """Coordinated modifiers after a head noun: v4 n#j \\v4 n#j \\v4."""
    rel: Tok
    members: List[List[object]]       # each a run of modifiers (Tok, or a nested Group)
    kind: str = "group"
    cls: str = ""
    mods: str = ""

    def leaves(self):
        for m in self.members:
            for x in m:
                yield from (x.leaves() if isinstance(x, Group) else [x])


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
        self.opener = {}        # closing-tag index -> index of the word it closes (see humanize.tree)

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
        return ((x.kind == "word" and x.cls in ("n", "nv")) or x.kind in ("num", "name")
                or (x.kind == "unk" and x.cls == "n")
                or (x.kind == "word" and x.cls == "v" and x.idx in COORD_VERBS))   # coordinated NP

    def is_verb(self, x):
        return x is not None and ((x.kind == "word" and x.cls == "v") or (x.kind == "unk" and x.cls == "v"))

    def expect_close(self, i, vid, opener):
        if self.peek(i) != f"\\v{vid}":
            raise ParseError(f"expected \\v{vid} at {i}, got {self.peek(i)}")
        self.opener[i] = opener
        return i + 1

    def clause(self, i):
        x = self.tokat(i)
        if x is None:
            raise ParseError(f"clause expected at {i}")
        np = None
        if x.kind == "word" and x.cls == "v" and x.idx in COORD_VERBS:
            # "and" over two noun phrases is a coordinated noun phrase; followed
            # by a verb it is the subject ("John and Mary met")
            save = i
            try:
                np, i = self.group_np(i)
            except ParseError:
                np, i = None, save
        if np is None and x.kind == "word" and x.cls == "v" and x.idx in CLAUSE_FORM_VERBS:
            o = i
            a, i = self.clause(i + 1)
            i = self.expect_close(i, x.idx, o)
            b, i = self.clause(i)
            i = self.expect_close(i, x.idx, o)
            return Logic(x, a, b), i
        if np is None:
            np, i = self.np(i)
        v = self.tokat(i)
        if self.is_verb(v):
            if v.kind == "word" and v.idx in WRAPPER_VERBS:
                o = i
                body, i = self.clause(i + 1)
                i = self.expect_close(i, v.idx, o)
                return Wrap(np, v, body), i
            vpos = i
            i += 1
            advs = []
            while True:
                a = self.tokat(i)
                if a is not None and (a.cls == "a" or self.is_attr_noun(a)):
                    advs.append(a)                  # "is anti-war": an attribute noun says what it is like
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
                        self.opener[j] = vpos
                        i = j + 1
                    else:
                        raise ParseError("not a complement")
                except ParseError:
                    i = save
                    if self.starts_np(i):
                        obj, i = self.np(i)
                        if self.peek(i) == f"\\v{vid}":   # the NP was a bare complement
                            self.opener[i] = vpos
                            comps.append(Bare(obj))
                            obj, i = None, i + 1
            while self.starts_np(i) or self.clause_form_start(i):
                c, i = self.clause(i)
                i = self.expect_close(i, vid, vpos)
                comps.append(c)
            return Core(np, v, advs, obj, comps), i
        return Bare(np), i

    @staticmethod
    def is_attr_noun(x):
        return "j" in x.mods and (x.cls in ("n", "nv") or x.kind == "name")

    def mod_run(self, i):
        """Modifiers up to the next closing tag: one member of a modifier group."""
        run = []
        while not self.is_close(i):
            x = self.tokat(i)
            if x is None:
                break
            if x.cls == "a" or x.kind == "num" or self.is_attr_noun(x):
                run.append(x); i += 1
            elif x.kind == "word" and x.cls == "v" and x.idx in COORD_VERBS and "y" not in x.mods:
                g, i = self.mod_group(i)
                run.append(g)
            else:
                raise ParseError(f"modifier expected at {i}: {self.peek(i)}")
        if not run:
            raise ParseError(f"empty modifier group at {i}")
        return run, i

    def mod_group(self, i):
        x, o = self.tokat(i), i
        a, i = self.mod_run(i + 1)
        i = self.expect_close(i, x.idx, o)
        b, i = self.mod_run(i)
        i = self.expect_close(i, x.idx, o)
        return Group(x, [a, b]), i

    def clause_form_start(self, i):
        x = self.tokat(i)
        return x is not None and x.kind == "word" and x.cls == "v" and x.idx in CLAUSE_FORM_VERBS

    def coord_np(self, i):
        x, o = self.tokat(i), i
        a, i = self.np(i + 1)
        i = self.expect_close(i, x.idx, o)
        b, i = self.np(i)
        i = self.expect_close(i, x.idx, o)
        return NP(x, members=[a, b]), i

    def group_np(self, i):
        """A coordinated noun phrase with the modifiers of the whole group after
        its closing tag (SPEC rule 13)."""
        g, i = self.coord_np(i)
        while i < self.n and not self.is_close(i):
            y = parse_tok(self.t[i])
            if y.kind == "word" and y.cls == "v" and y.idx == V["that"]:
                o = i
                c, i = self.clause(i + 1)
                i = self.expect_close(i, V["that"], o)
                g.rels.append(c)
            elif "y" in y.mods and (y.cls in ("n", "nv") or (y.kind == "word" and y.idx in COORD_VERBS)):
                g.poss, i = (self.group_np(i) if y.cls == "v" else self.np(i))
                break
            else:
                break
        return g, i

    def np(self, i):
        x = self.tokat(i)
        if x is not None and x.kind == "word" and x.cls == "v" and x.idx in COORD_VERBS:
            return self.group_np(i)
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
                o = i
                c, i = self.clause(i + 1)
                i = self.expect_close(i, V["that"], o)
                np.rels.append(c)
            elif "y" in x.mods and x.cls in ("n", "nv"):
                np.poss, i = self.np(i)
                break
            elif "y" in x.mods and x.kind == "word" and x.cls == "v" and x.idx in COORD_VERBS:
                np.poss, i = self.group_np(i)              # v4y A \v4 B \v4 = of (A and B)
                break
            elif x.kind == "word" and x.cls == "v" and x.idx in COORD_VERBS:
                # right after a noun, and/or over modifiers only is a modifier group;
                # anything else (a coordinated clause or noun phrase) ends the noun phrase
                opened = dict(self.opener)
                try:
                    g, j = self.mod_group(i)
                except ParseError:
                    self.opener = opened
                    break
                leaves = list(g.leaves())
                if all(t.kind == "num" for t in leaves):
                    np.nums.append(g)
                elif all(t.cls == "a" for t in leaves):
                    np.adjs.append(g)
                else:
                    np.comps.append(NP(g))
                i = j
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
# relations whose English word is a subordinating conjunction: a full clause may follow
SUBORDINATORS = {V["before"], V["after"], V["until"], V["since"]}
QUANT_WORD = {V["all"]: "every", V["most"]: "most", V["many"]: "many", V["few"]: "few"}
RESERVED_NOUN_BASES = {f"n{i}" for i in RESERVED_NOUNS}
# one-word pronouns that take their adjectives after them ("something new", "nobody else")
FUSED = {"something", "nothing", "someone", "nobody", "what", "who"}
# Adverb position.  Before the main verb: -ly adverbs and the frequency / time /
# focus words ("often bore quotes", "later withdrew", "also won"); between verb and
# object they would read as part of the object.  Every other adverb - particles
# ("pick it up"), flat adverbs ("ran fast", "worked harder") - follows the verb and
# its object.
PREVERBAL = {"often", "always", "never", "sometimes", "usually", "also", "still", "already",
             "just", "later", "soon", "even", "only", "rarely", "seldom", "once", "again",
             "first", "then", "now", "almost", "nearly", "eventually", "ever", "frequently"}


def preverbal(w):
    last = w.split()[-1]
    return last.endswith("ly") or last in PREVERBAL
VOWEL = re.compile(r"^[aeiou]", re.I)


def inflect(lemma, tag):
    if " " in lemma:
        if tag == "NNS" and " of " not in f" {lemma} ":
            # a compound noun's plural is on its head, the last word: "love songs"
            # ("cups of tea" keeps it on the first)
            rest, last = lemma.rsplit(" ", 1)
            return rest + " " + inflect(last, tag)
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
            return number_text(t.raw)
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
            own = self.lex.surface(t.cls, t.idx, adverb=True) if t.kind == "word" else None
            w = own or adverb_form(w)               # "currently", not "presently" from "present"
        base = self.lemma(t)
        stem = base.lower()[:-1] if base.lower().endswith("e") and not base.lower().endswith("ee") else base.lower()
        syll = len(re.findall(r"[aeiouy]+", stem))       # a final silent e is no syllable: large, simple
        # English compares short words with an ending: one syllable, or two ending in -y
        short = " " not in base and (syll <= 1 or (syll == 2 and base.endswith("y") and not adverb))
        for letter, tag, word in (("r", "JJR", "more"), ("z", "JJS", "most")):
            if letter in m:
                # "bigger", "faster" (an adverb compares like its short adjective);
                # long words and -ly adverbs take more / most: "more carefully"
                f = inflect(base, tag) if short else ""
                ok = f and f != base and " " not in f
                w = f if ok else f"{word} {w}"
        if "l" in m:
            w = "less " + w
        if "w" in m:
            w = "very " + w
        if "x" in m:
            w = "not " + w
        return self.affix(t, w)

    @staticmethod
    def affix(t, w):
        """Prefix letters back to English: b non-, i anti-."""
        for letter in ("b", "i"):
            if letter in t.mods:
                w = PREFIX_TEXT[letter] + w
        return w

    def mod_text(self, x, adverb=False):
        """Any modifier: adjective, attribute noun, count, or a group of them."""
        if isinstance(x, NP):
            x = x.head
        if isinstance(x, Group):
            word = "or" if x.rel.idx == V["or"] else "and"
            parts = []
            for i, m in enumerate(x.members):
                # a nested group of the same kind continues the list: "red, white and blue"
                if i == len(x.members) - 1 and len(m) == 1 and isinstance(m[0], Group) and m[0].rel.idx == x.rel.idx:
                    parts += self.mod_text(m[0], adverb).split(f" {word} ")
                else:
                    parts.append(" ".join(self.mod_text(t, adverb) for t in m))
            return ", ".join(parts[:-1]) + f" {word} " + parts[-1] if len(parts) > 2                 else f" {word} ".join(parts)
        if x.kind == "num":
            return self.lemma(x)
        if Parser.is_attr_noun(x):
            return self.affix(x, self.lemma(x))
        return self.adj(x, adverb)

    def count_text(self, n):
        w = self.mod_text(n)
        # "tens of kilometers": a plural round number counts in units of itself
        return w + (" of" if w.endswith("s") and not w[0].isdigit() and not isinstance(n, Group) else "")

    # -------------------------------------------------- noun phrases
    def namelike(self, np):
        """Possessor written with 's (John's, my, John and Mary's) rather than 'of'."""
        if np.members:
            return all(self.namelike(m) for m in np.members)
        h = np.head
        return h.kind == "name" or h.base in RESERVED_NOUN_BASES or self.lex.is_proper(h.raw)

    def matches(self, np, base):
        return base is not None and np is not None and np.head.base == base

    def np(self, np, ctx, case="subj"):
        h = np.head
        if np.members:                                   # "John and Mary"
            word = "or" if h.idx == V["or"] else "and"
            quant = ctx.quant.pop(h.base, None) if h.base in ctx.quant else None
            parts = [self.np(m, ctx, case if case != "poss" else "obj") for m in np.members]
            if np.rels or np.poss is not None:
                # a group with its own modifiers says its determiner once ("a writer and
                # political activist who ..."); a repeated one would split the group
                lead = re.match(r"(the|an?|these|those|this|that) ", parts[0])
                for k in range(1, len(parts)):
                    m = re.match(r"(the|an?|these|those|this|that) ", parts[k])
                    if lead and m and (m.group(1) == lead.group(1) or {m.group(1), lead.group(1)} <= {"a", "an"}):
                        parts[k] = parts[k][m.end():]
            s = f" {word} ".join(parts)
            if quant in ("every", "all"):                # "all of (cats and dogs)" = both cats and dogs
                s = ("both " if len(parts) == 2 and word == "and" else "all of ") + s
            elif quant:
                s = f"{quant} of {s}"
            if np.poss is not None:                      # "the leader and founder of the party"
                s += " of " + self.np(np.poss, ctx, "obj")
            for r in np.rels:                            # "a writer and activist who served"
                sub = Ctx()
                sub.gap = h.base
                person = all(m.head.kind == "name" or self.lex.is_person_token(m.head.raw) for m in np.members)
                sub.gap_word = "who" if person else "that"
                s += " " + self.clause(r, sub)
            return s + "'s" if case == "poss" else s
        if h.kind == "word" and h.cls == "n" and h.idx in RESERVED_NOUNS:
            key = pron_key(h.base, h.mods)
            if key in PRON and not ("x" in h.mods or "q" in h.mods) and not np.rels and not np.adjs:
                p = PRON[key]
                return p[0] if case == "subj" else p[1] if case == "obj" else p[2]
            # table keys spell the letters in a fixed order (determiner, then s)
            m = "".join(c for c in "qxdet" if c in h.mods) + ("s" if "s" in h.mods else "")
            word = SPECIAL.get((h.base, m)) or SPECIAL.get((h.base, m.replace("s", ""))) or RESERVED_NOUNS[h.idx]
            rq = ctx.quant.pop(h.base, None) if h.base in ctx.quant else None
            if rq:                                  # "many things", "all things"
                word = ("all" if rq == "every" else rq) + " " + SPECIAL.get((h.base, "s"), word)
            nums = [self.count_text(n) for n in np.nums]
            mods = [self.mod_text(a) for a in np.adjs] + [self.mod_text(c) for c in np.comps]
            if word.startswith(("the ", "all ", "many ", "most ", "few ")):   # "the tribal things"
                det_, noun_ = word.rsplit(" ", 1)
                words = [det_] + nums + mods + [noun_]
            elif word in FUSED:                     # "something toxic": English puts them after
                words = nums + [word] + mods
            else:                                   # "tribal things", "these red ones"
                words = nums + mods + [word]
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
        elif plural and (h.kind == "word" or (h.kind == "unk" and h.cls == "n")):
            noun = inflect(noun, "NNS")                  # also words outside the dictionary
        # an adverb in a counted phrase modifies the number: "about 100 degrees"
        noun = self.affix(h, noun)                   # "non-members"
        lead = [self.mod_text(a) for a in np.adjs if np.nums and self.is_adverb_sense(a)]
        pre = ([self.mod_text(a) for a in np.adjs if not (np.nums and self.is_adverb_sense(a))]
               + [self.mod_text(c) for c in np.comps])
        nums = [self.count_text(n) for n in np.nums]
        det = ""
        of_part = ""
        poss_said = False                               # the possessor is already the determiner
        quant = ctx.quant.pop(h.base, None) if h.base in ctx.quant else None
        if quant:
            if quant == "every" and (plural or np.poss is not None or set(h.mods) & set("det")):
                quant = "all"                               # all friends / all my friends / all these
            det = quant
            if np.poss is not None:                         # "all my friends", "most of John's books"
                det += (" " if quant == "all" else " of ") + self.np(np.poss, ctx, "poss")
            elif "d" in h.mods or "e" in h.mods:            # "most of these wars"
                dem = ("these" if plural else "this") if "d" in h.mods else ("those" if plural else "that")
                det += (" " if quant == "all" else " of ") + dem
            elif "t" in h.mods:                             # "all the dogs", "most of the apples"
                det += (" " if quant == "all" else " of ") + "the"
        elif np.poss is not None and "t" in h.mods and np.poss.poss is None and not np.poss.rels \
                and (self.namelike(np.poss) or np.rels):
            # "Simon's book"; and "the audience's appreciation for X" whenever the noun
            # has a phrase of its own after it, which "of the audience" would capture.
            # A chain reads as "of": "the award of Brandenburg's state"
            det = self.np(np.poss, ctx, "poss")          # "Simon's book", "their numbers" (definite)
            poss_said = True
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
        if np.poss is not None and not poss_said and not quant:
            of_part = "of " + self.np(np.poss, ctx, "obj")   # "the bank of the river", "an arson of houses"
        words = [det] + lead + nums + pre + [noun, of_part]
        for r in np.rels:
            sub = Ctx()
            sub.gap = h.base
            person = h.kind == "name" or self.lex.is_proper(h.raw) or (h.kind == "word" and self.lex.is_person_token(h.raw))
            sub.gap_word = "who" if person else "that"
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
        plural_subj = subj is not None and ("s" in subj.head.mods or sbase == "n2" or bool(subj.members))
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
        prep = REL_WORD.get(r, name)
        if r in SUBORDINATORS:                         # "after he left": a clause can follow
            if self.same_subject(c.a, c.b):
                a = self.gerund(c.a, ctx)              # "after matriculating"
            return f"{b} {prep} {a}"
        # a preposition takes no finite clause ("for he establishes" is not English):
        # same subject -> "to establish" (purpose) / "by sampling"; another subject ->
        # "for X to establish" / "with the group keeping the name"
        if self.same_subject(c.a, c.b):
            if r == V["for"] and self.infinitival_body(c.a):
                return f"{b} {self.infinitive(c.a, ctx)}"
            return f"{b} {prep} {self.gerund(c.a, ctx)}"
        subj = self.core_of(c.a)
        if subj is not None and self.infinitival_body(c.a):
            who = self.np(subj.subj, ctx, "obj")
            if r == V["for"]:
                return f"{b} for {who} {self.infinitive(c.a, ctx)}"
            return f"{b} {prep} {who} {self.gerund(c.a, ctx)}"
        return f"{b} {prep} {a}"

    @staticmethod
    def core_of(x):
        """The core clause at the bottom of relation wrappers / the first of a coordination."""
        while isinstance(x, (Wrap, Logic)):
            x = x.body if isinstance(x, Wrap) else x.a
        return x if isinstance(x, Core) else None

    def same_subject(self, a, b):
        """Every clause in a (a coordination too) has b's subject."""
        cb = self.core_of(b)
        if cb is None:
            return False
        if isinstance(a, Logic) and a.rel.idx in COORD_VERBS:
            return self.same_subject(a.a, b) and self.same_subject(a.b, b)
        ca = a if isinstance(a, Core) else None
        return ca is not None and ca.subj.head.base == cb.subj.head.base

    def infinitival_body(self, a):
        """Can a be said without tense ("to set up X and inspire Y")?"""
        if isinstance(a, Logic) and a.rel.idx in COORD_VERBS:
            return self.infinitival_body(a.a) and self.infinitival_body(a.b)
        return isinstance(a, Core) and not a.verb.mods.replace("x", "")

    def gerund(self, a, ctx):
        if isinstance(a, Logic) and a.rel.idx in COORD_VERBS:
            word = "or" if a.rel.idx == V["or"] else "and"
            return f"{self.gerund(a.a, ctx)} {word} {self.gerund(a.b, ctx)}"
        if isinstance(a, Core):
            return self.core(a, ctx, gerund=True)
        return self.clause(a, ctx)

    def wrap(self, c, ctx, question=False):
        r = c.rel.idx
        if VERB_KIND.get(r) == "quant":
            ctx.quant[c.land.head.base] = QUANT_WORD[r]
            return self.clause(c.body, ctx, question)
        if isinstance(c.body, Logic) and c.body.rel.idx not in COORD_VERBS and VERB_KIND.get(c.body.rel.idx) == "rel":
            # a relation around "B for A" modifies the main event B: "moved across India
            # to set up shakhas", not "... to set up shakhas across India"
            inner = c.body
            return self.logic(Logic(inner.rel, inner.a, Wrap(c.land, c.rel, inner.b)), ctx)
        key = (c.land.head.base, "".join(x for x in c.land.head.mods if x in "qde"))
        if key in ADVERBIAL:
            word = SPECIAL[key]
            if "q" in key[1]:
                self._q = True
                return f"{word} {self.clause(c.body, ctx, question=True)}"
            return f"{self.clause(c.body, ctx, question)} {word}"
        if r == V["with"] and c.land.head.base == "n7" and c.land.adjs and not c.land.head.mods:
            advs = " ".join(self.mod_text(a, adverb=True) for a in c.land.adjs)
            return f"{advs}, {self.clause(c.body, ctx, question)}"
        if self.matches(c.land, ctx.gap):          # "the house in which I live"
            inner = self.clause(c.body, Ctx(), question)
            pron = "whom" if ctx.gap_word == "who" else "which"   # "a woman with whom he writes"
            return f"{self.preposition(c)} {pron} {inner}"
        rel = self.relation(c, ctx)
        body = self.clause(c.body, ctx, question)
        core = c.body
        while isinstance(core, Wrap) and VERB_KIND.get(core.rel.idx) != "quant":
            core = core.body                          # stacked relations: look at the clause inside
        if isinstance(core, Core) and core.obj is not None and not question:
            # "Simon read his book on the bank": a final phrase could attach to the
            # object; in front it can only modify the clause, as it does in UGL
            return f"{rel}, {body}"                  # (the sentence start is capitalised once, at the end)
        return f"{body} {rel}".strip()

    def preposition(self, c):
        r = c.rel.idx
        prep = REL_WORD.get(r, RESERVED_VERBS[r][0])
        m = re.fullmatch(r"#(\d{3,4}|-)((?:-\d\d){0,2})", c.land.head.raw) if c.land.head.kind == "num" else None
        if r == V["time"] and m:
            # calendar landmarks: "in 2017", "in June 1944", "on 21 June 1940", "on 11 March"
            prep = "on" if m.group(2).count("-") == 2 else "in"
        if "x" in c.rel.mods:
            prep = {"with": "without", "like": "unlike"}.get(prep, "not " + prep)
        return prep

    def relation(self, c, ctx):
        """A relation wrapper's own words: "to the school"."""
        return f"{self.preposition(c)} {self.np(c.land, ctx, 'obj')}"

    def core(self, c, ctx, question=False, infinitive=False, gerund=False):
        v = c.verb
        if gerund:
            lemma = RESERVED_VERBS[v.idx][0] if v.kind == "word" and v.idx in RESERVED_VERBS else self.lemma(v)
            copula = v.kind == "word" and v.idx == V["be"]
            parts = (["not"] if "x" in v.mods else []) + [inflect(lemma, "VBG")]
            parts += [self.mod_text(a, adverb=not copula) for a in c.advs]
            if c.obj is not None:
                parts.append(self.np(c.obj, ctx, "obj"))
            parts += [self.complement(x, c, ctx) for x in c.comps]
            return " ".join(p for p in parts if p)
        if infinitive:
            lemma = RESERVED_VERBS[v.idx][0] if v.kind == "word" and v.idx in RESERVED_VERBS else self.lemma(v)
            copula = v.kind == "word" and v.idx == V["be"]
            parts = (["not"] if "x" in v.mods else []) + ["to", lemma]
            parts += [self.mod_text(a, adverb=not copula) for a in c.advs]
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
        advs = [self.mod_text(a, adverb=not copula) for a in c.advs]
        obj = ""
        if c.obj is not None and not obj_gap and not obj_wh:
            obj = self.np(c.obj, ctx, "obj")
        comps = [self.complement(x, c, ctx) for x in c.comps]
        late = advs
        if not copula and after:
            # English puts an adverb before the main verb ("his letters often bore
            # quotes"); between verb and object it reads as part of the object
            early = [w for w in advs if preverbal(w)]
            late = [w for w in advs if not preverbal(w)]
            after = after[:-1] + early + after[-1:]
        particles = [w for w in late if w in ("up", "down", "out", "off", "away", "back", "over", "in", "on")]
        late = [w for w in late if w not in particles]
        parts = before + [subj] + after + particles + [obj] + late + comps
        s = " ".join(p for p in parts if p)
        if obj_gap:
            s = f"{ctx.gap_word} {s}"
        if obj_wh:
            s = f"{self.np(c.obj, ctx)} {s}"
        return s

    def infinitival(self, comp, main):
        """Can this complement be said as "to ..."?  Its subject is the main
        clause's subject or object and its verb carries no tense or mood; a
        joined complement qualifies when every part does."""
        if isinstance(comp, Logic) and comp.rel.idx in (V["and"], V["or"]):
            return self.infinitival(comp.a, main) and self.infinitival(comp.b, main)
        if isinstance(comp, Core):
            same = comp.subj.head.base in (main.subj.head.base, main.obj.head.base if main.obj else None)
            return same and not comp.verb.mods.replace("x", "")
        if isinstance(comp, Wrap) and VERB_KIND.get(comp.rel.idx) != "quant" \
                and (comp.land.head.base, "".join(x for x in comp.land.head.mods if x in "qde")) not in ADVERBIAL:
            return self.infinitival(comp.body, main)        # "to be committed to X"
        return False

    def infinitive(self, comp, ctx, to=True):
        if isinstance(comp, Logic):
            word = "or" if comp.rel.idx == V["or"] else "and"
            return f"{self.infinitive(comp.a, ctx, to)} {word} {self.infinitive(comp.b, ctx, to=False)}"
        if isinstance(comp, Wrap):
            return f"{self.infinitive(comp.body, ctx, to)} {self.relation(comp, ctx)}"
        s = self.core(comp, ctx, infinitive=True)
        return s if to else s.replace("to ", "", 1) if s.startswith(("to ", "not to ")) else s

    def complement(self, comp, main, ctx):
        if self.infinitival(comp, main):
            return self.infinitive(comp, ctx)
        return "that " + self.clause(comp, ctx)

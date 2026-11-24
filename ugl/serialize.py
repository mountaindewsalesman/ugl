"""Event/Entity graph -> UGL tokens.  The only place that knows token spelling
and the ordering rules of SPEC.md:

    clause := quantifier wrappers ( relation wrappers ( core ) ) + coordination
    core   := subject verb adverbs [object] (complement \\verb)*
    np     := head attrs count restrictive-clauses possessor
"""
import re

from .core import V, add_mods, literal, tok
from .ir import Entity, Event, refer

WORD = re.compile(r"^(nv|n|v|a)(\d+)([a-z]*)$")


class Serializer:
    def __init__(self, lexicon):
        self.lex = lexicon
        self.unknown = set()
        self.words = []           # current sentence words, for sense choice

    # ------------------------------------------------------------- words
    def word(self, lemma, cls, mods="", pos=None):
        sent = (self.words, pos) if pos is not None and self.words else None
        r = self.lex.encode_word(lemma, cls, mods, sent=sent)
        if r is None and cls == "a":
            r = self.lex.encode_word(lemma, "r", mods, sent=sent)
        if r is None:
            self.unknown.add((lemma, cls))
            return [literal("unk", lemma, mods, "a" if cls == "r" else cls)]
        return r

    # ------------------------------------------------------------- nouns
    def mention(self, e):
        """Bare re-mention of an entity (replaces pronouns): head only, definite."""
        e = e.root()
        if e.kind == "coord":
            return self.mention(e.members[0])
        if e.kind == "lex":
            head = self.word(e.lemma, "v" if e.cls == "nv" else "n", pos=e.pos)[0]
            m = WORD.match(head)
            if m:
                cls = "nv" if e.cls == "nv" else m.group(1)
                return [tok(cls, m.group(2), "t" + ("s" if e.plural else ""))]
            return [head]
        if e.kind == "name":
            return self.proper(e.lemma, "s" if e.plural else "")[:1]
        if e.kind == "reserved":
            return [add_mods(e.lemma, "s" if e.plural else "")]
        if e.kind == "num":
            return [e.lemma]
        return ["n3t"]

    @staticmethod
    def name(s):
        return literal("name", s)

    def proper(self, text, mods=""):
        """A frequent name is a numbered noun; a rare one stays a literal."""
        t = self.lex.name_tokens(text, mods)
        return t if t is not None else [add_mods(self.name(text), mods)]

    def np(self, e, quants):
        if e.ref is not None:
            return self.mention(e)
        if e.kind == "coord":
            return self.np(e.members[0], quants)       # only reached for coordinated possessors etc.
        if e.kind == "pronoun":                          # unresolved and not handled by coref
            return ["n3t"]
        mods = ("s" if e.plural else "") + (e.det if (not e.quant or e.det in "de") else "") \
            + ("x" if e.neg else "") + ("q" if e.question else "")
        if e.kind == "reserved":
            head = [add_mods(e.lemma, mods)]
        elif e.kind == "name":
            head = self.proper(e.lemma, mods)
        elif e.kind == "num":
            head = [e.lemma]
        else:
            toks = self.word(e.lemma, "v" if e.cls == "nv" else "n", mods, pos=e.pos)
            if e.cls == "nv" and toks[0].startswith("v"):
                toks[0] = "nv" + toks[0][1:]
            head = toks
        out = list(head)
        for a in e.attrs:
            out += self.attr(a)
        if e.count:
            out.append(e.count)
        for r in e.restrictors:
            out += [tok("v", V["that"])] + self.clause(r) + [f"\\v{V['that']}"]
        if e.possessor is not None:
            p = self.np(e.possessor, quants)
            out += [add_mods(p[0], "y")] + p[1:]
        if e.quant:
            quants.append((list(head), V[e.quant]))
        return out

    def attr(self, a):
        if a.kind == "name":
            t = self.proper(a.lemma)
            return [add_mods(t[0], "j")] + t[1:]
        if a.kind == "num":
            return [a.lemma]
        if a.kind == "noun":
            t = self.word(a.lemma, "n", pos=a.pos)
            return [add_mods(t[0], "j")] + t[1:]
        return self.word(a.lemma, "r" if a.adverb else "a", a.mods, pos=a.pos)

    # ------------------------------------------------------------- clauses
    @staticmethod
    def indefinite(e):
        """'there is X' only for indefinite things; 'I am in Plano' keeps its verb."""
        if e is None:
            return True
        if e.kind == "coord":
            return all(Serializer.indefinite(m) for m in e.members)
        # a re-mention inside a restrictor ("speakers [that are] in Slovakia") needs no "be" either
        return e.ref is not None or (e.kind == "lex" and not e.det and e.possessor is None)

    def verb(self, ev):
        if ev.pred in ("be", "exist"):
            return [tok("v", V["be"], ev.mods)]
        if ev.pred == "have":
            return [tok("v", V["have"], ev.mods)]
        return self.word(ev.lemma, "v", ev.mods, pos=ev.pos)

    def core(self, ev, subj, obj, quants):
        s = self.np(subj, quants) if subj is not None else ["n3t"]
        if ev.pred == "exist" and obj is None and not ev.pred_attrs and not ev.comps \
                and not ev.mods and self.indefinite(subj):
            return s                                     # bare noun phrase: "there is a dog"
        vt = self.verb(ev)
        out = s + vt
        for a in ev.pred_attrs:                      # "is tall": only "be" has these
            out += self.attr(a)
        if ev.pred not in ("be", "exist"):           # after "be" the slot means a predicate adjective,
            for a in ev.manner:                      # so manner goes into a "with an X way" relation there
                out += self.attr(a)
        if obj is not None:
            out += self.np(obj, quants)
        m = WORD.match(vt[0])
        vid = m.group(2) if m else "0"
        for c in ev.comps:
            out += self.clause(c) + [f"\\v{vid}"]
        return out

    def distribute(self, ev, quants):
        """Coordinated subject/object noun phrases: one core clause per member."""
        def members(e):
            if e is None:
                return [None], "and"
            r = e.root()
            if r.kind != "coord" or not r.members:
                return [e], "and"
            # a re-mention of a coordination re-mentions each member
            return ([refer(m) for m in r.members] if e.ref is not None else r.members), r.logic
        subjs, ls = members(ev.subj)
        objs, lo = members(ev.obj)
        logic = ls if len(subjs) > 1 else lo
        cores = [self.core(ev, s, o, quants) for s in subjs for o in objs]
        out = cores[-1]
        for c in reversed(cores[:-1]):
            out = self.logic(logic, c, out)
        return out

    @staticmethod
    def logic(name, a, b):
        i = V[name]
        return [tok("v", i)] + a + [f"\\v{i}"] + b + [f"\\v{i}"]

    def clause(self, ev):
        quants = []
        out = self.distribute(ev, quants)
        if ev.pred in ("be", "exist") and ev.manner:
            way = Entity("reserved", "n7", attrs=list(ev.manner))
            out = self.np(way, quants) + [tok("v", V["with"])] + out + [f"\\v{V['with']}"]
        for r in reversed(ev.rels):
            i = V[r.rel]
            if isinstance(r.landmark, Event):
                out = [tok("v", i, "x" if r.neg else "")] + self.clause(r.landmark) + [f"\\v{i}"] + out + [f"\\v{i}"]
            else:
                out = self.np(r.landmark, quants) + [tok("v", i, "x" if r.neg else "")] + out + [f"\\v{i}"]
        for restrictor, qid in reversed(quants):
            out = restrictor + [tok("v", qid)] + out + [f"\\v{qid}"]
        for name, other in ev.coord:
            out = self.logic(name, out, self.clause(other))
        return out

    def sentence(self, ev, words):
        self.words = words
        return " ".join(self.clause(ev))

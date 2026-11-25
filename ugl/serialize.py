"""Event/Entity graph -> UGL tokens.  The only place that knows token spelling
and the ordering rules of SPEC.md:

    clause := quantifier wrappers ( relation wrappers ( core ) ) + coordination
    core   := subject verb adverbs [object] (complement \\verb)*
    np     := head attrs count restrictive-clauses possessor
    attr   := a# | n#j | #num | L attr+ \\L attr+ \\L     (coordinated modifiers, L = and / or)
"""
import re

from .core import V, add_mods, literal, tok
from .ir import Entity, Event

WORD = re.compile(r"^(nv|n|v|a)(\d+)([a-z]*)$")


class Serializer:
    def __init__(self, lexicon):
        self.lex = lexicon
        self.unknown = set()
        self.words = []           # current sentence words, for sense choice
        self.top = None           # the event the current line is about

    # ------------------------------------------------------------- words
    def word(self, lemma, cls, mods="", pos=None, only=None):
        sent = (self.words, pos) if pos is not None and self.words else None
        r = self.lex.encode_word(lemma, cls, mods, sent=sent, only=only)
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
        if e.kind == "coord":                            # "they" = the same group again
            return self.coordinated([self.mention(m) for m in e.members], e.logic)
        if e.kind == "lex":
            # the sense was chosen where the entity was introduced (maybe in an
            # earlier sentence, whose words are no longer the context)
            head = e.head or self.word(e.lemma, "v" if e.cls == "nv" else "n")[0]
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
            out = self.mention(e)
            if e.quant:                              # "both" -> all of <what it refers to>
                quants.append((list(out), V[e.quant]))
            return out
        if e.kind == "coord":
            out = self.coordinated([self.np(m, quants) for m in e.members], e.logic)
            # modifiers of the whole group follow its last closing tag (SPEC rule 13)
            for r in e.restrictors:
                out += [tok("v", V["that"])] + self.clause(r) + [f"\\v{V['that']}"]
            if e.possessor is not None:
                p = self.np(e.possessor, quants)
                out += [add_mods(p[0], "y")] + p[1:]
            return out
        if e.kind == "pronoun":                          # unresolved and not handled by coref
            return ["n3t"]
        # a quantifier keeps "the": "all the dogs" (that group) is not "all dogs"
        mods = ("s" if e.plural else "") + e.det \
            + ("x" if e.neg else "") + ("q" if e.question else "")
        if e.kind == "reserved":
            head = [add_mods(e.lemma, mods)]
        elif e.kind == "name":
            head = self.proper(e.lemma, mods)
        elif e.kind == "num":
            head = [e.lemma]
        else:
            toks = self.word(e.lemma, "v" if e.cls == "nv" else "n", mods, pos=e.pos, only=e.senses)
            if e.cls == "nv" and toks[0].startswith("v"):
                toks[0] = "nv" + toks[0][1:]
            head = toks
            e.head = toks[0]
        out = list(head)
        for a in sorted(e.attrs, key=self.attr_rank):
            out += self.attr(a)
        if e.count:
            out.append(e.count)
        for r in e.restrictors:
            out += [tok("v", V["that"])] + self.clause(r) + [f"\\v{V['that']}"]
        if e.possessor is not None:
            p = self.np(e.possessor, quants)       # possessor letter on its head (v4y for a group)
            out += [add_mods(p[0], "y")] + p[1:]
        if e.quant:
            quants.append((list(head), V[e.quant]))
        return out

    @staticmethod
    def attr_rank(a):
        """Canonical modifier order: adjectives, then attribute nouns and names,
        then numbers, which share the count's slot right after them ("the 1902
        plague epidemic" = "the plague epidemic of 1902").  English order inside
        a rank is kept (sorted is stable).  The decoder places each kind by its
        own English rule, so this order is only notation."""
        kinds = {x.kind for x in a.leaves()}
        return 2 if kinds == {"num"} else 0 if kinds == {"adj"} else 1

    def attr(self, a):
        if a.kind == "coord":                  # "cell and car numbers": v4 n#j \v4 n#j \v4
            return self.coordinated([[t for x in member for t in self.attr(x)] for member in a.group], a.logic)
        if a.kind == "name":
            t = self.proper(a.lemma)
            return [add_mods(t[0], "j")] + t[1:]
        if a.kind == "num":
            return [a.lemma]
        # the parser's word class for a modifier is a guess ("bacterial" tagged as a
        # compound noun, "nil" as an adjective); a word the dictionary has only in
        # the other modifier class is that word, not an unknown literal
        if a.kind == "noun":
            if self.known(a.lemma, "n") or not self.known(a.lemma, "a"):
                t = self.word(a.lemma, "n", pos=a.pos)
                return [add_mods(t[0], "j")] + t[1:]
            return self.word(a.lemma, "a", a.mods, pos=a.pos)
        cls = "r" if a.adverb else "a"
        if not self.known(a.lemma, cls) and self.known(a.lemma, "n") and not a.adverb:
            t = self.word(a.lemma, "n", pos=a.pos)
            return [add_mods(t[0], "j" + a.mods)] + t[1:]
        return self.word(a.lemma, cls, a.mods, pos=a.pos)

    def known(self, lemma, cls):
        return bool(self.lex.senses(lemma, cls)) or self.lex.prefixed(lemma, cls) is not None

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
        # a re-mention may go bare only inside its own noun phrase ("speakers [that are]
        # in Slovakia"); split off as a sentence of its own it needs its "be":
        # "the tower is at a height of 21.6 m", not "there is the tower at ..."
        top_ref = ev is self.top and subj is not None and subj.ref is not None
        if ev.pred == "exist" and obj is None and not ev.pred_attrs and not ev.comps \
                and not ev.mods and self.indefinite(subj) and not top_ref:
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

    @staticmethod
    def coordinated(parts, logic="and"):
        """Coordinated noun phrase: v4 A \\v4 B \\v4  (nested for 3+ members)."""
        i = V[logic] if logic in ("or", "to") else V["and"]     # "to": a range of numbers
        out = parts[-1]
        for p in reversed(parts[:-1]):
            out = [tok("v", i)] + p + [f"\\v{i}"] + out + [f"\\v{i}"]
        return out

    def distribute(self, ev, quants):
        """The core clause.  Coordinated noun phrases stay one group (no
        distribution), so collective readings survive: "John and Mary met"."""
        return self.core(ev, ev.subj, ev.obj, quants)

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
        # stacked relations all modify the same clause, so their order carries no
        # meaning: a canonical order (relation number, then the landmark's tokens,
        # so two "in" phrases are ordered too) gives one UGL per meaning
        done = []
        for r in ev.rels:
            q = []
            land = self.clause(r.landmark) if isinstance(r.landmark, Event) else self.np(r.landmark, q)
            done.append((V[r.rel], land, r, q))
        done.sort(key=lambda d: (d[0], d[1]))
        for i, land, r, q in reversed(done):
            quants += q
            if isinstance(r.landmark, Event):
                out = [tok("v", i, "x" if r.neg else "")] + land + [f"\\v{i}"] + out + [f"\\v{i}"]
            else:
                out = land + [tok("v", i, "x" if r.neg else "")] + out + [f"\\v{i}"]
        for restrictor, qid in reversed(quants):
            out = restrictor + [tok("v", qid)] + out + [f"\\v{qid}"]
        for name, other in ev.coord:
            out = self.logic(name, out, self.clause(other))
        return out

    def sentence(self, ev, words):
        self.words = words
        self.top = ev
        return " ".join(self.clause(ev))

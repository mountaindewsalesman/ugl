"""Universal Dependencies tree -> Event/Entity graph (see ir.py).

Input: one sentence as a list of token dicts
    {form, upos, xpos, feats, head (1-based, 0 = root), deprel}
as produced by ugl.parser (or read from a treebank).

Each UD relation is handled in exactly one place:
  clause level  nsubj nsubj:pass csubj expl aux aux:pass cop obj iobj obl* advcl
                advmod ccomp xcomp mark conj cc parataxis
  noun level    det amod nummod nmod nmod:poss compound flat appos acl acl:relcl case
"""
import re

from .ir import Attr, Entity, Event, Rel, refer

try:
    from lemminflect import getLemma
except ImportError:  # pragma: no cover
    getLemma = None

# ------------------------------------------------------------------ tables
MODALS = {"will": "f", "shall": "f", "would": "h", "can": "c", "could": "ch", "may": "m",
          "might": "mh", "must": "u", "should": "uh", "ought": "uh",
          # semi-modals, after restructure() has turned them into auxiliaries
          "have to": "u", "have got to": "u", "be able to": "c", "be going to": "f",
          "be supposed to": "uh"}
CLITIC_LEMMA = {"n't": "not", "'m": "be", "'re": "be", "'ve": "have", "'ll": "will", "wo": "will",
                "ca": "can", "sha": "shall", "gon": "go", "na": "to", "ta": "to"}
NUM_WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen twenty".split())}
NUM_WORDS.update({"thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
                  "ninety": 90, "hundred": 100, "thousand": 1000, "million": 10**6, "billion": 10**9})
DEGREE = {"very": "w", "extremely": "w", "really": "w", "highly": "w", "so": "w", "too": "w",
          "quite": "w", "more": "r", "most": "z", "less": "l", "least": "lz", "not": "x", "n't": "x"}
QUANT = {"every": "all", "each": "all", "all": "all", "most": "most", "many": "many", "few": "few"}
# standalone quantifier words that refer back ("both are smelly", "most were sold")
QUANT_PRONOUNS = {"both": "all", "all": "all", "each": "all", "most": "most", "many": "many", "few": "few"}

# preposition (case + fixed words) -> relation name
PREP = {
    "at": "at", "in": "in", "inside": "in", "within": "in", "into": "to", "on": "on", "onto": "to",
    "upon": "on", "under": "under", "below": "under", "beneath": "under", "underneath": "under",
    "above": "above", "over": "above", "near": "near", "beside": "near", "next to": "near",
    "by": "by", "between": "between", "among": "between", "amongst": "between", "around": "around",
    "round": "around", "through": "through", "throughout": "through", "via": "through",
    "across": "across", "along": "along", "behind": "behind", "from": "from", "to": "to",
    "toward": "to", "towards": "to", "with": "with", "without": "with", "for": "for",
    "about": "about", "regarding": "about", "concerning": "about", "like": "like", "unlike": "like",
    "against": "against", "despite": "despite", "in spite of": "despite", "except": "except",
    "except for": "except", "as": "as", "of": "of", "before": "before", "after": "after",
    "during": "during", "until": "until", "till": "until", "since": "since",
    "because of": "cause", "due to": "cause", "out of": "out of", "outside": "out of",
    "in front of": "in front of", "according to": "from", "instead of": "except",
    "per": "for", "than": "against", "beyond": "out of", "past": "after", "off": "from",
    "up": "above", "down": "under", "out": "out of", "away from": "from", "upon": "on",
}
NEGATING = {"without", "unlike"}
TIME_SENSITIVE = {"at", "in", "on"}        # become "time" when the landmark is a time
# subordinator -> relation name (clause form)
SUBORD = {"because": "cause", "since": "cause", "as": "cause", "if": "if",
          "unless": "if", "when": "time", "whenever": "time", "once": "after", "while": "during",
          "before": "before", "after": "after", "until": "until", "till": "until",
          "although": "but", "though": "but", "whereas": "but", "even though": "but",
          "so that": "for", "in order to": "for", "to": "for", "that": "for", "where": "at",
          "wherever": "at", "like": "like", "as if": "like", "than": "against", "whether": "if"}
CC = {"and": "and", "or": "or", "but": "but", "nor": "or", "yet": "but", "&": "and", "plus": "and",
      "so": "cause"}                 # "A, so B": A causes B  (logic(cause, A, B))
DEICTIC = {"here": ("n5", "d", "at"), "there": ("n5", "e", "at"), "now": ("n6", "d", "time"),
           "then": ("n6", "e", "time"), "where": ("n5", "q", "at"), "when": ("n6", "q", "time"),
           "why": ("n8", "q", "cause"), "how": ("n7", "q", "with")}
TIME_NOUN_ADV = {"today", "yesterday", "tomorrow", "tonight", "recently", "earlier", "later",
                 "soon", "ago"}
# connective adverbs: on a joined clause ("It rained, so I stayed") they say
# how the two clauses relate.  logic(name, A, B): A is the earlier clause.
CONNECTIVE = {"so": "cause", "therefore": "cause", "thus": "cause", "hence": "cause",
              "consequently": "cause", "however": "but", "nevertheless": "but", "nonetheless": "but",
              "yet": "but", "then": "after", "afterwards": "after", "otherwise": "or"}
CLAUSE_DEPS = {"nsubj", "nsubj:pass", "csubj", "csubj:pass", "expl", "aux", "aux:pass", "cop",
               "obj", "iobj", "obl", "obl:tmod", "obl:npmod", "obl:agent", "advcl", "ccomp",
               "xcomp", "mark", "parataxis", "discourse", "vocative", "dislocated", "advmod",
               "compound:prt", "conj", "cc"}


class Tok:
    __slots__ = ("i", "form", "lower", "lemma", "upos", "xpos", "feats", "head", "deprel", "kids")

    def __init__(self, i, d):
        self.i, self.form, self.lower = i, d["form"], d["form"].lower()
        self.upos, self.xpos, self.deprel = d["upos"], d["xpos"], d["deprel"]
        self.feats = dict(kv.split("=", 1) for kv in d["feats"].split("|") if "=" in kv)
        self.head = d["head"] - 1          # -1 = root
        self.kids = []
        self.lemma = d.get("lemma") or None

    def f(self, k):
        return self.feats.get(k)

    def dep(self, *names):
        return [k for k in self.kids if k.deprel in names]

    def dep1(self, *names):
        d = self.dep(*names)
        return d[0] if d else None

    def __repr__(self):
        return f"{self.form}/{self.upos}/{self.deprel}"


def lemmatize(t, toks):
    if t.lemma:
        return t.lemma
    w = t.lower
    if w in CLITIC_LEMMA:
        return CLITIC_LEMMA[w]
    if w in ("'s", "’s") and t.upos == "AUX":
        head = toks[t.head] if t.head >= 0 else None
        return "have" if head is not None and head.xpos == "VBN" and t.deprel == "aux" else "be"
    if w in ("'d", "’d"):
        head = toks[t.head] if t.head >= 0 else None
        return "have" if head is not None and head.xpos == "VBN" else "would"
    if t.upos == "PROPN":
        return t.form
    if getLemma and t.upos in ("NOUN", "VERB", "ADJ", "ADV", "AUX"):
        r = getLemma(t.form if t.upos != "NOUN" or not t.form.isupper() else t.form, upos=t.upos)
        if r:
            return r[0].lower()
    return w


def build_tokens(sent):
    toks = [Tok(i, d) for i, d in enumerate(sent)]
    for t in toks:
        if t.head >= 0:
            toks[t.head].kids.append(t)
    for t in toks:
        t.lemma = lemmatize(t, toks)
    restructure(toks)
    return toks


# ------------------------------------------------------------------ tree normalisation
def semi_modal(v):
    """'have to X', 'is able to X', 'is going to X', 'is supposed to X':
    -> (the MODALS key, the xcomp X) when v heads such a construction, else None.
    These mean what a modal means (have to = must), so they get its letter, not a verb."""
    k = next((x for x in v.dep("xcomp") if x.upos in ("VERB", "AUX")
              and any(m.lower == "to" for m in x.dep("mark"))), None)
    if k is None or v.dep("obj", "ccomp"):
        return None
    aux = {a.lemma for a in v.dep("aux", "aux:pass", "cop")}
    if v.lemma == "have" and v.upos in ("VERB", "AUX"):
        return "have to", k
    if v.lemma == "get" and v.xpos == "VBN" and "have" in aux:
        return "have got to", k
    if v.lemma == "able" and "be" in aux:
        return "be able to", k
    if v.lemma == "go" and v.xpos == "VBG" and "be" in aux:
        return "be going to", k
    if v.lemma == "suppose" and v.xpos == "VBN" and "be" in aux:
        return "be supposed to", k
    return None


def restructure(toks):
    """Rewrite constructions into the one shape the analyzer reads, so each
    meaning has one code path.  Semi-modals become auxiliaries of their verb,
    exactly as UD already attaches real modals ("must go"):
        She has to leave  ->  leave <-aux- has(lemma "have to")"""
    for v in toks:
        sm = semi_modal(v)
        if sm is None:
            continue
        key, k = sm
        # k takes v's place in the tree
        k.head, k.deprel = v.head, v.deprel
        if v.head >= 0:
            sib = toks[v.head].kids
            sib[sib.index(v)] = k
        for c in v.kids:
            if c is k:
                continue
            c.head = k.i
            if c.deprel in ("cop", "aux:pass"):          # "is (able)": carries tense; now an aux of X
                c.deprel = "aux"
            if c.deprel == "aux" and c.lemma == "have" and key == "have got to":
                c.deprel = "fixed"                       # part of the semi-modal, not perfect aspect
            k.kids.append(c)
        if k.dep("aux:pass"):
            for c in k.dep("nsubj"):
                c.deprel = "nsubj:pass"                  # "has to be told": the subject is the patient
        for m in k.dep("mark"):
            if m.lower == "to":
                m.deprel = "fixed"
        v.kids, v.head, v.deprel, v.lemma = [], k.i, "aux", key
        k.kids.append(v)
        k.kids.sort(key=lambda x: x.i)


def subtree_start(t):
    """First word of t's phrase (punctuation not counted)."""
    return min([t.i] + [subtree_start(k) for k in t.kids if k.deprel != "punct"])


# number words that English also uses as plural nouns: "tens of kilometers"
NUM_NOUNS = {"ten": 10, "hundred": 100, "thousand": 1000, "million": 10**6, "billion": 10**9,
             "dozen": 12, "score": 20}
# amount words that head a noun phrase on their own ("not as much has been done")
# ("most" is left to QUANT_PRONOUNS: "most were rotten" = most of the apples)
AMOUNT = {"much": ("much", ""), "more": ("much", "r"), "less": ("much", "l"),
          "little": ("little", ""), "enough": ("enough", "")}


def number_parts(t):
    """t and the number words that build it with it ("two hundred thousand")."""
    parts = [k for k in t.kids if k.deprel in ("compound", "flat", "nummod") and
             (k.upos == "NUM" or k.lower in NUM_WORDS or k.lower in NUM_NOUNS)]
    return sorted(parts + [t], key=lambda x: x.i)


def number_value(words):
    """English number words -> int, or None: "two hundred thousand" -> 200000."""
    total = cur = 0
    seen = False
    for w in (p for x in words for p in x.replace(",", "").split("-") if p):
        if w.isdigit():
            cur += int(w)
        elif w in NUM_WORDS and NUM_WORDS[w] < 100:
            cur += NUM_WORDS[w]
        elif w in ("hundred",):
            cur = (cur or 1) * 100
        elif w in NUM_WORDS or w in NUM_NOUNS:
            v = NUM_WORDS.get(w) or NUM_NOUNS[w]
            total += (cur or 1) * v
            cur = 0
        else:
            return None
        seen = True
    return total + cur if seen else None


def number_literal(t, parts=None):
    """A number (with its number-word parts) -> literal.  Digits stay as
    written ("1,500" -> #1500, "3.5" -> #3.5); words become digits."""
    parts = parts or [t]
    v = number_value([p.lemma if p.upos == "NOUN" else p.lower for p in parts])
    if v is not None:
        return "#" + str(v)
    return "#" + "_".join(p.form for p in parts).replace(",", "")


class Analyzer:
    """lexicon is used only for semantic classes (person / time) and multiword lookup."""

    def __init__(self, lexicon):
        self.lex = lexicon

    # =============================================================== entry
    def sentence(self, sent, question=False):
        self.toks = build_tokens(sent)
        self.issues = []                         # what this sentence's UGL may have wrong (see Encoder)
        for t in self.toks:
            if t.deprel in ("dep", "orphan", "reparandum", "goeswith", "list"):
                self.issues.append(f"'{t.form}' has no clear grammatical role ({t.deprel}); left out")
        self.relheads = {}                       # token index of a relative word -> its head entity
        roots = [t for t in self.toks if t.head < 0]
        if not roots:
            return None
        ev = self.clause(roots[0])
        if question and not self._has_wh(ev):
            ev.mods += "q"
        return ev

    def _has_wh(self, ev):
        return any(t.lower in ("what", "who", "whom", "which", "where", "when", "why", "how")
                   for t in self.toks)

    # =============================================================== clause
    def clause(self, v, inherit_subj=None):
        """Build the Event headed by token v (a verb, or a nominal/adjectival predicate)."""
        cop = v.dep1("cop")
        ev = Event(pred="verb", lemma=v.lemma, pos=v.i)
        passive = bool(v.dep("aux:pass", "nsubj:pass", "csubj:pass", "obl:agent"))
        ev.passive = passive

        # ---- predicate type
        nominal = v.upos not in ("VERB", "AUX") or cop is not None
        if nominal and cop is None and not v.dep("nsubj", "nsubj:pass", "csubj", "expl"):
            # fragment ("The man who left.", a heading): just the noun phrase
            ev.pred, ev.lemma = "exist", "be"
            ev.subj = self.entity(v)
            for k in v.dep("advcl"):
                ev.rels.append(self.subordinate(k, ev))
            return ev
        if nominal:
            ev.pred, ev.lemma, ev.pos = "be", "be", (cop.i if cop is not None else v.i)
            if v.upos in ("ADJ", "ADV") or (v.upos == "VERB" and cop is not None):
                ev.pred_attrs.append(self.attr(v))
            elif v.dep("case") and cop is not None:
                # "John is in the house": be + relation
                ev.pred = "exist"
                ev.rels.append(self.rel_from_case(v))
            else:
                ev.obj = self.entity(v)
        elif v.lemma == "have" and not v.dep("xcomp", "ccomp"):
            ev.pred, ev.lemma = "have", "have"
        elif v.lemma == "be":
            ev.pred = "be" if v.dep("obj", "attr") else "exist"
        loose_prt = None
        if not nominal and ev.pred == "verb":
            prt = v.dep1("compound:prt")
            if prt is not None and self.lex.senses(v.lemma + " " + prt.lower, "v"):
                ev.lemma = v.lemma + " " + prt.lower           # a real phrasal verb: "look up"
            elif prt is not None:
                loose_prt = prt                                # meaning kept below: "sat on X", "came back"

        # ---- tense / aspect / mood / negation
        ev.mods += self.verb_mods(v, cop, passive)

        # ---- subject and object
        subj = v.dep1("nsubj", "csubj", "expl")
        psubj = v.dep1("nsubj:pass", "csubj:pass")
        agent = next((o for o in v.dep("obl", "obl:agent")
                      if any(c.lower == "by" for c in o.dep("case"))), None) if passive else None
        if passive:
            ev.obj = self.entity(psubj) if psubj is not None else ev.obj
            ev.subj = self.entity(agent) if agent is not None else (inherit_subj and Entity("reserved", "n4")) or Entity("reserved", "n4")
        elif subj is not None and subj.deprel == "expl":
            ev.subj = None   # "there is X": X arrives as nsubj in UD, handled below
            real = v.dep1("nsubj")
            ev.subj = self.entity(real) if real is not None and real is not subj else Entity("reserved", "n3", det="t")
        elif subj is not None and subj.deprel == "csubj":
            ev.subj = Entity("reserved", "n3", det="t", restrictors=[self.clause(subj)])
        elif subj is not None:
            ev.subj = self.entity(subj)
        elif inherit_subj is not None:
            ev.subj = refer(inherit_subj)
        elif not v.dep("aux", "aux:pass") and ("Imp" in (v.f("Mood") or "") or (v.head < 0 and v.xpos == "VB")):
            ev.subj = Entity("reserved", "n2")
            ev.mods += "o"
        else:
            ev.subj = Entity("reserved", "n4")
        if ev.pred in ("verb", "have", "be") and not nominal:
            o = v.dep1("obj")
            if o is not None:
                ev.obj = self.entity(o)
        if loose_prt is not None:
            rel = PREP.get(loose_prt.lower)
            if rel and ev.obj is not None:                     # "sat on the bank": a relation over the object
                ev.rels.append(Rel(rel, ev.obj))
                ev.obj = None
            else:                                              # "came back", "gave up": an adverb
                ev.manner.append(Attr(loose_prt.lower, "adj", "", loose_prt.i, adverb=True))
        io = v.dep1("iobj")
        if io is not None:
            ev.rels.append(Rel("to", self.entity(io)))

        # ---- adverbials, relations, subordinate clauses, complements
        for k in v.kids:
            d = k.deprel
            if d == "advmod":
                self.adverbial(k, ev)
            elif d in ("obl", "obl:tmod", "obl:npmod", "obl:agent") and k is not agent:
                ev.rels.append(self.rel_from_case(k))
            elif d == "advcl" and self.unmarked(k):
                # "..., observing X": English states no relation (time? cause? manner?),
                # so UGL states none either: the two clauses are simply joined
                sub = self.clause(k, inherit_subj=ev.subj)
                if not set(sub.mods) & set("pf"):          # a participle has no tense of its own
                    sub.mods += "".join(c for c in ev.mods if c in "pf")
                ev.coord.append(("and", sub))
            elif d == "advcl" and self.marker(k) == "so":
                # "A so B": A (the main clause) causes B
                ev.coord.append(("cause", self.clause(k, inherit_subj=ev.subj)))
            elif d == "advcl":
                ev.rels.append(self.subordinate(k, ev))
            elif d in ("ccomp", "xcomp"):
                ev.comps.append(self.complement(k, ev))
        # ---- coordination
        for k in v.dep("conj"):
            if k.upos in ("VERB", "AUX", "ADJ") or k.dep("nsubj", "nsubj:pass", "cop"):
                ev.coord.append((self.link(k), self.clause(k, inherit_subj=ev.subj)))
        for k in v.dep("parataxis"):
            if k.upos in ("VERB", "AUX") or k.dep("nsubj", "cop"):
                ev.coord.append((self.link(k), self.clause(k)))
        return ev

    def verb_mods(self, v, cop, passive):
        m = ""
        auxes = v.dep("aux", "aux:pass") + ([cop] if cop is not None else [])
        finite = [v] + auxes
        for a in v.dep("aux", "aux:pass"):
            l = a.lemma
            if l in MODALS:
                m += MODALS[l]
            elif l == "have" and v.xpos == "VBN":
                m += "k"
            elif l == "be" and v.xpos == "VBG" and a.deprel == "aux":
                m += "g"
        if any(a.lemma == "go" and v.xpos == "VB" for a in v.dep("aux")):
            m += "f"
        if any(t.f("Tense") == "Past" and (t.f("VerbForm") in (None, "Fin")) and t.xpos in ("VBD",)
               for t in finite):
            m += "p"
        if any(k.lower in ("not", "n't", "never") for k in v.dep("advmod")):
            m += "x"
        return m

    def adverbial(self, k, ev):
        w = k.lower
        head = self.toks[k.head] if k.head >= 0 else None
        if w in CONNECTIVE and head is not None and head.deprel in ("conj", "parataxis"):
            return                                     # used as the clause link (see link())
        if w in ("not", "n't", "never"):
            return                                     # handled in verb_mods
        if k.i in self.relheads and w in DEICTIC:     # "the place where I live": at <the place>
            ev.rels.append(Rel(DEICTIC[w][2], refer(self.relheads[k.i])))
            return
        if w in DEICTIC:
            if k.head >= 0 and self.toks[k.head].deprel in ("advcl", "acl:relcl") and DEICTIC[w][1] == "q":
                return                                 # subordinator ("when he left"), used by parent
            tokn, mod, rel = DEICTIC[w]
            if mod == "q" and self.embedded(k):
                mod = "t"                              # "I know where X" = the place at which X
            ev.rels.append(Rel(rel, Entity("reserved", tokn + mod)))
            return
        if w in TIME_NOUN_ADV and self.lex.senses(w, "n"):
            ev.rels.append(Rel("time", Entity("lex", w, pos=k.i)))
            return
        for o in k.dep("obl", "nmod", "obl:npmod"):     # "away [from the stock]"
            ev.rels.append(self.rel_from_case(o))
        if w in ("all", "both", "each") and ev.subj is not None and not ev.subj.quant:
            ev.subj.quant = "all"                      # floating quantifier: "the dogs all barked"
            return
        if w in ("also", "too", "just", "even", "only", "then", "so", "still", "already"):
            return                                     # discourse particles: no content in UGL v0
        ev.manner.append(self.attr(k))

    EMBEDDED = {"ccomp", "xcomp", "advcl", "acl", "acl:relcl", "csubj", "csubj:pass"}

    def embedded(self, t):
        """Is token t inside a subordinate / complement clause?"""
        seen = set()
        while t.head >= 0 and t.i not in seen:
            seen.add(t.i)
            if t.deprel in self.EMBEDDED:
                return True
            t = self.toks[t.head]
        return False

    def rel_from_case(self, k):
        """obl / nmod with its case marker -> Rel(relation, landmark)."""
        case = [c for c in k.dep("case") if c.upos in ("ADP", "SCONJ", "ADV", "PART")]
        words = []
        for c in case:
            words.append(c.lower)
            words += [f.lower for f in c.dep("fixed")]
        prep = " ".join(words)
        land = self.entity(k)
        if not prep:
            rel = "time" if k.deprel in ("obl:tmod", "nmod:tmod") or self.is_time(k, land) else "of"
            return Rel(rel, land)
        rel = PREP.get(prep) or PREP.get(words[0])
        if rel is None:
            self.issues.append(f"preposition '{prep}' not in the relation table; read as 'of'")
            rel = "of"
        if rel in TIME_SENSITIVE and self.is_time(k, land):
            rel = "time"
        return Rel(rel, land, neg=prep in NEGATING)

    def is_time(self, k, ent):
        if ent.kind == "num":
            return True
        if ent.kind == "lex":
            return self.lex.in_class(ent.lemma, "time")
        if ent.kind == "name":
            return self.lex.in_class(ent.lemma.lower(), "time")
        return False

    @staticmethod
    def link(k):
        """How a joined clause k relates to the previous one: its connective adverb
        ("so" = cause, "then" = after), else its conjunction, else and.
        logic "after" with A = earlier clause reads "B after A" = "A, then B"."""
        for a in k.dep("advmod"):
            if a.lower in CONNECTIVE:
                return CONNECTIVE[a.lower]
        cc = k.dep1("cc")
        return CC.get(cc.lower if cc is not None else "and", "and")

    @staticmethod
    def unmarked(k):
        """An adverbial clause with no subordinator and no wh-word: a bare
        participle or verbless phrase ("Running, he fell")."""
        return (not k.dep("mark") and k.upos in ("VERB", "AUX") and k.xpos in ("VBG", "VBN")
                and not any(a.lower in ("when", "where", "while", "whenever") for a in k.dep("advmod")))

    @staticmethod
    def marker(k):
        return " ".join(w for m in k.dep("mark") for w in [m.lower] + [f.lower for f in m.dep("fixed")])

    def subordinate(self, k, main):
        """advcl -> clause-form relation: Rel(name, Event)."""
        marks = k.dep("mark")
        words = []
        for m in marks:
            words.append(m.lower)
            words += [f.lower for f in m.dep("fixed")]
        if not words:
            wh = next((a for a in k.dep("advmod") if a.lower in ("when", "where", "while", "whenever")), None)
            if wh is not None:
                words = [wh.lower]
        key = " ".join(words)
        if words:
            rel = (SUBORD.get(key) or SUBORD.get(words[0]) or PREP.get(key) or PREP.get(words[0]) or "of")
        else:
            rel = "time"                               # bare participial clause: "Running, he fell"
        sub = self.clause(k, inherit_subj=main.subj)
        if key == "unless":
            sub.mods += "x"
        return Rel(rel, sub)

    def complement(self, k, main):
        if k.deprel == "xcomp":
            ctrl = main.obj if main.obj is not None else main.subj
            if k.upos in ("ADJ", "NOUN", "PROPN") and not k.dep("cop"):
                # "made him happy", "seems happy", "elected him president"
                e = Event(pred="be", lemma="be", subj=refer(ctrl) if ctrl else None)
                if k.upos == "ADJ":
                    e.pred_attrs.append(self.attr(k))
                else:
                    e.obj = self.entity(k)
                return e
            return self.clause(k, inherit_subj=ctrl)
        return self.clause(k)

    # =============================================================== nouns
    def attr(self, t):
        mods = ""
        deg = t.f("Degree")
        if deg == "Cmp":
            mods += "r"
        elif deg == "Sup":
            mods += "z"
        for a in t.dep("advmod"):
            mods += DEGREE.get(a.lower, "")
        lemma = t.lemma
        if t.upos == "VERB":       # participle as adjective: try the surface form first
            lemma = t.lower if self.lex.senses(t.lower, "a") else t.lemma
        return Attr(lemma, "adj", mods, t.i, adverb=t.upos == "ADV")

    def entity(self, t):
        return self.coordinate(t, self.head_entity(t))

    def coordinate(self, t, e):
        """'X and Y': a coordination node around X; X keeps its identity, so its
        own relative clauses still refer to X alone."""
        conj = [c for c in t.dep("conj") if c.upos in ("NOUN", "PROPN", "PRON", "NUM") and not c.dep("cop", "nsubj")]
        if not conj:
            return e
        cc = next((g for c in conj for g in c.dep("cc")), None)
        members = [e] + [self.head_entity(c) for c in conj]
        return Entity("coord", members=members, logic=CC.get(cc.lower if cc else "and", "and"), pos=e.pos)

    QUANT_OF = {"all": "all", "each": "all", "both": "all", "every": "all", "most": "most",
                "many": "many", "few": "few"}

    def head_entity(self, t):
        # "all/most/many of X": the quantifier ranges over X, whatever tag the parser gave the word
        of = next((c for c in t.dep("nmod", "obl") if any(k.lower == "of" for k in c.dep("case"))), None)
        if t.lower in self.QUANT_OF and of is not None:
            inner = self.entity(of)
            inner.quant = self.QUANT_OF[t.lower]
            return inner
        if t.lemma in NUM_NOUNS and t.upos == "NOUN":
            # "tens of kilometers": the counted thing, with a count in tens
            lit = number_literal(t, number_parts(t)) + ("s" if t.f("Number") == "Plur" else "")
            if of is not None:
                inner = self.entity(of)
                inner.count = lit
                return inner
            return Entity("reserved", "n3", plural=True, count=lit, pos=t.i)
        # a quantity word standing as a noun phrase by itself, whatever tag it got
        alone = (t.upos in ("ADJ", "ADV", "PRON", "DET", "NOUN") and t.deprel not in ("amod", "advmod", "det")
                 and not t.dep("nsubj", "cop", "det"))
        if t.lower in QUANT_PRONOUNS and alone:      # "most were rotten", "many came"
            return self.pronoun(t)
        if t.lower in AMOUNT and alone:
            # "not as much has been done": an amount
            word, mods = AMOUNT[t.lower]
            a = self.attr(t)
            return Entity("reserved", "n9", attrs=[Attr(word, "adj", mods + a.mods, t.i)], pos=t.i)
        if t.upos == "PRON" or (t.upos == "DET" and t.deprel not in ("det",)):
            e = self.pronoun(t)
            if e.kind == "reserved":                   # "someone who can help", "something new"
                self.np_mods(t, e)
            return e
        if t.upos == "NUM" or (t.xpos == "CD"):
            parts = number_parts(t)
            e = Entity("num", number_literal(t, parts), pos=t.i)
            self.np_mods(t, e, skip=set(id(p) for p in parts))
            return e
        if t.upos == "PROPN":
            parts = [c for c in t.dep("flat", "compound") if c.upos == "PROPN"]
            span = sorted(parts + [t], key=lambda x: x.i)
            e = Entity("name", " ".join(x.form for x in span), pos=t.i, is_person=False)
            self.np_mods(t, e, skip=set(id(p) for p in parts))
            return e
        e = Entity("lex", t.lemma, pos=t.i)
        if t.xpos == "VBG" or t.upos == "VERB":
            e.cls = "nv"
        e.plural = t.f("Number") == "Plur"
        # multiword noun ("ice cream") before treating compounds as attributes
        comps = [c for c in t.dep("compound") if c.i < t.i and c.upos == "NOUN"]
        skip = set()
        if comps:
            joined = " ".join([c.lemma for c in comps] + [t.lemma])
            if self.lex.senses(joined, "n"):
                e.lemma, skip = joined, set(id(c) for c in comps)
        e.is_person = self.lex.in_class(e.lemma, "person")
        self.np_mods(t, e, skip)
        return e

    def pronoun(self, t):
        w, pt = t.lower, t.f("PronType")
        person = t.f("Person")
        plural = t.f("Number") == "Plur"
        if t.i in self.relheads:
            return refer(self.relheads[t.i])       # relative pronoun = its head noun
        if pt == "Rel":
            e = Entity("pronoun", w, pos=t.i)      # free relative without a head
            e.person = 3
            return e
        if person == "1":
            return Entity("reserved", "n1", plural=plural)
        if person == "2":
            return Entity("reserved", "n2")
        if pt == "Int" or w in ("what", "who", "whom", "which"):
            tokn = "n4" if w in ("who", "whom") else "n3"
            if self.embedded(t):                       # "do what I want" = the thing that I want
                return Entity("reserved", tokn, det="t", pos=t.i)
            return Entity("reserved", tokn, question=True)
        if pt == "Dem" or w in ("this", "that", "these", "those"):
            return Entity("reserved", "n3", det="d" if w in ("this", "these") else "e", plural=w in ("these", "those"))
        if w in ("something", "anything"):
            return Entity("reserved", "n3")
        if w in ("someone", "somebody", "anyone", "anybody"):
            return Entity("reserved", "n4")
        if w in ("nothing", "none"):
            return Entity("reserved", "n3", neg=True)
        if w in ("nobody", "noone", "no-one"):
            return Entity("reserved", "n4", neg=True)
        if w in QUANT_PRONOUNS:
            # "...cats and dogs because both are smelly": coreference finds what
            # "both" ranges over; unresolved, it falls back to "all things"
            return Entity("pronoun", w, pos=t.i, person=3, plural=True, quant=QUANT_PRONOUNS[w])
        if w in ("everything", "everyone", "everybody"):
            return Entity("reserved", "n4" if w.startswith("every") and w != "everything" else "n3", quant="all")
        if w in ("most", "many", "few"):               # "most of these wars": quantifier over the of-phrase
            return Entity("reserved", "n3", quant=QUANT[w])
        # third-person personal / possessive / reflexive: coreference decides
        e = Entity("pronoun", w, pos=t.i, person=3, plural=plural, gender=t.f("Gender"))
        return e

    def np_mods(self, t, e, skip=()):
        for c in t.kids:
            if id(c) in skip:
                continue
            d = c.deprel
            if d in ("det", "det:predet"):
                w = c.lower
                if w == "the":
                    e.det = "t"
                elif w in ("this", "these"):
                    e.det = "d"
                elif w in ("that", "those"):
                    e.det = "e"
                elif w == "no":
                    e.neg = True
                elif w in ("which", "what"):
                    e.question = True
                elif w in QUANT:
                    e.quant = QUANT[w]
                elif c.f("Poss") == "Yes" or c.upos == "PRON":
                    e.possessor = self.pronoun(c)
                elif w in ("another", "other"):
                    e.attrs.append(Attr("other", "adj", "", c.i))
            elif d == "amod":
                if c.upos == "VERB" and not self.lex.senses(c.lower, "a"):
                    e.restrictors.append(self.participle(c, e))
                else:
                    e.attrs.append(self.attr(c))
            elif d == "nummod":
                e.count = number_literal(c, number_parts(c))
            elif d == "compound":
                if c.upos == "PROPN":
                    e.attrs.append(Attr(c.form, "name", "", c.i))
                elif c.upos == "NUM":
                    e.attrs.append(Attr(number_literal(c), "num", "", c.i))
                elif c.upos in ("NOUN", "ADJ", "X"):
                    e.attrs.append(Attr(c.lemma, "noun" if c.upos != "ADJ" else "adj", "", c.i))
            elif d == "nmod:poss":
                e.possessor = self.entity(c)
            elif d in ("nmod", "nmod:tmod", "nmod:npmod"):
                r = self.rel_from_case(c)
                if r.rel == "of":
                    e.possessor = r.landmark
                else:
                    e.restrictors.append(Event(pred="exist", subj=refer(e), rels=[r]))
            elif d == "appos":
                e.restrictors.append(Event(pred="be", lemma="be", subj=refer(e), obj=self.entity(c),
                                           aside=self.after_comma(c)))
            elif d == "acl:relcl":
                e.restrictors.append(self.relative(c, e))
                e.restrictors[-1].aside = self.after_comma(c)
            elif d == "acl":
                e.restrictors.append(self.participle(c, e))
                e.restrictors[-1].aside = self.after_comma(c)

    def after_comma(self, c):
        """Is phrase c set off by a comma right before it ("my mother, who lives
        in Ohio," / "Paris, the capital,")?  Then it is non-restrictive: a
        separate statement about its head, not part of what picks the head out."""
        start = subtree_start(c)
        return start > 0 and self.toks[start - 1].lower in (",", ";", "(", "—", "–")

    def relative(self, c, head):
        """acl:relcl.  Relative words (who/which/that, and where/when/why) are
        registered BEFORE the clause is read, so they resolve to the head as soon
        as they are met, and anything that inherits them (the subject of "to do"
        in "someone who can manage to do X") refers to the head too."""
        relwords = [k for k in c.kids if k.f("PronType") == "Rel" or
                    (k.lower in ("who", "whom", "which", "that", "whose") and
                     k.deprel in ("nsubj", "obj", "nsubj:pass", "obl", "nmod:poss"))]
        relwords += [k for k in c.dep("advmod") if k.lower in ("where", "when", "why")]
        for k in relwords:
            self.relheads[k.i] = head
        ev = self.clause(c)
        ref = lambda: refer(head)
        if not relwords:
            # "the book I read": gap in object position (or subject if none)
            if ev.obj is None and ev.subj is not None and ev.subj.ref is not head and ev.pred == "verb":
                ev.obj = ref()
            elif ev.subj is None:
                ev.subj = ref()
        return ev

    def participle(self, c, head):
        """acl / participial amod: 'the man sitting there', 'particles released from rain'."""
        ev = self.clause(c)
        r = refer(head)
        if c.xpos == "VBN":                       # passive participle: head is the object
            if ev.obj is None:
                ev.obj = r
            if ev.subj is None or (ev.subj.kind == "reserved" and ev.subj.lemma == "n3"):
                agent = next((rel for rel in ev.rels if rel.rel == "by"), None)
                if agent is not None:
                    ev.rels.remove(agent)
                    ev.subj = agent.landmark
                else:
                    ev.subj = Entity("reserved", "n4")
        else:
            ev.subj = r
        return ev

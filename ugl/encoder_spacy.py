"""LEGACY English -> UGL encoder on spaCy (kept for comparison; the default is
ugl.encoder, which uses UGL's own parser).  Needs `pip install ugl[spacy]`.


spaCy gives a dependency tree; a fixed set of rules rewrites it:
    subject verb [adverbs] [object] [clause complements \\verb]
wrapped (outermost first, in sentence order) by
    landmark  RELATION  <clause>  \\RELATION      prepositional phrases
    RELATION <A> \\RELATION <B> \\RELATION        subordinate clauses / logic
    noun      QUANT     <clause>  \\QUANT         every/most/many/few
Pronouns are replaced by the noun they refer to (simple recency heuristic);
I/me -> n1 (speaker), you -> n2 (listener).
"""
import re

from .core import (V, PREP_TO_REL, NEGATING_PREPS, TIME_PREPS, MARK_TO_REL, CC_TO_LOGIC,
                   QUANT_DETS, MODAL_MODS, add_mods, tok)

NUM_WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen twenty".split())}
NUM_WORDS.update({"thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
                  "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1000,
                  "million": 1000000, "billion": 1000000000})

TIME_ADV = {"now": ("n6", "d"), "then": ("n6", "e"), "today": None, "yesterday": None,
            "tomorrow": None, "tonight": None, "recently": None}
PLACE_ADV = {"here": ("n5", "d"), "there": ("n5", "e")}
WH = {"what": ("n3", "q"), "who": ("n4", "q"), "whom": ("n4", "q"), "which": ("n3", "q"),
      "where": ("n5", "q"), "when": ("n6", "q"), "how": ("n7", "q"), "why": ("n8", "q")}
WH_REL = {"where": V["at"], "when": V["time"], "how": V["with"], "why": V["cause"]}
INTENS = {"very": "w", "extremely": "w", "really": "w", "highly": "w", "so": "w", "too": "w",
          "more": "r", "most": "z", "less": "l", "least": "lz", "not": "x"}
TIME_WORDS = {"year", "month", "week", "day", "hour", "minute", "second", "century", "decade",
              "morning", "evening", "night", "afternoon", "noon", "midnight", "time", "moment",
              "period", "era", "age", "season", "spring", "summer", "autumn", "fall", "winter",
              "january", "february", "march", "april", "may", "june", "july", "august",
              "september", "october", "november", "december", "monday", "tuesday", "wednesday",
              "thursday", "friday", "saturday", "sunday", "o'clock", "today", "yesterday",
              "tomorrow", "weekend", "date", "birth", "death", "war"}
PERSONAL = {"i", "me", "my", "mine", "myself"}
WE = {"we", "us", "our", "ours", "ourselves"}
YOU = {"you", "your", "yours", "yourself", "yourselves"}
THIRD = {"he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
         "they", "them", "their", "theirs", "themselves"}


class Mention:
    def __init__(self, tokens, person, plural):
        self.tokens, self.person, self.plural = tokens, person, plural


_ASIDE = re.compile(r"\s*[\(\[][^\(\)\[\]]*[\)\]]")


def strip_asides(text):
    """Drop (parenthetical) and [bracketed] asides, innermost first."""
    prev = None
    while prev != text:
        prev, text = text, _ASIDE.sub("", text)
    return text


class SpacyEncoder:
    def __init__(self, lexicon, nlp=None, strip_parens=True):
        import spacy
        self.lex = lexicon
        self.nlp = nlp or spacy.load("en_core_web_md")
        self.strip_parens = strip_parens
        self.memory = []          # recent Mentions, newest last
        self.unknown = set()

    # ================================================================ API
    def encode(self, text, keep_memory=False):
        if not keep_memory:
            self.memory = []
        if self.strip_parens:
            text = strip_asides(text)
        doc = self.nlp(text)
        out = []
        for sent in doc.sents:
            if not any(t.is_alpha for t in sent):
                continue
            try:
                out.append(" ".join(self.sentence(sent)))
            except Exception as e:  # never crash on odd input; mark it
                out.append(f"n?[unparsed:{sent.text.strip()[:40].replace(' ', '_')}]")
        return out

    # ============================================================ helpers
    def context(self, t):
        return [w.lemma_.lower() for w in t.sent if w.is_alpha and not w.is_stop]

    def word(self, t, cls, mods="", lemma=None):
        lemma = (lemma or t.lemma_).lower()
        r = self.lex.encode_word(lemma, cls, mods, self.context(t), sent=(t.sent, t.i))
        if r is None and cls == "a":       # participle used as adjective: "broken", "running"
            r = self.lex.encode_word(t.lower_, "a", mods, self.context(t), sent=(t.sent, t.i))
        if r is None:
            self.unknown.add((lemma, cls))
            r = [f"{cls}?{lemma.replace(' ', '_')}"]
        return r

    def is_person(self, t):
        if t.ent_type_ in ("PERSON", "NORP"):
            return True
        if t.pos_ == "NOUN":
            return self.lex.is_person(t.lemma_)
        return False

    def remember(self, t, toks):
        if t.pos_ in ("NOUN", "PROPN") and toks:
            plural = "Plur" in t.morph.get("Number")
            bare = [toks[0]] if not toks[0].startswith("'") else [toks[0]]
            if t.pos_ == "NOUN":
                bare = [add_mods(re.sub(r"[a-z]+$", "", toks[0]) if re.match(r"^n\d", toks[0]) else toks[0], "t" + ("s" if plural else ""))]
                bare += [x for x in toks[1:] if x.startswith("a")]  # keep compound expansion adjectives
            self.memory.append(Mention(bare, self.is_person(t), plural))
            self.memory = self.memory[-30:]

    def resolve(self, t):
        w = t.lower_
        if w in PERSONAL:
            return ["n1"]
        if w in WE:
            return ["n1s"]
        if w in YOU:
            return ["n2"]
        if w in THIRD:
            if t.dep_ == "expl":
                return ["n3t"]
            want_person = w in {"he", "him", "his", "himself", "she", "her", "hers", "herself"}
            want_plural = w.startswith("the")
            # never resolve to a noun inside the same clause as subject->object (crude reflexive rule)
            for m in reversed(self.memory):
                if want_person and m.person and not m.plural:
                    return list(m.tokens)
                if not want_person and m.plural == want_plural and (w.startswith("the") or not m.person):
                    return list(m.tokens)
            if want_person:   # no known person: fall back to the latest name
                for m in reversed(self.memory):
                    if m.tokens and m.tokens[0].startswith("'"):
                        return list(m.tokens)
            return ["n4"] if want_person else (["n3ts"] if want_plural else ["n3t"])
        if w in ("this", "these"):
            return ["n3d" + ("s" if w == "these" else "")]
        if w in ("that", "those"):
            return ["n3e" + ("s" if w == "those" else "")]
        if w in ("something", "anything"):
            return ["n3"]
        if w in ("someone", "somebody", "anyone", "anybody"):
            return ["n4"]
        if w in ("nothing",):
            return ["n3x"]
        if w in ("nobody", "noone"):
            return ["n4x"]
        if w in ("everything", "everyone", "everybody"):
            return ["n3" if w == "everything" else "n4"]  # quantifier added by caller
        if w in WH:
            return [WH[w][0] + WH[w][1]]
        return None

    # ================================================================ NP
    def np(self, t, quants, rel_sub=None):
        """Noun phrase rooted at token t -> list of UGL tokens.
        quants collects (restrictor tokens, quant verb id) to wrap the clause.
        rel_sub maps token index -> replacement tokens (relative pronouns)."""
        if rel_sub and t.i in rel_sub:
            return list(rel_sub[t.i])
        if t.pos_ == "PRON" or t.lower_ in WH:
            r = self.resolve(t)
            if r is not None:
                if t.lower_ in ("everything", "everyone", "everybody"):
                    quants.append((list(r), V["all"]))
                return r
        if t.pos_ == "NUM" or t.like_num:
            return [self.number(t)]
        if t.pos_ == "PROPN":
            name, parts = self.name_of(t)
            out = [name]
            out += self.np_mods(t, quants, rel_sub, skip=parts)
            self.remember(t, out[:1])
            return out

        mods = ""
        if "Plur" in t.morph.get("Number"):
            mods += "s"
        lemma = t.lemma_.lower()
        # multiword noun in WordNet ("ice cream")
        comp = [c for c in t.children if c.dep_ == "compound" and c.i < t.i]
        skip = set()
        if comp:
            joined = " ".join([c.lemma_.lower() for c in comp] + [lemma])
            if self.lex.senses(joined, "n"):
                lemma, skip = joined, {c.i for c in comp}
        quant = None
        for c in t.children:
            if c.dep_ in ("det", "predet"):
                d = c.lower_
                if d == "the":
                    mods += "t"
                elif d in ("this", "these"):
                    mods += "d"
                elif d in ("that", "those"):
                    mods += "e"
                elif d == "no":
                    mods += "x"
                elif d in ("which", "what"):
                    mods += "q"
                elif d in QUANT_DETS:
                    quant = QUANT_DETS[d]
        if t.pos_ == "VERB" or (t.tag_ == "VBG"):
            head = ["nv" + self.word(t, "v")[0][1:]]
        else:
            head = self.word(t, "n", mods, lemma=lemma)
        out = head + self.np_mods(t, quants, rel_sub, skip)
        if quant:
            quants.append((list(head), quant))
        self.remember(t, out)
        return out

    def np_mods(self, t, quants, rel_sub, skip=()):
        adjs, comps, nums, rels, poss = [], [], [], [], []
        for c in t.children:
            if c.i in skip:
                continue
            if c.dep_ == "amod" or (c.dep_ == "advmod" and c.pos_ == "ADJ"):
                adjs += self.adj(c)
            elif c.dep_ == "compound":
                if c.pos_ == "PROPN":
                    comps.append(add_mods("'" + c.text.replace("'", "") + "'", "j"))
                else:
                    e = self.word(c, "n")
                    comps += [add_mods(e[0], "j")] + e[1:]
            elif c.dep_ == "nummod":
                nums.append(self.number(c))
            elif c.dep_ == "poss":
                p = self.np(c, quants, rel_sub)
                poss += [add_mods(p[0], "y")] + p[1:]
            elif c.dep_ == "prep":
                obj = [g for g in c.children if g.dep_ == "pobj"]
                if not obj:
                    continue
                if c.lower_ == "of":
                    p = self.np(obj[0], quants, rel_sub)
                    poss += [add_mods(p[0], "y")] + p[1:]
                else:
                    rid = self.rel_for(c, obj[0])
                    neg = "x" if c.lower_ in NEGATING_PREPS else ""
                    land = self.np(obj[0], quants, rel_sub)
                    head_ref = self.bare_ref(t)
                    rels += [tok("v", V["that"])] + land + [tok("v", rid, neg)] + head_ref + [f"\\v{rid}", f"\\v{V['that']}"]
            elif c.dep_ in ("relcl", "acl"):
                sub = dict(rel_sub or {})
                for g in c.subtree:
                    if g.tag_ in ("WDT", "WP", "WP$") or (g.lower_ == "that" and g.dep_ in ("nsubj", "dobj", "nsubjpass", "pobj")):
                        sub[g.i] = self.bare_ref(t)
                subj = None
                if c.dep_ == "acl" or not any(g.dep_ in ("nsubj", "nsubjpass") for g in c.children):
                    subj = self.bare_ref(t)
                clause = self.clause(c, subj_override=subj, rel_sub=sub)
                rels += [tok("v", V["that"])] + clause + [f"\\v{V['that']}"]
            elif c.dep_ == "appos":
                q2 = []
                app = self.np(c, q2, rel_sub)
                rels += [tok("v", V["that"])] + self.bare_ref(t) + [tok("v", V["be"])] + app + [f"\\v{V['that']}"]
        return adjs + comps + nums + rels + poss

    def name_of(self, t):
        parts = [c for c in t.children if c.dep_ in ("compound", "flat") and c.pos_ == "PROPN" and c.i < t.i]
        span = t.doc[min([c.i for c in parts] + [t.i]): t.i + 1]
        return "'" + "_".join(w.text for w in span).replace("'", "") + "'", {c.i for c in parts}

    @staticmethod
    def bare(toks):
        """Re-mention form of an encoded noun phrase: head only, made definite."""
        if not toks:
            return ["n3t"]
        h = toks[0]
        m = re.match(r"^(nv|n)(\d+)([a-z]*)$", h)
        if m and int(m.group(2)) >= 100 or (m and m.group(1) == "nv"):
            return [tok(m.group(1), m.group(2), "t" + ("s" if "s" in m.group(3) else ""))]
        if m:
            return [tok(m.group(1), m.group(2), "".join(c for c in m.group(3) if c in "sq"))]
        if h.startswith("'"):
            return [h.rsplit("'", 1)[0] + "'"]
        return [h]

    def bare_ref(self, t):
        """Re-mention of a noun without its modifiers (used instead of pronouns)."""
        if t.pos_ == "PROPN":
            return [self.name_of(t)[0]]
        if t.pos_ == "PRON":
            return self.resolve(t) or ["n3"]
        e = self.word(t, "n", "t" + ("s" if "Plur" in t.morph.get("Number") else ""))
        return e

    def adj(self, c):
        mods = ""
        deg = c.morph.get("Degree")
        if "Cmp" in deg:
            mods += "r"
        if "Sup" in deg:
            mods += "z"
        for g in c.children:
            if g.dep_ in ("advmod", "neg") and g.lower_ in INTENS:
                mods += INTENS[g.lower_]
        cls_lemma = c.lemma_.lower()
        return self.word(c, "a", mods, lemma=cls_lemma)

    def number(self, t):
        w = t.lower_.replace(",", "")
        if w in NUM_WORDS:
            return f"#{NUM_WORDS[w]}"
        return "#" + re.sub(r"\s+", "_", t.text)

    def rel_for(self, prep, obj):
        p = prep.lower_
        if p in TIME_PREPS and (obj.ent_type_ in ("DATE", "TIME") or obj.lemma_.lower() in TIME_WORDS
                                or obj.like_num):
            return V["time"]
        if p == "by" and obj.ent_type_ in ("DATE", "TIME"):
            return V["before"]
        return PREP_TO_REL.get(p, V["of"])

    # ============================================================ clause
    def sentence(self, sent):
        root = sent.root
        toks = self.clause(root)
        if sent.text.strip().endswith("?") and not any(t.lower_ in WH for t in sent):
            toks = self.mark_main_verb(toks, "q")
        return toks

    def mark_main_verb(self, toks, mods):
        # first verb token that is not a reserved wrapper
        for i, x in enumerate(toks):
            m = re.match(r"^v(\d+)", x)
            if m and int(m.group(1)) >= 100 or x in ("v1", "v2") or (m and int(m.group(1)) in (1, 2)):
                toks[i] = add_mods(x, mods)
                break
        return toks

    def clause(self, v, subj_override=None, rel_sub=None, subj_force=None, obj_force=None, _nodist=False):
        quants = []
        wrappers = []      # (kind, tokens...) applied innermost-last
        logic = []
        children = list(v.children)

        # --- verb group: tense / aspect / mood / negation
        vmods = ""
        passive = False
        tense = v.morph.get("Tense")
        aux = [c for c in children if c.dep_ in ("aux", "auxpass")]
        for a in aux:
            w = a.lower_
            if w in MODAL_MODS:
                vmods += MODAL_MODS[w]
            elif a.lemma_ == "have" and v.tag_ == "VBN":
                vmods += "k"
            elif a.lemma_ == "be" and v.tag_ == "VBG":
                vmods += "g"
            elif a.lemma_ == "be" and a.dep_ == "auxpass":
                passive = True
            elif a.lemma_ == "going":
                vmods += "f"
            if "Past" in a.morph.get("Tense"):
                vmods += "p"
        if not aux and "Past" in tense and v.tag_ in ("VBD",):
            vmods += "p"
        if v.tag_ == "VBD" and not aux:
            vmods += "p"
        if any(c.dep_ == "neg" for c in children):
            vmods += "x"

        # --- what is the predicate?
        copula = v.lemma_ == "be"
        pred_np = pred_adj = None
        if copula:
            for c in children:
                if c.dep_ == "attr":
                    pred_np = c
                elif c.dep_ == "acomp":
                    pred_adj = c
        pred_clause_adj = None   # "become happy" -> X become [X be happy] \become
        if not copula:
            for c in children:
                if c.dep_ == "attr" and not any(g.dep_ in ("dobj", "obj") for g in children):
                    pred_np = c
                elif c.dep_ == "acomp":
                    pred_clause_adj = c

        # --- subject
        subj_tok = next((c for c in children if c.dep_ in ("nsubj", "nsubjpass", "csubj", "expl")), None)
        agent = next((c for c in children if c.dep_ == "agent"), None)
        obj_tok = next((c for c in children if c.dep_ in ("dobj", "obj")), None)
        dative = next((c for c in children if c.dep_ == "dative"), None)
        if passive:
            # undo the passive:  "X was eaten by Y" -> Y ate X
            obj_tok, subj_np = subj_tok, None
            if agent is not None:
                ap = [g for g in agent.children if g.dep_ == "pobj"]
                subj_np = self.np(ap[0], quants, rel_sub) if ap else ["n4"]
            else:
                subj_np = ["n4"]
        elif subj_force is not None:
            subj_np = list(subj_force)
        elif subj_tok is not None and subj_tok.dep_ != "csubj":
            subj_np = self.np(subj_tok, quants, rel_sub)
        elif subj_override is not None:
            subj_np = list(subj_override)
        elif v.dep_ == "ROOT" and v.tag_ == "VB":
            subj_np, vmods = ["n2"], vmods + "o"
        else:
            subj_np = ["n3"]

        # --- verb token
        if copula:
            vt = [tok("v", V["be"], vmods)]
        elif v.lemma_ == "have" and not any(c.dep_ in ("xcomp", "ccomp") for c in children):
            vt = [tok("v", V["have"], vmods)]
        elif v.pos_ in ("VERB", "AUX"):
            lemma = v.lemma_.lower()
            prt = [c for c in children if c.dep_ == "prt"]
            if prt and self.lex.senses(lemma + " " + prt[0].lower_, "v"):
                lemma = lemma + " " + prt[0].lower_
            vt = self.word(v, "v", vmods, lemma=lemma)
        else:
            # verbless fragment / noun root: treat as "X exists"
            np_ = self.np(v, quants, rel_sub)
            return self.wrap_quants(np_, quants)

        # --- adverbs
        advs = []
        for c in children:
            if c.dep_ == "advmod":
                w = c.lower_
                if w in WH and v.dep_ in ("advcl", "relcl", "acl", "ccomp", "xcomp", "pcomp", "conj"):
                    continue   # here "when/where" is a subordinator, handled by the parent
                elif w in WH and w != "how":
                    wrappers.append([WH[w][0] + WH[w][1], tok("v", WH_REL[w])])
                elif w in TIME_ADV:
                    land = [TIME_ADV[w][0] + TIME_ADV[w][1]] if TIME_ADV[w] else self.word(c, "n")
                    wrappers.append(land + [tok("v", V["time"])])
                elif w in PLACE_ADV:
                    wrappers.append([PLACE_ADV[w][0] + PLACE_ADV[w][1], tok("v", V["at"])])
                elif w == "how":
                    wrappers.append(["n7q", tok("v", V["with"])])
                elif w in ("not", "never"):
                    vt[0] = add_mods(vt[0], "x")
                elif c.pos_ in ("ADV", "ADJ") and w not in ("also", "too", "just", "then", "so", "even", "only"):
                    advs += self.adj(c)
            elif c.dep_ == "npadvmod" and c.pos_ in ("NOUN", "PROPN"):
                wrappers.append(self.np(c, quants, rel_sub) + [tok("v", V["time"])])

        # --- object / predicate
        obj = []
        if pred_np is not None:
            obj = self.np(pred_np, quants, rel_sub)
        elif pred_adj is not None:
            obj = []
            advs = advs + self.adj(pred_adj)
        elif obj_force is not None:
            obj = list(obj_force)
        elif obj_tok is not None:
            obj = self.np(obj_tok, quants, rel_sub)
        if dative is not None:
            dn = [g for g in dative.children if g.dep_ == "pobj"] if dative.pos_ == "ADP" else [dative]
            if dn:
                wrappers.append(self.np(dn[0], quants, rel_sub) + [tok("v", V["to"])])

        # --- prepositional phrases
        for c in children:
            if c.dep_ == "prep" or (c.dep_ == "agent" and not passive):
                objs = [g for g in c.children if g.dep_ == "pobj"]
                pcomp = [g for g in c.children if g.dep_ == "pcomp"]
                neg = "x" if c.lower_ in NEGATING_PREPS else ""
                if objs:
                    # "because of", "in front of", "out of": the real object hangs off "of"
                    o = objs[0]
                    rid = self.rel_for(c, o)
                    wrappers.append(self.np(o, quants, rel_sub) + [tok("v", rid, neg)])
                elif pcomp:
                    inner = pcomp[0]
                    sub_of = [g for g in inner.children if g.dep_ == "pobj"]
                    if inner.pos_ == "ADP" and sub_of:
                        rid = PREP_TO_REL.get(c.lower_, V["of"])
                        wrappers.append(self.np(sub_of[0], quants, rel_sub) + [tok("v", rid, neg)])
                    else:
                        rid = PREP_TO_REL.get(c.lower_, V["of"])
                        a = self.clause(inner, subj_override=self.bare(subj_np), rel_sub=rel_sub)
                        logic.append((rid, a, neg))

        # --- clause complements (ccomp / xcomp)
        comps = []
        for c in children:
            if c.dep_ in ("ccomp", "xcomp"):
                so = None
                if c.dep_ == "xcomp":
                    so = self.bare(obj if obj else subj_np)
                if c.dep_ == "xcomp" and c.pos_ in ("ADJ",):
                    advs += self.adj(c)
                    continue
                comps.append(self.clause(c, subj_override=so, rel_sub=rel_sub))

        # --- subordinate clauses
        for c in children:
            if c.dep_ == "advcl":
                mark = next((g for g in c.children if g.dep_ == "mark"
                             or (g.dep_ == "advmod" and g.tag_ == "WRB" and g.lower_ in MARK_TO_REL)), None)
                m = mark.lower_ if mark is not None else ""
                rid = MARK_TO_REL.get(m, V["time"])
                a = self.clause(c, subj_override=self.bare(subj_np), rel_sub=rel_sub)
                if m == "unless":
                    a = self.mark_main_verb(a, "x")
                if m == "so":
                    logic.append(("so", a, ""))
                else:
                    logic.append((rid, a, ""))

        if pred_clause_adj is not None:
            comps.insert(0, self.bare(subj_np) + [tok("v", V["be"])] + self.adj(pred_clause_adj))
        core = subj_np + vt + advs + obj
        vid = re.match(r"^v(\d+)", vt[0])
        for cp in comps:
            core += cp + [f"\\v{vid.group(1)}" if vid else "\\v0"]

        out = core
        for w in reversed(wrappers):
            rid = re.match(r"^v(\d+)", w[-1]).group(1)
            out = w + out + [f"\\v{rid}"]
        for rid, a, neg in logic:
            if rid == "so":   # "A, so B": main clause causes the subordinate one
                out = [tok("v", V["cause"])] + out + [f"\\v{V['cause']}"] + a + [f"\\v{V['cause']}"]
            else:
                out = [tok("v", rid, neg)] + a + [f"\\v{rid}"] + out + [f"\\v{rid}"]
        out = self.wrap_quants(out, quants)

        if _nodist:
            return out
        # --- coordinated noun phrases are distributed: "John and Mary left"
        for slot, head in (("subj", subj_tok if not passive else None), ("obj", obj_tok)):
            if head is None:
                continue
            cc = next((g for g in head.children if g.dep_ == "cc"), None)
            lid = CC_TO_LOGIC.get(cc.lower_ if cc is not None else "and", V["and"])
            for cj in [g for g in head.conjuncts if g.i > head.i]:
                q2 = []
                alt = self.np(cj, q2, rel_sub)
                kw = {"subj_force": alt} if slot == "subj" else {"obj_force": alt, "subj_force": self.bare(subj_np)}
                b = self.clause(v, subj_override=subj_override, rel_sub=rel_sub, _nodist=True, **kw)
                out = [tok("v", lid)] + out + [f"\\v{lid}"] + b + [f"\\v{lid}"]
        # --- coordination: "X ate and Y slept"
        for c in children:
            if c.dep_ == "conj" and c.pos_ in ("VERB", "AUX"):
                cc = next((g for g in children if g.dep_ == "cc"), None)
                lid = CC_TO_LOGIC.get(cc.lower_ if cc is not None else "and", V["and"])
                b = self.clause(c, subj_override=self.bare(subj_np), rel_sub=rel_sub)
                out = [tok("v", lid)] + out + [f"\\v{lid}"] + b + [f"\\v{lid}"]
        return out

    def wrap_quants(self, out, quants):
        for restrictor, qid in reversed(quants):
            out = restrictor + [tok("v", qid)] + out + [f"\\v{qid}"]
        return out

"""Coverage check: every content word of the English must reach the meaning
graph.  A word that does not was dropped somewhere (a relation the analyzer
does not handle, a phrase the parser attached oddly); the sentence is then
flagged instead of silently losing meaning.

    missing(tree, events) -> list of word forms
"""
from .ir import Entity, Event

CONTENT = {"NOUN", "PROPN", "VERB", "ADJ", "NUM"}
# relations whose words are folded into their head on purpose
FOLDED = {"aux", "aux:pass", "cop", "fixed", "flat", "flat:name", "goeswith", "det", "case", "mark",
          "cc", "punct", "expl", "compound:prt", "discourse", "reparandum"}


def positions(ev, out=None):
    """Token indices that reached the graph (events, entities, attributes)."""
    out = set() if out is None else out
    seen = set()

    def ent(e):
        if e is None or id(e) in seen:
            return
        seen.add(id(e))
        if e.pos is not None:
            out.add(e.pos)
        out.update(e.parts)
        for a in (x for at in e.attrs for x in at.leaves()):
            if a.pos is not None:
                out.add(a.pos)
            out.update(a.parts)
        for m in e.members:
            ent(m)
        ent(e.possessor)            # (a re-mention's antecedent is said elsewhere: not followed)
        for r in e.restrictors:
            event(r)

    def event(x):
        if x is None or id(x) in seen:
            return
        seen.add(id(x))
        if x.pos is not None:
            out.add(x.pos)
        for a in x.manner + x.pred_attrs:
            if a.pos is not None:
                out.add(a.pos)
        ent(x.subj)
        ent(x.obj)
        for r in x.rels:
            (event if isinstance(r.landmark, Event) else ent)(r.landmark)
        for c in x.comps:
            event(c)
        for _, c in x.coord:
            event(c)
    event(ev)
    return out


def missing(toks, ev):
    """Content words of the analyzed sentence (analyze.Tok list, after
    restructure()) that are absent from the graph."""
    got = positions(ev)
    out = []
    for t in toks:
        if t.upos not in CONTENT or t.deprel in FOLDED or t.i in got:
            continue
        if t.upos == "PROPN" and t.deprel == "compound":           # inside a name
            continue
        out.append(t.form)
    return out

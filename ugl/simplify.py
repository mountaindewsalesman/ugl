"""One sentence's Event graph -> several simpler sentences with the same meaning.

Runs between coref and serialize.  It only moves material that is already a
separate statement, so no meaning is added or lost:

  * clauses joined by plain "and" at the top level:
        "A and B"                      ->  A.  B.
  * asides: restrictors set off by commas (non-restrictive relative clauses,
    appositions, participial phrases) say something extra about their head
    rather than picking it out:
        "My mother, who lives in Ohio, is ill"
                                       ->  my mother is ill.  the mother lives in Ohio.
    The aside refers back to its head with a definite re-mention (refer()).

Restrictive clauses ("the city that we visited") identify their head and
stay inside the noun phrase; "but", "or", "if", "because" ... links stay too,
since splitting them would drop the relation.
"""
from .ir import Entity, Event


def split(ev):
    """-> list of Events, main statement first, then its asides in reading order."""
    out = []
    for main in split_and(ev):
        asides = []
        hoist_event(main, asides)
        out.append(main)
        for a in asides:
            out += split(a)
    return out


def split_and(ev):
    """Peel trailing top-level "and" clauses off (the serializer nests
    coordination left to right, so trailing "and"s are the outermost)."""
    tail = []
    while ev.coord and ev.coord[-1][0] == "and":
        tail.append(ev.coord.pop()[1])
    out = [ev]
    for t in reversed(tail):
        out += split_and(t)
    return out


def hoist_event(ev, asides):
    for e in (ev.subj, ev.obj):
        hoist_entity(e, asides)
    for r in ev.rels:
        if isinstance(r.landmark, Event):
            hoist_event(r.landmark, asides)
        else:
            hoist_entity(r.landmark, asides)
    for c in ev.comps:
        hoist_event(c, asides)
    for _, c in ev.coord:
        hoist_event(c, asides)


def hoist_entity(e, asides):
    if e is None or e.ref is not None:
        return
    for m in e.members:
        hoist_entity(m, asides)
    hoist_entity(e.possessor, asides)
    keep = []
    for r in e.restrictors:
        if r.aside:
            r.aside = False
            asides.append(r)
        else:
            hoist_event(r, asides)
            keep.append(r)
    e.restrictors = keep

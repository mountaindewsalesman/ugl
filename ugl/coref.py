"""Pronoun resolution over the Event/Entity graph of a whole document.

Each third-person pronoun gets `ref` pointing to the entity it means:
  * reflexives (himself, itself...)  -> the subject of their own clause
  * he/she/him/her/his/hers          -> latest person-like entity (person noun
                                        or name), subjects preferred
  * it/its                           -> latest singular non-person entity
  * they/them/their                  -> latest plural or coordinated entity
Candidates come from the current and the two previous sentences.  Anything
unresolved becomes a reserved noun: n4 (someone) or n3t (the thing).
"""
from .ir import Entity, Event, Rel

PERSONAL = {"he", "him", "his", "himself", "she", "her", "hers", "herself"}
NEUTER = {"it", "its", "itself"}
PLURAL = {"they", "them", "their", "theirs", "themselves"}
QUANTIFIER = {"both", "all", "each", "most", "many", "few"}     # "both are smelly" = all of them
REFLEXIVE = {"himself", "herself", "itself", "themselves"}
RELATIVE = {"who", "whom", "which", "that", "whose"}
WINDOW = 2


def mentions(ev, sent_no, out, clause_subj=None):
    """Collect (entity, role, sentence, clause subject) in textual order."""
    def ent(e, role):
        if e is None:
            return
        out.append((e, role, sent_no, clause_subj))
        if e.possessor is not None:
            ent(e.possessor, "poss")
        for m in e.members:
            ent(m, role)
        for r in e.restrictors:
            mentions(r, sent_no, out)

    # roles follow the English surface: in a passive the patient was the subject
    # ("The bridge was destroyed by the flood ... it" = the bridge)
    subj = ev.obj if ev.passive and ev.obj is not None else ev.subj
    ent(ev.subj, "obl" if ev.passive and ev.obj is not None else "subj")
    clause_subj = subj
    if ev.obj is not None:
        out.append((ev.obj, "subj" if ev.passive else "obj", sent_no, subj))
        if ev.obj.possessor is not None:
            out.append((ev.obj.possessor, "poss", sent_no, subj))
        for m in ev.obj.members:
            out.append((m, "obj", sent_no, subj))
        for r in ev.obj.restrictors:
            mentions(r, sent_no, out)
    for r in ev.rels:
        if isinstance(r.landmark, Event):
            mentions(r.landmark, sent_no, out)
        else:
            e = r.landmark
            out.append((e, "obl", sent_no, subj))
            if e.possessor is not None:
                out.append((e.possessor, "poss", sent_no, subj))
            for rr in e.restrictors:
                mentions(rr, sent_no, out)
    for c in ev.comps:
        mentions(c, sent_no, out)
    for _, c in ev.coord:
        mentions(c, sent_no, out)


def _order(item):
    e = item[0]
    return (item[2], e.pos if e.pos is not None else 10**6)


def topic_entity(found, topic):
    """The document's topic (e.g. a Wikipedia title) as an entity: its first
    mention in the text if there is one (articles open with their subject),
    else a name made from the title."""
    words = {w.lower() for w in topic.replace("(", " ").replace(")", " ").split() if len(w) > 2}
    for e, _role, _sno, _ in found:
        if e.kind in ("name", "lex") and e.ref is None and words & {w.lower() for w in e.lemma.split()}:
            return e
    return Entity("name", topic.split(" (")[0])


def resolve(events, unresolved=None, topic=None):
    """events: list of sentence-level Events (one document).  Mutates in place.
    unresolved: optional list; gets (sentence index, pronoun) for every pronoun
    with nothing to refer to (it then becomes someone / the thing).
    topic: what the document is about (a title); a pronoun that nothing in the
    last sentences fits refers to it ("It was founded in 1877")."""
    found = []
    for i, ev in enumerate(events):
        if ev is not None:
            mentions(ev, i, found)
    found.sort(key=_order)
    topic_e = topic_entity(found, topic) if topic else None
    seen = []          # (entity, role, sentence) of real (non-pronoun) mentions
    for e, role, sno, csubj in found:
        if e.kind == "pronoun" and e.ref is None and e.lemma not in RELATIVE:
            e.ref = pick(e, role, sno, csubj, seen, topic_e)
            if e.ref is None:
                if unresolved is not None:
                    unresolved.append((sno, e.lemma))
                fallback(e)
        elif e.kind in ("lex", "name", "coord") and e.ref is None:
            seen.append((e, role, sno))
    return events


def pick(p, role, sno, csubj, seen, topic=None):
    w = p.lemma
    if w in REFLEXIVE and csubj is not None and csubj is not p:
        return csubj.root() if csubj.kind != "pronoun" else csubj
    recent = [s for s in seen if sno - s[2] <= WINDOW]
    if w in PERSONAL:
        ok = lambda e: (e.kind == "name" or e.is_person) and not e.plural and e.kind != "coord"
    elif w in NEUTER:
        ok = lambda e: e.kind == "lex" and not e.is_person and not e.plural
    elif w in PLURAL or w in QUANTIFIER:
        ok = lambda e: e.plural or e.kind == "coord"
    else:
        return None
    cands = [s for s in recent if ok(s[0])]
    if w in PLURAL or w in QUANTIFIER:
        # "cats and dogs ... both / they": the group, not its last member
        inside = {id(m) for s in cands if s[0].kind == "coord" for m in s[0].members}
        cands = [s for s in cands if id(s[0]) not in inside]
    if role in ("obj", "obl") and csubj is not None:
        # an ordinary (non-reflexive) pronoun never means its own clause's
        # subject: "John saw him" -> him != John (that would be "himself")
        own = csubj.root()
        cands = [s for s in cands if s[0] is not own]
    if not cands and w in NEUTER:          # "Paris ... it": allow names as a last resort
        cands = [s for s in recent if s[0].kind == "name"]
    if not cands:
        # the document topic, under the same agreement rules (names fit he/she/it)
        if topic is not None and topic is not (csubj.root() if csubj is not None and role != "subj" else None) \
                and (ok(topic) or (w in NEUTER and topic.kind == "name")):
            return topic
        return None
    # salience: latest sentence first, then subjects, then latest mention
    best = max(cands, key=lambda s: (s[2], s[1] == "subj", seen.index(s)))
    return best[0]


def fallback(p):
    """Unresolvable pronoun -> reserved noun (someone / the thing / the things)."""
    if p.lemma in QUANTIFIER:                  # nothing to range over: "all things" (quant kept)
        p.kind, p.lemma, p.plural = "reserved", "n3", True
    elif p.lemma in PERSONAL:
        p.kind, p.lemma = "reserved", "n4"
    elif p.lemma in PLURAL:
        p.kind, p.lemma, p.det, p.plural = "reserved", "n3", "t", True
    else:
        p.kind, p.lemma, p.det = "reserved", "n3", "t"

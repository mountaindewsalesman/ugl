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
ANAPHORIC = PERSONAL | NEUTER | PLURAL      # pronouns that refer back to something said
MALE_PRONOUNS = {"he", "him", "his", "himself"}
# person nouns whose gender is part of the word: "her brother ... she" cannot be the brother
GENDER = {**dict.fromkeys(
    "man boy father brother son husband king prince uncle nephew grandfather grandson monk "
    "actor waiter lord duke emperor gentleman boyfriend stepfather stepson widower groom "
    "bridegroom sir baron count earl tsar sultan".split(), "m"),
    **dict.fromkeys(
    "woman girl mother sister daughter wife queen princess aunt niece grandmother granddaughter "
    "nun actress waitress lady duchess empress girlfriend stepmother stepdaughter widow bride "
    "baroness countess tsarina sultana mistress heroine".split(), "f")}


def gender_of(e):
    """m / f / None: from the noun itself, or learned from an earlier he / she."""
    if e.gender is None and e.kind == "lex":
        e.gender = GENDER.get(e.lemma.lower())
    return e.gender


def mentions(ev, sent_no, out, clause_subj=None):
    """Collect (entity, role, sentence, clause subject) in textual order, at any
    depth: possessors of possessors ("the years of his life"), group members,
    relative clauses, relation landmarks."""
    def ent(e, role, csubj):
        if e is None:
            return
        out.append((e, role, sent_no, csubj))
        if e.possessor is not None:
            ent(e.possessor, "poss", csubj)
        for m in e.members:
            ent(m, role, csubj)
        for r in e.restrictors:
            mentions(r, sent_no, out)

    # roles follow the English surface: in a passive the patient was the subject
    # ("The bridge was destroyed by the flood ... it" = the bridge)
    passive = ev.passive and ev.obj is not None
    subj = ev.obj if passive else ev.subj
    ent(ev.subj, "obl" if passive else "subj", clause_subj)
    ent(ev.obj, "subj" if ev.passive else "obj", subj)
    for r in ev.rels:
        if isinstance(r.landmark, Event):
            mentions(r.landmark, sent_no, out)
        else:
            ent(r.landmark, "obl", subj)
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
    seen = []          # (entity, role, sentence, by pronoun?) of every mention with a referent
    for e, role, sno, csubj in found:
        if e.kind == "pronoun" and e.ref is None and e.lemma not in RELATIVE:
            e.ref = pick(e, role, sno, csubj, seen, topic_e)
            if e.ref is not None and e.ref.lemma == UNKNOWN:
                e.ref = None                     # continues a chain that refers to nothing known
            if e.ref is not None and e.lemma in PERSONAL and e.ref.gender is None:
                # "Ann ... she": Ann is female from now on, so a later "he" is someone else
                e.ref.gender = "m" if e.lemma in MALE_PRONOUNS else "f"
            if e.ref is not None:
                # the chain goes on: "Her brother installed it on her laptop" - the
                # second "her" continues the first one's referent
                seen.append((e.ref.root(), role, sno, True))
            if e.ref is None:
                if unresolved is not None and e.lemma in ANAPHORIC:
                    # "what he did", "no one": not a reference back, so nothing is missing
                    unresolved.append((sno, e.lemma))
                ph = placeholder(e.lemma)
                if ph is not None:
                    # "They saw they were caught and threw down their rifles. They were
                    # frightened": the unknown "they" stays the likeliest referent of the
                    # next "they", which must not jump to "the rifles"
                    seen.append((ph, role, sno, True))
                fallback(e)
        elif e.kind in ("lex", "name", "coord") and e.ref is None:
            seen.append((e, role, sno, False))
    return events


UNKNOWN = "?unknown"


def placeholder(w):
    """Stand-in for whatever an unresolved pronoun w refers to, with w's agreement
    (number, gender, person or thing), so later pronouns can continue its chain."""
    if w in PERSONAL:
        return Entity("name", UNKNOWN, animate=True, gender="m" if w in MALE_PRONOUNS else "f")
    if w in NEUTER:
        return Entity("lex", UNKNOWN)
    if w in PLURAL:
        return Entity("lex", UNKNOWN, plural=True)
    return None


def pick(p, role, sno, csubj, seen, topic=None):
    w = p.lemma
    if w in REFLEXIVE and csubj is not None and csubj is not p:
        return csubj.root() if csubj.kind != "pronoun" else csubj
    # never the pronoun's own chain: in "After familiarizing themselves ..., they
    # developed" the reflexive comes first and points at the later "they", which is
    # then among the mentions seen; "they" referring to itself would make a cycle
    recent = [s for s in seen if sno - s[2] <= WINDOW and s[0].root() is not p]
    if topic is not None and topic.root() is p:
        topic = None
    if w in PERSONAL:     # a name may be a person unless the dictionary knows it is not (Paris)
        g = "m" if w in MALE_PRONOUNS else "f"
        ok = lambda e: ((e.kind == "name" and e.animate is not False) or e.is_person) \
            and not e.plural and e.kind != "coord" and gender_of(e) in (None, g)
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
    if not cands and w in PLURAL:
        pair = split_antecedent(recent, role, csubj)
        if pair is not None:
            return pair
    if not cands and w in NEUTER:          # "Paris ... it": names known not to be people
        cands = [s for s in recent if s[0].kind == "name" and s[0].animate is False]
    if not cands:
        # the document topic, under the same agreement rules (names fit he/she/it)
        if topic is not None and topic is not (csubj.root() if csubj is not None and role != "subj" else None) \
                and (ok(topic) or (w in NEUTER and topic.kind == "name" and topic.animate is not True)):
            return topic
        return None
    # salience: latest sentence first, then what a pronoun already refers to
    # (the entity the text is about), then subjects, then latest mention
    best = max(cands, key=lambda s: (s[2], s[3], s[1] == "subj", seen.index(s)))
    return best[0]


def split_antecedent(recent, role, csubj):
    """'Tom found a box. Sue opened it. They laughed.': with no group to refer
    to, "they" means the two people of the last sentences -> a group of the two.
    Only when there are exactly two (three or more: which ones is a guess)."""
    def person(e):
        return e.lemma != UNKNOWN and not e.plural and e.kind != "coord" and             ((e.kind == "name" and e.animate is not False) or e.is_person)
    own = csubj.root() if role in ("obj", "obl") and csubj is not None else None
    people = {}
    for e, *_ in recent:
        e = e.root()
        if person(e) and e is not own:
            people.setdefault((e.kind, e.lemma.lower()), e)
    if own is not None and person(own):
        people.pop((own.kind, own.lemma.lower()), None)  # "Tom gave them a ball": not Tom
    if len(people) != 2:
        return None
    return Entity("coord", "and", plural=True, members=list(people.values()))


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

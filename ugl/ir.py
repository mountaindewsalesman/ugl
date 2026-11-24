"""Intermediate representation: who did what to whom, where, when.

The encoder runs in three independent passes:
    analyze.py    UD dependency tree      -> Event / Entity graph   (syntax -> meaning)
    coref.py      resolve pronouns to the entities they mean
    serialize.py  Event / Entity graph    -> UGL tokens              (meaning -> notation)
Nothing in here knows about any parser or about UGL token spelling.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Union


@dataclass
class Attr:
    """An adjective / adverb / noun-as-attribute modifying something."""
    lemma: str
    kind: str = "adj"          # adj | noun | name
    mods: str = ""             # r z l w x  (more, most, less, very, not)
    pos: Optional[int] = None  # token index, for sense choice
    adverb: bool = False       # came from an adverb (look it up as one: "away", "twice")
    # kind == "coord": coordinated modifiers ("cell and car numbers", "red or blue cars").
    # Each member is a run of modifiers ("glass transition and melting temperatures").
    group: List[List["Attr"]] = field(default_factory=list)
    logic: str = "and"

    def leaves(self):
        """This attribute, or every modifier inside a group."""
        if self.kind != "coord":
            yield self
            return
        for member in self.group:
            for a in member:
                yield from a.leaves()


@dataclass
class Entity:
    kind: str                       # lex | name | num | reserved | pronoun | coord | ref
    lemma: str = ""                 # lex/unknown: lemma; name: the name; num: digits; reserved: token (n1..)
    cls: str = "n"                  # n | nv (verb used as noun)
    plural: bool = False
    det: str = ""                   # "" | t (the) | d (this) | e (that)
    neg: bool = False               # "no X"
    question: bool = False          # "which X"
    attrs: List[Attr] = field(default_factory=list)
    count: Optional[str] = None     # "#2"
    possessor: Optional["Entity"] = None
    restrictors: List["Event"] = field(default_factory=list)   # relative clauses, appositions, NP-level PPs
    quant: Optional[str] = None     # all | most | many | few
    ref: Optional["Entity"] = None  # this mention re-mentions ref (pronouns, relative pronouns, inherited subjects)
    members: List["Entity"] = field(default_factory=list)      # kind == coord
    logic: str = "and"                                         # kind == coord
    # features used by coreference
    person: Optional[int] = None    # pronouns: 1 2 3
    gender: Optional[str] = None
    is_person: bool = False
    animate: Optional[bool] = None  # names: True person / False not a person / None unknown
    pos: Optional[int] = None
    senses: List[str] = field(default_factory=list)   # restrict the sense choice (unit symbols)
    parts: List[int] = field(default_factory=list)   # other token indices folded into this
                                                     # entity (multiword lemma, name, number words)
    head: Optional[str] = None      # set by the serializer: the head token chosen at the first
                                    # mention, so every re-mention names the same sense

    def root(self):
        e = self
        while e.ref is not None:
            e = e.ref
        return e


def refer(e):
    """A re-mention of entity e (relative pronouns, inherited subjects, gaps)."""
    return Entity("ref", ref=e)


@dataclass
class Rel:
    """Relation between a landmark and an event: 'nine' TIME [I ate]."""
    rel: str                                # relation name (core.V key), e.g. "time", "with"
    landmark: Union[Entity, "Event"]
    neg: bool = False


@dataclass
class Event:
    pred: str                         # verb | be | have | exist
    lemma: str = ""
    pos: Optional[int] = None
    mods: str = ""                    # p f g k h c m u q o x
    subj: Optional[Entity] = None
    obj: Optional[Entity] = None
    manner: List[Attr] = field(default_factory=list)        # adverbs ("quickly", "unfortunately")
    pred_attrs: List[Attr] = field(default_factory=list)    # predicate adjectives of "be" ("is tall")
    comps: List["Event"] = field(default_factory=list)      # clause complements (say that ..., want to ...)
    rels: List[Rel] = field(default_factory=list)           # wrapped around the clause, outermost first
    coord: List[tuple] = field(default_factory=list)        # (logic name, Event)
    passive: bool = False             # English said it in the passive: obj was the surface subject
                                      # (UGL has no passive; coreference salience still needs it)
    aside: bool = False               # a restrictor set off by commas ("Ann, who lives here,"):
                                      # it adds a fact about its head instead of picking it out

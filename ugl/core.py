"""Fixed (non-frequency) parts of UGL: reserved words, modifier letters,
preposition table, and the token format.  Everything here is frozen;
the frequency-ordered lexicon lives in lexicon.json.

Token shapes
    n123st    class letter + index + modifier letters (here: plural, definite)
    nv57      class shift: verb 57 used as a noun ("eating")
    \\v57      closing tag for a clause opened by verb 57
    #9        number literal
    'Paris'   name literal (spaces become underscores)
    n?word    unknown word literal of a class (n/v/a)
"""
import re

# Lexical (frequency-ordered) indices start here for nouns and verbs.
# 1..99 are reserved for the fixed words below.  Adjectives have no reserved
# block and start at 1.
LEXICAL_START = {"n": 100, "v": 100, "a": 1}

# ---------------------------------------------------------------- nouns
RESERVED_NOUNS = {
    1: "speaker",      # replaces I/me/my (n1s = we)
    2: "listener",     # replaces you
    3: "thing",        # something / what (n3q) / dummy "it"
    4: "person",       # someone / who (n4q)
    5: "place",        # here (n5d), there (n5e), where (n5q)
    6: "time",         # now (n6d), then (n6e), when (n6q)
    7: "way",          # how (n7q)
    8: "reason",       # why (n8q)
    9: "amount",       # how much (n9q)
}

# ---------------------------------------------------------------- verbs
# Kinds:
#   core     ordinary verbs that are just very structural
#   relmark  NP-internal relative clause marker:  n5 v3 <clause> \v3
#   rel      relation; NP form   X R <clause> \R      (X is R of clause)
#            or clause form      R <A> \R <B> \R      (A is landmark, B the event)
#   logic    relation that is only used in clause form
#   quant    quantifier, NP form: X Q <clause> \Q  (for Q X: clause)
RESERVED_VERBS = {
    1: ("be", "core"), 2: ("have", "core"), 3: ("that", "relmark"),
    4: ("and", "logic"), 5: ("or", "logic"), 6: ("but", "logic"),
    7: ("cause", "rel"), 8: ("if", "logic"),
    9: ("all", "quant"), 10: ("most", "quant"), 11: ("many", "quant"), 12: ("few", "quant"),
    20: ("time", "rel"), 21: ("before", "rel"), 22: ("after", "rel"),
    23: ("during", "rel"), 24: ("until", "rel"), 25: ("since", "rel"),
    30: ("in", "rel"), 31: ("on", "rel"), 32: ("at", "rel"), 33: ("under", "rel"),
    34: ("above", "rel"), 35: ("near", "rel"), 36: ("between", "rel"),
    37: ("around", "rel"), 38: ("through", "rel"), 39: ("across", "rel"),
    40: ("along", "rel"), 41: ("behind", "rel"), 42: ("in front of", "rel"),
    43: ("out of", "rel"),
    50: ("from", "rel"), 51: ("to", "rel"), 52: ("with", "rel"), 53: ("for", "rel"),
    54: ("about", "rel"), 55: ("like", "rel"), 56: ("against", "rel"),
    57: ("despite", "rel"), 58: ("except", "rel"), 59: ("as", "rel"),
    60: ("of", "rel"), 61: ("by", "rel"),
}
V = {name: i for i, (name, _) in RESERVED_VERBS.items()}
VERB_KIND = {i: kind for i, (_, kind) in RESERVED_VERBS.items()}
WRAPPER_VERBS = {i for i, k in VERB_KIND.items() if k in ("rel", "quant")}
CLAUSE_FORM_VERBS = {i for i, k in VERB_KIND.items() if k in ("rel", "logic")}

# English preposition -> relation verb id.  "without" = with + negation.
PREP_TO_REL = {
    "at": V["at"], "in": V["in"], "inside": V["in"], "within": V["in"], "into": V["to"],
    "on": V["on"], "onto": V["to"], "upon": V["on"],
    "under": V["under"], "below": V["under"], "beneath": V["under"], "underneath": V["under"],
    "above": V["above"], "over": V["above"],
    "near": V["near"], "beside": V["near"], "by": V["by"], "next": V["near"],
    "between": V["between"], "among": V["between"], "amongst": V["between"],
    "around": V["around"], "round": V["around"],
    "through": V["through"], "throughout": V["through"], "via": V["through"],
    "across": V["across"], "along": V["along"], "behind": V["behind"],
    "from": V["from"], "to": V["to"], "toward": V["to"], "towards": V["to"],
    "with": V["with"], "without": V["with"], "for": V["for"],
    "about": V["about"], "regarding": V["about"], "concerning": V["about"],
    "like": V["like"], "unlike": V["like"], "against": V["against"],
    "despite": V["despite"], "except": V["except"], "as": V["as"], "of": V["of"],
    "before": V["before"], "after": V["after"], "during": V["during"],
    "until": V["until"], "till": V["until"], "since": V["since"],
    "because": V["cause"], "due": V["cause"], "out": V["out of"], "outside": V["out of"],
    "front": V["in front of"],
}
NEGATING_PREPS = {"without", "unlike"}
# prepositions whose meaning becomes "time" when the object is a time expression
TIME_PREPS = {"at", "in", "on"}

# Subordinating conjunction (mark) -> clause-form relation
MARK_TO_REL = {
    "because": V["cause"], "since": V["cause"], "as": V["cause"],
    "if": V["if"], "unless": V["if"],               # unless = if + negated A
    "when": V["time"], "whenever": V["time"], "while": V["during"],
    "before": V["before"], "after": V["after"], "until": V["until"], "till": V["until"],
    "although": V["but"], "though": V["but"], "whereas": V["but"],
    "so": V["cause"],
}
CC_TO_LOGIC = {"and": V["and"], "or": V["or"], "but": V["but"], "nor": V["or"], "yet": V["but"]}

QUANT_DETS = {"every": V["all"], "each": V["all"], "all": V["all"],
              "most": V["most"], "many": V["many"], "few": V["few"]}

# ------------------------------------------------------------- modifiers
VERB_MODS = {
    "p": "past", "f": "future", "g": "ongoing", "k": "completed",
    "q": "question", "o": "command", "h": "hypothetical", "x": "not",
    "c": "able (can)", "m": "possible (may)", "u": "required (must)",
}
NOUN_MODS = {
    "s": "plural", "t": "definite (the)", "d": "this/these", "e": "that/those",
    "x": "none (no)", "q": "which?/what?", "y": "possessor of preceding head",
    "j": "attribute of preceding head (noun used as adjective)",
}
ADJ_MODS = {"r": "more", "z": "most", "l": "less", "w": "very", "x": "not"}
MOD_ORDER = "pfgkhcmuoqxstdeyjrzlw"   # canonical letter order in output

MODAL_MODS = {"will": "f", "shall": "f", "would": "h", "can": "c", "could": "ch",
              "may": "m", "might": "mh", "must": "u", "should": "uh", "ought": "uh"}

# ------------------------------------------------------------------ tokens
TOKEN_RE = re.compile(r"""
    (?P<close>\\v\d+)
  | (?P<word>(?:nv|n|v|a)\d+[a-z]*)
  | (?P<unk>[nva]\?[^\s]+)
  | (?P<num>\#[^\s]+)
  | (?P<name>'[^'\s]+'[a-z]*)
""", re.X)
WORD_RE = re.compile(r"^(nv|n|v|a)(\d+)([a-z]*)$")


def canon_mods(mods):
    """Deduplicate and sort modifier letters into canonical order."""
    s = set(mods)
    return "".join(c for c in MOD_ORDER if c in s)


def tok(cls, idx, mods=""):
    return f"{cls}{idx}{canon_mods(mods)}"


def add_mods(token, mods):
    """Append modifier letters to an existing token string."""
    if not mods:
        return token
    m = WORD_RE.match(token)
    if m:
        return tok(m.group(1), m.group(2), m.group(3) + mods)
    if token.startswith("'"):
        base, _, old = token.rpartition("'")
        return base + "'" + canon_mods(old + mods)
    return token  # numbers / unknowns take no modifiers


def tokenize(text):
    out, pos = [], 0
    text = text.strip()
    for part in text.split():
        m = TOKEN_RE.fullmatch(part)
        if not m:
            raise ValueError(f"bad UGL token: {part!r}")
        out.append(part)
    return out

"""Fixed (non-frequency) parts of UGL: reserved words, modifier letters,
preposition table, and the token format.  Everything here is frozen;
the frequency-ordered lexicon lives in lexicon.json.

Token shapes
    n123st    class letter + index + modifier letters (here: plural, definite)
    nv57      class shift: verb 57 used as a noun ("eating")
    \\v57      closing tag for a clause opened by verb 57
    #9        number literal
    'Paris'   name literal (spaces become underscores)
    n?'word'j unknown-word literal of a class (n/v/a); takes modifiers like names
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
    "b": "non- (something other than this: non-member)", "i": "anti- (against: anti-war)",
}
ADJ_MODS = {"r": "more", "z": "most", "l": "less", "w": "very", "x": "not", "i": "anti- (against)"}
MOD_ORDER = "pfgkhcmuoqxbistdeyjrzlw"   # canonical letter order in output

# Word-forming prefixes that are grammar, not vocabulary: a word missing from the
# dictionary whose stem is in it is the stem plus a letter.  Per class of the stem;
# non- on an adjective or verb is plain negation (non-toxic = not toxic), so one
# meaning keeps one spelling.  A word the dictionary has as a whole ("antibody",
# "nonsense") never reaches this.
PREFIXES = {"non": {"n": "b", "a": "x", "r": "x", "v": "x"},
            "anti": {"n": "i", "a": "i", "r": "i", "v": "i"}}
PREFIX_TEXT = {"b": "non-", "i": "anti-"}

MODAL_MODS = {"will": "f", "shall": "f", "would": "h", "can": "c", "could": "ch",
              "may": "m", "might": "mh", "must": "u", "should": "uh", "ought": "uh"}

# ------------------------------------------------------------------ tokens
TOKEN_RE = re.compile(r"""
    (?P<close>\\v\d+)
  | (?P<word>(?:nv|n|v|a)\d+[a-z]*)
  | (?P<unk>[nva]\?(?:'[^'\s]+'[a-z]*|[^\s']+))   # n?'word'mods  (old form n?word read-only)
  | (?P<num>\#[^\s]+)
  | (?P<name>'[^'\s]+'[a-z]*)
""", re.X)
WORD_RE = re.compile(r"^(nv|n|v|a)(\d+)([a-z]*)$")


def canon_mods(mods):
    """Deduplicate and sort modifier letters into canonical order."""
    s = set(mods)
    return "".join(c for c in MOD_ORDER if c in s)


ROUND_NUMBERS = {10: "ten", 12: "dozen", 20: "score", 100: "hundred", 1000: "thousand",
                 10**6: "million", 10**9: "billion"}


MONTH_NAMES = ("January February March April May June July August September October "
               "November December").split()


def number_text(raw):
    """Number literal -> English: "#9" -> "9", "#10s" -> "tens" (an amount
    counted in tens), "#1000s" -> "thousands"; dates: "#2025-02-14" ->
    "14 February 2025", "#2025-02" -> "February 2025", "#--02-14" -> "14 February"."""
    body = raw[1:]
    m = re.fullmatch(r"(\d{4}|-)-(\d\d)(?:-(\d\d))?", body)
    if m and 1 <= int(m.group(2)) <= 12:
        month = MONTH_NAMES[int(m.group(2)) - 1]
        day = str(int(m.group(3))) + " " if m.group(3) else ""
        year = " " + m.group(1) if m.group(1) != "-" else ""
        return f"{day}{month}{year}"
    if body.endswith("s") and body[:-1].isdigit() and int(body[:-1]) in ROUND_NUMBERS:
        return ROUND_NUMBERS[int(body[:-1])] + "s"
    return body.replace("_", " ")


def tok(cls, idx, mods=""):
    return f"{cls}{idx}{canon_mods(mods)}"


def add_mods(token, mods):
    """Append modifier letters to an existing token string."""
    if not mods:
        return token
    m = WORD_RE.match(token)
    if m:
        return tok(m.group(1), m.group(2), m.group(3) + mods)
    if token.startswith("'") or (len(token) > 2 and token[1] == "?"):
        kind, cls, text, old = parse_literal(token)
        return literal(kind, text, old + mods, cls)
    return token  # numbers take no modifiers


def literal(kind, text, mods="", cls="n"):
    """Spell a name or unknown-word literal: 'New_York'y, n?'blorft'j."""
    body = re.sub(r"\s+", "_", text.strip()).replace("'", "")
    if kind == "name":            # one spelling per name: "K. B. Hedgewar" = "K B Hedgewar"
        body = re.sub(r"(?<=[^\W\d_])\.", "", body)
    q = f"'{body}'{canon_mods(mods)}"
    return q if kind == "name" else f"{cls}?{q}"


def parse_literal(token):
    """-> (kind 'name'|'unk', cls, text, mods) for a name / unknown literal."""
    if token.startswith("'"):
        body, _, mods = token[1:].rpartition("'")
        return "name", "n", body.replace("_", " "), mods
    cls, rest = token[0], token[2:]
    if rest.startswith("'"):
        body, _, mods = rest[1:].rpartition("'")
        return "unk", cls, body.replace("_", " "), mods
    return "unk", cls, rest.replace("_", " "), ""


def tokenize(text):
    out, pos = [], 0
    text = text.strip()
    for part in text.split():
        m = TOKEN_RE.fullmatch(part)
        if not m:
            raise ValueError(f"bad UGL token: {part!r}")
        out.append(part)
    return out

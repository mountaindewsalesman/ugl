"""Back-references (SPEC rule 15): a repeated mention becomes @k.

UGL has no pronouns: the encoder writes out who or what is meant every time
("Tom landed on Tom's back").  That is precise but costly to predict: a
language model must spell the whole name again where English says "he".  A
back-reference keeps the precision and costs about what a pronoun does:

    @k    the referent mentioned k-th most recently (k = 1..9) in this text
    @ky   its possessive (y, as on any noun)

    'Tom' v_land_p v30 n_back_t 'Tom'y \\v30        (full form)
    'Tom' v_land_p v30 n_back_t @1y \\v30           (with back-references)

Referents are names ('Lily', or numbered proper nouns), definite noun
phrases' heads (n_dog_t, n_cat_ts, n_box_d) and two-member coordinated groups
of those (v4 'Tom' \\v4 'Lily' \\v4, rule 9).  Recency is in token order: each
mention moves its referent to the front, and a group's mention moves the group
(written in full, its members count first, then the group).

compress() and expand() are exact inverses on any UGL text, and depend on the
text alone, so stored full-form UGL can be compressed for training and expanded
back for decoding.  Only references inside the given lines are used: the
first mention of anything in a text is always written in full.
"""
import re

MAX_REF = 9
REF_RE = re.compile(r"^@([1-9])([a-z]*)$")
_NOUN = re.compile(r"^n(\d+)([a-z]*)$")
_NAME = re.compile(r"^('[^'\s]+')([a-z]*)$")
GROUP_WORDS = ("v4", "v5")             # and, or
RESERVED_MAX = 99                      # n1..n99 are fixed words (I, you, thing, ...): never referents


def _split(token, is_proper):
    """A referent mention -> (key, extra modifiers), else None.  The possessive y
    is not part of who is meant ('Tom'y and 'Tom' are the same referent)."""
    m = _NAME.match(token)
    if m:
        body, mods = m.groups()
    else:
        m = _NOUN.match(token)
        if not m:
            return None
        idx, mods = m.groups()
        if int(idx) <= RESERVED_MAX:
            return None
        body = "n" + idx
        definite = any(c in mods for c in "tde")
        if not definite and not (is_proper and is_proper(body)):
            return None
    if "j" in mods or "q" in mods:      # an attribute noun or a question word refers to nothing
        return None
    key = body + mods.replace("y", "")
    return key, ("y" if "y" in mods else "")


class _Recency:
    def __init__(self):
        self.keys = []                  # most recent first

    def find(self, key):
        try:
            k = self.keys.index(key)
        except ValueError:
            return None
        return k + 1 if k < MAX_REF else None

    def touch(self, key):
        if key in self.keys:
            self.keys.remove(key)
        self.keys.insert(0, key)

    def get(self, k):
        if k > len(self.keys):
            raise ValueError(f"@{k}: only {len(self.keys)} referents mentioned so far")
        return self.keys[k - 1]


def _group_at(toks, i):
    """(L, has_y) if toks[i:i+5] is a two-member group L A \\L B \\L, else None."""
    if i + 4 >= len(toks):
        return None
    head = toks[i]
    base = head.rstrip("y") if head[:2] in GROUP_WORDS and head.rstrip("y") in GROUP_WORDS else None
    if base is None or head not in (base, base + "y"):
        return None
    if toks[i + 2] != "\\" + base or toks[i + 4] != "\\" + base:
        return None
    return base, head.endswith("y")


def _member_key(token, is_proper, rec, expanding):
    """Key of a group member (resolving @k when expanding), or None."""
    if expanding:
        m = REF_RE.match(token)
        if m:
            key = rec.get(int(m.group(1)))
            return key if " " not in key and not m.group(2) else None
    s = _split(token, is_proper)
    return s[0] if s and not s[1] else None


def compress(lines, is_proper=None):
    """Full-form UGL sentences (in order) -> the same with back-references."""
    rec, out = _Recency(), []
    for line in lines:
        toks, res, i = line.split(), [], 0
        while i < len(toks):
            g = _group_at(toks, i)
            if g:
                base, y = g
                a = _member_key(toks[i + 1], is_proper, rec, False)
                b = _member_key(toks[i + 3], is_proper, rec, False)
                if a and b:
                    gkey = f"{base} {a} \\{base} {b} \\{base}"
                    k = rec.find(gkey)
                    if k:
                        res.append(f"@{k}" + ("y" if y else ""))
                    else:
                        res.append(toks[i])
                        res.append(_one(toks[i + 1], is_proper, rec))
                        res.append(toks[i + 2])
                        res.append(_one(toks[i + 3], is_proper, rec))
                        res.append(toks[i + 4])
                    rec.touch(gkey)
                    i += 5
                    continue
            res.append(_one(toks[i], is_proper, rec))
            i += 1
        out.append(" ".join(res))
    return out


def _one(token, is_proper, rec):
    s = _split(token, is_proper)
    if not s:
        return token
    key, extra = s
    k = rec.find(key)
    rec.touch(key)
    return f"@{k}{extra}" if k else token


def expand(lines, is_proper=None):
    """UGL sentences with back-references -> full form (the inverse of compress)."""
    rec, out = _Recency(), []
    for line in lines:
        toks, res, i = line.split(), [], 0
        while i < len(toks):
            g = _group_at(toks, i)
            if g:
                base, _ = g
                a = _member_key(toks[i + 1], is_proper, rec, True)
                if a:
                    first = _expand_one(toks[i + 1], is_proper, rec)
                    b = _member_key(toks[i + 3], is_proper, rec, True)
                    if b:
                        second = _expand_one(toks[i + 3], is_proper, rec)
                        res += [toks[i], first, toks[i + 2], second, toks[i + 4]]
                        rec.touch(f"{base} {a} \\{base} {b} \\{base}")
                        i += 5
                        continue
                    res += [toks[i], first]  # not a group of referents after all: go on token by token
                    i += 2
                    continue
            res.append(_expand_one(toks[i], is_proper, rec))
            i += 1
        out.append(" ".join(res))
    return out


def _expand_one(token, is_proper, rec):
    m = REF_RE.match(token)
    if m:
        key = rec.get(int(m.group(1)))
        rec.touch(key)
        if " " in key:                      # a group: the possessive goes on its first word
            base = key.split()[0]
            return key.replace(base, base + m.group(2), 1) if m.group(2) else key
        return _with_mods(key, m.group(2))
    s = _split(token, is_proper)
    if s:
        rec.touch(s[0])
    return token


def _with_mods(key, extra):
    from .core import canon_mods
    if not extra:
        return key
    if key.startswith("'"):
        body, _, mods = key[1:].rpartition("'")
        return f"'{body}'" + canon_mods(mods + extra)
    m = _NOUN.match(key)
    return f"n{m.group(1)}" + canon_mods(m.group(2) + extra)

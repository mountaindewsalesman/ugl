"""Does the decoded English still say what the source said?  (the meaning check)

The round trip (decode, encode again, same UGL?) only shows that the UGL is
stable: a sentence translated wrongly the same way twice passes it.  This
compares the decoded English with the SOURCE, word by word in meaning: the
content words of each side (nouns, names, verbs, adjectives, adverbs,
numbers; lemmatized), matched as

  same / synonym   the same word, or words sharing a dictionary sense (big ~ large)
  referent         a name or noun from the document standing where the source
                   had a pronoun (UGL writes out who is meant: expected)
  changed          a source word replaced by a word with another meaning (cry -> call)
  lost             a source word with nothing for it in the output
  added            an output word from nowhere

What UGL leaves out on purpose is not a loss: "once upon a time" says "once".
Used by encode_lines(roundtrip=True) to flag sentences, and by
tools/fidelity_report.py to count the families.
"""
import re

CONTENT = {"NOUN", "PROPN", "VERB", "ADJ", "ADV", "NUM"}
SKIP = {"be", "have", "do", "not", "so", "very", "then", "there", "also", "just", "too", "n't"}
NUMBERS = {w: str(i) for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve "
                                            "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split())}
# what UGL drops on purpose (it says the same with less), removed from the source first
INTENDED = [(re.compile(r"\bonce upon a time\b", re.I), "once")]
CLS = {"NOUN": "n", "PROPN": "n", "VERB": "v", "ADJ": "a", "ADV": "a"}


def normalize_source(text):
    for pat, rep in INTENDED:
        text = pat.sub(rep, text)
    return text


def lemma(form, upos, freq=None):
    """freq: word form -> uses per million (lexicon.word_freq); a lemma far rarer
    than the word as written is a wrong guess ("other" is not a comparative of "oth")."""
    if upos == "PROPN":
        return form
    if upos == "NUM" or form.lower() in NUMBERS:     # UGL writes numbers as digits: one = 1
        return NUMBERS.get(form.lower(), form.replace(",", ""))
    from lemminflect import getLemma
    got = getLemma(form, upos=upos)
    lm = (got[0] if got else form).lower()
    if freq is not None and lm != form.lower() and freq.get(lm, 0.0) < 0.01 * freq.get(form.lower(), 0.0):
        return form.lower()
    return lm


def content_words(tree, freq=None):
    """[(lemma, upos)] of a parsed sentence (UD token dicts)."""
    out = []
    for t in tree:
        if t["upos"] in CONTENT:
            lm = lemma(t["form"], t["upos"], freq)
            if lm.lower() not in SKIP and any(c.isalnum() for c in lm):
                out.append((lm, t["upos"]))
    return out


class Meaning:
    def __init__(self, lexicon):
        self.lex = lexicon
        self._senses = {}

    def spellings(self, w):
        """(UGL spelling of the word's first sense, spellings of all its senses)."""
        k = (w[0].lower(), CLS.get(w[1]))
        if k not in self._senses:
            senses = [s for s in self.lex.senses(k[0], k[1]) if s in self.lex.senses_] if k[1] else []
            toks = [tuple(self.lex.senses_[s]["tokens"]) for s in senses]
            self._senses[k] = (toks[0] if toks else None, set(toks))
        return self._senses[k]

    def similar(self, a, b):
        """b says what a said: the same word, or a word that mostly means something
        a can mean (big ~ large; pretty ~ beautiful, one concept in the dictionary).
        Sharing a rare sense is not enough: "call" can mean shout, as "cry" can,
        but "called" for "cried" changes the meaning; "woman" for "girl" too."""
        if a[0].lower() == b[0].lower():
            return True
        first_b = self.spellings(b)[0]
        return first_b is not None and first_b in self.spellings(a)[1]

    def compare(self, src, out, context):
        """src, out: content words; context: lowercase words of the whole document
        (what a pronoun can have meant).  -> dict of matched / changed / lost /
        added / referents."""
        out = list(out)
        matched, lost = 0, []
        for s in src:
            hit = next((k for k, o in enumerate(out) if self.similar(s, o)), None)
            if hit is None:
                lost.append(s)
            else:
                matched += 1
                out.pop(hit)
        referents = [o for o in out if o[1] in ("PROPN", "NOUN") and o[0].lower() in context]
        added = [o for o in out if o not in referents]
        changed = []
        for s in list(lost):            # a lost word and an added word of the same kind: a change
            k = next((i for i, o in enumerate(added) if o[1] == s[1] or {o[1], s[1]} <= {"NOUN", "PROPN"}), None)
            if k is not None:
                changed.append((s, added.pop(k)))
                lost.remove(s)
        return {"matched": matched, "content": len(src), "changed": changed, "lost": lost,
                "added": added, "referents": referents}

    @staticmethod
    def issue(r):
        """A flag for the corpus filter, or None if the meaning survived."""
        parts = []
        if r["changed"]:
            parts.append("changed " + ", ".join(f"{s[0]} -> {o[0]}" for s, o in r["changed"]))
        if r["lost"]:
            parts.append("lost " + ", ".join(s[0] for s in r["lost"]))
        if r["added"]:
            parts.append("added " + ", ".join(o[0] for o in r["added"]))
        return ("meaning changed in the round trip: " + "; ".join(parts)) if parts else None

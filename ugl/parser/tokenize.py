"""Sentence splitting and Penn/UD-style word tokenization (pure Python).

Splits clitics the way the UD English treebanks do:
    "don't" -> do n't      "John's" -> John 's      "I'll" -> I 'll
"""
import re

ABBREV = {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e", "u.s",
          "u.k", "inc", "ltd", "co", "corp", "jan", "feb", "mar", "apr", "jun", "jul", "aug",
          "sep", "sept", "oct", "nov", "dec", "no", "fig", "al", "approx", "dept", "est", "gen",
          "gov", "mt", "ft", "lt", "col", "sgt", "capt", "rev", "vol", "ca"}

_TOKEN = re.compile(r"""
    (?:[A-Za-z]\.){2,}                    # U.S.  e.g.
  | [^\W\d_]+\d[^\W_]*(?:-\d+)?          # letters then digits are one word: RC1  MP3  H2O
  | [^\W\d_]+-\d+                        # COVID-19  B-52
  | \d+(?:[.,:]\d+)*(?:st|nd|rd|th|s)?    # 1,823  3.14  12:30  1990s  21st
  | [^\W\d_]+(?:[-'’][^\W\d_]+)*          # words, hyphenated words, o'clock
  | °[CFK](?![A-Za-z])|°                 # 100 °C: the unit is one word
  | \.\.\.|--|[^\w\s]                     # punctuation
  | \S
""", re.X | re.U)

_CLITIC = re.compile(r"^(.+?)(n't|'s|'re|'ve|'ll|'d|'m|’s|’re|’ve|’ll|’d|’m|n’t)$", re.I)
_SPECIAL = {"can't": ["ca", "n't"], "won't": ["wo", "n't"], "cannot": ["can", "not"],
            "shan't": ["sha", "n't"], "gonna": ["gon", "na"], "wanna": ["wan", "na"],
            "gotta": ["got", "ta"]}


def split_sentences(text):
    """Split on . ! ? followed by space + capital/quote/digit, skipping
    abbreviations.  A line break (paragraph break) always ends a sentence."""
    if "\n" in text:
        return [s for line in text.split("\n") for s in split_sentences(line)]
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    out, start = [], 0
    for m in re.finditer(r"[.!?]+[\"')\]]*(?=\s+[\"'(\[]?[A-Z0-9])", text):
        end = m.end()
        before = text[start:m.start()].split()
        last = before[-1].lower().rstrip(".") if before else ""
        if m.group().startswith(".") and (last in ABBREV or (len(last) == 1 and last.isalpha())):
            continue
        out.append(text[start:end].strip())
        start = end
    if text[start:].strip():
        out.append(text[start:].strip())
    return out


def tokenize(sentence):
    toks = []
    for m in _TOKEN.finditer(sentence):
        w = m.group()
        low = w.lower()
        if low in _SPECIAL:
            parts = _SPECIAL[low]
            toks.append(w[: len(parts[0])])
            toks += parts[1:]
            continue
        c = _CLITIC.match(w)
        if c and len(c.group(1)) > 0:
            toks += [c.group(1), c.group(2)]
            continue
        toks.append(w)
    # re-attach "'" + s that the word regex split ("John" "'" "s")
    out, i = [], 0
    while i < len(toks):
        if toks[i] in ("'", "’") and i + 1 < len(toks) and toks[i + 1].lower() in ("s", "re", "ve", "ll", "d", "m") and out:
            out.append(toks[i] + toks[i + 1]); i += 2
            continue
        out.append(toks[i]); i += 1
    return out

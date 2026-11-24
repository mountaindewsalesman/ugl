"""Token features shared by training (torch) and inference (numpy)."""
import re


def shape(w):
    if re.fullmatch(r"\d+([.,:]\d+)*", w):
        return "d"
    if not any(c.isalnum() for c in w):
        return "p"
    if w.isupper():
        return "X" if len(w) > 1 else "X1"
    if w[:1].isupper():
        return "Xx"
    if any(c.isdigit() for c in w):
        return "xd"
    if "-" in w:
        return "x-"
    return "x"


def norm(w):
    w = w.lower()
    w = re.sub(r"\d", "0", w)
    return w.replace("’", "'")


def affixes(w):
    n = norm(w)
    return {"pre2": n[:2], "suf2": n[-2:], "suf3": n[-3:], "suf4": n[-4:]}


AFFIX_KEYS = ["pre2", "suf2", "suf3", "suf4"]

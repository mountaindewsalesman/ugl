"""Glue between the playground page and the real `ugl` package.

Runs inside Pyodide (and under CPython for testing).  It only calls the
library's own decoder, tokenizer, lexicon and CLI gloss function, plus the
reserved/modifier tables in ugl.core, and returns plain JSON for the page.
"""
import json

import ugl
from ugl.__main__ import gloss_line
from ugl.core import (ADJ_MODS, NOUN_MODS, RESERVED_NOUNS, RESERVED_VERBS, VERB_KIND,
                      VERB_MODS, WORD_RE, tokenize)

CLASS_NAMES = {"n": "noun", "v": "verb", "a": "adjective / adverb", "nv": "verb used as a noun"}
MODS_FOR = {"n": NOUN_MODS, "nv": NOUN_MODS, "v": VERB_MODS, "a": ADJ_MODS, "name": NOUN_MODS}
KIND_NAMES = {
    "core": "core verb",
    "relmark": "relative-clause marker",
    "rel": "relation",
    "logic": "logic connective",
    "quant": "quantifier",
}

U = None


def init():
    global U
    U = ugl.UGL(wsd=False)
    return json.dumps({
        "ugl_version": ugl.__version__,
        "dictionary_version": U.version,
        "vocab_size": len(U.tokenizer),
    })


def _mods(cls, letters):
    table = MODS_FOR.get(cls, {})
    return [{"letter": c, "meaning": table.get(c, "(not a modifier of this class)")} for c in letters]


def _concept(cls, idx):
    """Lemma, gloss and dictionary details for class letter + index."""
    lex = U.lexicon
    base = "v" if cls == "nv" else cls
    i = int(idx)
    out = {"id": f"{base}{i}", "lemma": lex.lemma_of(base, i)}
    if base == "n" and i in RESERVED_NOUNS:
        out["reserved"] = "reserved noun"
    elif base == "v" and i in RESERVED_VERBS:
        out["reserved"] = KIND_NAMES.get(VERB_KIND[i], VERB_KIND[i])
    else:
        e = lex.ids.get(f"{base}{i}")
        if e:
            out.update(sense=e.get("sense"), gloss=e.get("gloss"), freq=e.get("freq"))
        else:
            out["missing"] = True
    return out


def explain_token(t):
    m = WORD_RE.match(t)
    if m:
        cls, idx, mods = m.groups()
        d = {"token": t, "kind": "word", "cls": cls, "class": CLASS_NAMES[cls]}
        d.update(_concept(cls, idx))
        d["mods"] = _mods(cls, mods)
        return d
    if t.startswith("\\"):
        d = {"token": t, "kind": "close", "class": "closing tag"}
        d["closes"] = _concept("v", t[2:]) if t[2:].isdigit() else {"id": t[1:], "lemma": "?"}
        return d
    if t.startswith("#"):
        return {"token": t, "kind": "num", "class": "number literal", "lemma": t[1:].replace("_", " ")}
    if t.startswith("'"):
        body, _, mods = t[1:].rpartition("'")
        return {"token": t, "kind": "name", "class": "name literal",
                "lemma": body.replace("_", " "), "mods": _mods("name", mods)}
    if len(t) > 2 and t[1] == "?":
        return {"token": t, "kind": "unk", "class": f"unknown {CLASS_NAMES.get(t[0], t[0])}",
                "lemma": t[2:].replace("_", " ")}
    return {"token": t, "kind": "bad", "class": "not a UGL token"}


def analyze(text):
    """UGL text (one sentence per line) -> per-sentence tokens, gloss, English, ids."""
    tok = U.tokenizer
    sentences = []
    good_lines = []
    for line in text.strip().split("\n"):
        line = " ".join(line.split())
        if not line:
            continue
        s = {"ugl": line, "tokens": [explain_token(t) for t in line.split()]}
        try:
            tokenize(line)
        except ValueError as e:
            s["error"] = str(e)
            sentences.append(s)
            continue
        s["readable"] = U.humanize(line)
        s["gloss"] = gloss_line(U.lexicon, line)
        try:
            s["back"] = U.decode(line)
        except Exception as e:  # the CLI prints these the same way
            s["back_error"] = f"{type(e).__name__}: {e}"
        per_token = [tok.encode_token(t) for t in line.split()]
        s["ids"] = [{"token": t, "ids": ids, "pieces": [tok.vocab[i] for i in ids]}
                    for t, ids in zip(line.split(), per_token)]
        s["roundtrip_ok"] = tok.decode(tok.encode(line)) == line
        good_lines.append(line)
        sentences.append(s)
    joined = "\n".join(good_lines)
    all_ids = tok.encode(joined, bos=True, eos=True) if good_lines else []
    return json.dumps({
        "sentences": sentences,
        "ids": all_ids,
        "pieces": [tok.vocab[i] for i in all_ids],
    })


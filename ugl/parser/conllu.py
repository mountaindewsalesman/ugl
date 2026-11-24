"""Minimal CoNLL-U reader (Universal Dependencies treebank format)."""


def read_conllu(path):
    """Yield sentences as lists of dicts with keys
    form, lemma, upos, xpos, feats, head (int), deprel.
    Multi-word token lines and empty nodes are skipped."""
    sent = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                if sent:
                    yield sent
                sent = []
                continue
            if line.startswith("#"):
                continue
            cols = line.split("\t")
            if "-" in cols[0] or "." in cols[0]:
                continue
            sent.append({"form": cols[1], "lemma": cols[2], "upos": cols[3], "xpos": cols[4],
                         "feats": cols[5], "head": int(cols[6]), "deprel": cols[7].split(":")[0]
                         if cols[7] not in KEEP_SUBTYPES else cols[7]})
    if sent:
        yield sent


# relation subtypes that carry meaning UGL needs; all others are collapsed
KEEP_SUBTYPES = {"nsubj:pass", "aux:pass", "obl:tmod", "obl:npmod", "nmod:poss", "nmod:tmod",
                 "acl:relcl", "compound:prt", "det:predet", "csubj:pass", "obl:agent"}

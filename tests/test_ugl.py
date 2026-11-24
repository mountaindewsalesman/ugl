"""Run:  python -m pytest tests  (or: python tests/test_ugl.py)
Structural tests: they check the *shape* of the output (reserved words,
modifiers, scopes) and that every encoding parses back, not exact lexical
numbers, which depend on the corpus the lexicon was built from."""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ugl
from ugl.decoder import Parser
from ugl.core import tokenize

U = ugl.load()
# token numbers are only meaningful within one dictionary version: tests that
# spell out tokens by number use this frozen one
U02 = ugl.load("ugl-dict-0.2.0.json.gz")
ENC = U.encoder
DEC = U.decoder
TOK = U.tokenizer


def enc(s):
    return ENC.encode(s)[0]


def balanced(u):
    stack = []
    for t in u.split():
        if t.startswith("\\"):
            continue
    Parser(tokenize(u)).parse()   # raises if scopes don't close properly
    return True


N = r"(?:'[A-Za-z]+'|n\d+)"      # a name: numbered noun (frequent) or literal

CASES = [
    ("At nine, I ate the apple.", [r"^#9 v20 n1 v\d+p n\d+t \\v20$"]),
    ("I ate quickly.", [r"^n1 v\d+p a\d+$"]),
    ("The dog was bitten by the cat.", [r"^n\d+t v\d+p n\d+t$"]),
    ("John did not eat.", [rf"^{N} v\d+px$"]),
    ("Every dog barks.", [r"^n(\d+) v9 n\1 v\d+ \\v9$"]),
    ("I stayed because it rained.", [r"^v7 .* \\v7 n1 v\d+p \\v7$"]),
    ("Mary said that John left.", [rf"^{N} v\d+p {N} v\d+p \\v\d+$"]),
    ("I want to eat.", [r"^n1 v(\d+) n1 v\d+ \\v\1$"]),
    ("I ate without John.", [r"v52x"]),
    ("John's dog is big.", [rf"{N}y", r"v1 a\d+"]),
    ("We will have eaten.", [r"^n1s v\d+fk$"]),
    ("Did you eat?", [r"^n2 v\d+pq$"]),
    ("The man who left saw me.", [r"v3 n(\d+)t v\d+p \\v3", r"v\d+p n1$"]),
    # semi-modals carry the modal letter, not a verb ("have to" = must)
    ("I have to go.", [r"^n1 v\d+u$"]),
    ("We had to wait.", [r"^n1s v\d+pu$"]),
    ("She is able to swim.", [r"^n4 v\d+c$"]),
]


def test_simplify_splits_asides_and_and():
    out = ENC.encode("My mother, who lives in Ohio, is ill.")
    assert len(out) == 2 and "v3" not in out[0], out          # aside became its own sentence
    out = ENC.encode("The man who left saw me.")
    assert len(out) == 1 and "v3" in out[0], out              # restrictive clause stays
    out = ENC.encode("I ate and she slept.")
    assert len(out) == 2 and "v4" not in " ".join(out), out
    out = ENC.encode("I ate but she slept.")
    assert len(out) == 1 and out[0].startswith("v6"), out     # "but" is a relation: kept


def test_passive_subject_is_salient():
    # "it" = the bridge (surface subject of the passive), not the agent
    a, b = ENC.encode("The bridge was destroyed by the flood. The town rebuilt it.")
    bridge = re.search(r"(n\d+)t v\d+p (n\d+)t$", a).group(2)
    assert b.endswith(bridge + "t"), (a, b)


def test_cases():
    fails = []
    for en, pats in CASES:
        u = enc(en)
        for p in pats:
            if not re.search(p, u):
                fails.append(f"{en!r} -> {u!r} does not match {p!r}")
        balanced(u)
        DEC.decode(u)
    assert not fails, "\n".join(fails)


def test_roundtrip_readable():
    for en, _ in CASES:
        back = DEC.decode(enc(en))
        assert back and "<" not in back, (en, back)


def test_tokenizer_roundtrip():
    for en, _ in CASES:
        u = enc(en)
        assert TOK.decode(TOK.encode(u)) == u
    u = "'Céline_Arnauld'y #1,823 n?'by-product'j n100st \\v9"
    assert TOK.decode(TOK.encode(u)) == u


def test_literals_take_modifiers():
    from ugl.core import add_mods, literal, parse_literal
    assert literal("unk", "DPR", "j") == "n?'DPR'j"
    assert add_mods("n?'DPR'", "j") == "n?'DPR'j"
    assert parse_literal("n?'by-product'j") == ("unk", "n", "by-product", "j")
    assert parse_literal("n?blorft") == ("unk", "n", "blorft", "")      # old form still reads


def test_parentheses_ignored():
    assert enc("John (a doctor) ate.") == enc("John ate.")


def test_humanize():
    h = U02.humanize
    assert h("n4 v2 n2344s #3") == "someone has apples three"
    assert U02.decode("n4 v2 n2344s #3") == "Someone has 3 apples."
    assert h(r"#9 v20 n1 v444p n2344t \v20") == r"nine time I ate the apple \time"
    assert h("n136t v117p n1") == "the man saw me"


def test_public_api():
    assert ugl.decode(ugl.encode("I ate the apple.")[0]) == "I ate the apple."
    ids = ugl.to_ids("Every dog barks.")
    assert ugl.from_ids(ids) == ugl.encode("Every dog barks.")[0]


if __name__ == "__main__":
    for en, _ in CASES:
        u = enc(en)
        print(f"{en:35} {u:45} {DEC.decode(u)}")
    test_cases()
    test_roundtrip_readable()
    test_tokenizer_roundtrip()
    test_parentheses_ignored()
    test_humanize()
    test_public_api()
    print("all tests passed")

"""Back-references (SPEC rule 15, ugl/refs.py)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ugl  # noqa: E402
from ugl.refs import compress, expand  # noqa: E402

U = ugl.load(wsd=False)
P = U.lexicon.is_proper


def both(lines):
    c = compress(lines, P)
    assert expand(c, P) == lines, (lines, c)
    return c


def test_name_and_possessive():
    assert both(["'Tom' v1p a13", "'Tom' v153p v30 n181t 'Tom'y \\v30"]) == \
        ["'Tom' v1p a13", "@1 v153p v30 n181t @2y \\v30"]


def test_definite_nouns_refer_indefinite_do_not():
    # "a dog ... a dog" are not the same dog; "the dog ... the dog" is
    assert both(["n909 v1p a13", "n909 v1p a13"]) == ["n909 v1p a13", "n909 v1p a13"]
    assert both(["n909t v1p a13", "n909t v1p a13"]) == ["n909t v1p a13", "@1 v1p a13"]
    assert both(["n909ts v1p a13", "n909t v1p a13"]) == ["n909ts v1p a13", "n909t v1p a13"]   # plural != singular


def test_fixed_words_and_attributes_never_refer():
    lines = ["n1 v153p n2", "n1 v153p n2", "n200 n180j", "n200 n180j"]
    assert both(lines)[1] == "n1 v153p n2"
    assert "n180j" in both(lines)[3]


def test_groups():
    lines = ["v4 'Tom' \\v4 'Lily' \\v4 v153p", "'Tom' v1p a13", "v4 'Tom' \\v4 'Lily' \\v4 v268p",
             "v4y 'Tom' \\v4 'Lily' \\v4 n181 v1p a13"]
    c = both(lines)
    assert c[1] == "@3 v1p a13"                       # Tom: then Lily, then the group came after him
    assert c[2] == "@2 v268p"                         # the group, behind Tom
    assert c[3].startswith("@1y ")                    # the group's possessive


def test_only_nine_back():
    lines = [f"'P{i}' v1p a13" for i in range(12)] + ["'P0' v1p a13", "'P11' v1p a13"]
    c = both(lines)
    assert c[12] == "'P0' v1p a13"                    # 12th most recent: written out
    assert c[13] == "@2 v1p a13"


def test_refs_round_trip_through_tokenizer_and_decoder():
    text = "Tom saw a big dog. He liked the dog and gave it his ball. The dog ran to Tom."
    full = U.encode(text)
    short = U.encode(text, refs=True)
    assert any("@" in l for l in short)
    flat = [l for s in full for l in s.split("\n")]
    assert expand(short, P) == flat
    ids = U.tokenizer.encode("\n".join(short))
    assert U.tokenizer.decode(ids).split("\n") == short
    assert U.decode("\n".join(short)) == U.decode("\n".join(flat))

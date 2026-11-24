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
    assert len(out) == 2 and "v4" not in " ".join(out).split(), out
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


# ------------------------------------------------------------------ bug families
# each case is one member of a family reported from the playground; the fix is
# structural, so every member must hold (decoded English shown in comments)
FAMILIES = [
    # unmarked participial clause: no invented "when" / "because"
    ("Running home, he fell.", lambda u, e: "v20" not in u.split() and "v7" not in u.split()),
    ("How can we model emissions, observing their impacts?", lambda u, e: "when" not in e.lower()),
    # coordinated "to" complements stay infinitives
    ("I like to eat food and run.", lambda u, e: "to eat" in e and "that" not in e),
    ("She wants to sing or dance.", lambda u, e: "to sing or dance" in e),
    # standalone quantifiers refer back
    ("He doesn't like cats and dogs because both are smelly.", lambda u, e: "both" in e and "something" not in e),
    ("I bought the apples. Most were rotten.", lambda u, e: "Most of the apples" in e),
    ("The dogs all barked.", lambda u, e: "All the dogs" in e),
    # number words used as nouns
    ("It spans tens of kilometers.", lambda u, e: "#10s" in u and "tens of" in e),
    ("Hundreds attended.", lambda u, e: "#100s" in u),
    ("Two hundred thousand people came.", lambda u, e: "#200000" in u),
]


def test_families():
    fails = []
    for en, ok in FAMILIES:
        lines = ENC.encode(en)
        u = "\n".join(lines)
        e = DEC.decode(u)
        if not ok(u, e):
            fails.append(f"{en!r} -> {u!r} -> {e!r}")
    assert not fails, "\n".join(fails)


def test_quality_flags():
    lines = U.encode_lines("The flurbish gizmo wanged.")
    assert any("not in the dictionary" in i for l in lines for i in l.issues)
    lines = U.encode_lines("It was founded in 1877.")
    assert any("refers to nothing" in i for l in lines for i in l.issues)
    lines = U.encode_lines("The dog barked.")
    assert not any(l.issues for l in lines), lines


def test_tree_view():
    rows = U.humanizer.tree(r"#9 v20 n1 v444p n2344t \v20")
    assert [r["depth"] for r in rows] == [0, 1, 0] and rows[-1]["ugl"] == r"\v20"


FAMILIES_2 = [
    # unit symbols are one word and a real unit
    ("Water boils at 100 °C.", lambda u, e: "°" not in u and "'C'" not in u),
    # reduced passive clauses: the subject is what undergoes the action
    ("Polystyrene flows if heated.", lambda u, e: "someone heats" in e.lower()),
    # approximate numbers keep their adverb
    ("It costs about 100 dollars.", lambda u, e: "about" in e or "approximately" in e),
    # extraposed clause subjects are kept
    ("It will take me a while to read the book.", lambda u, e: "read" in e),
]


def test_families_2():
    fails = []
    for en, ok in FAMILIES_2:
        u = "\n".join(ENC.encode(en))
        e = DEC.decode(u)
        if not ok(u, e):
            fails.append(f"{en!r} -> {u!r} -> {e!r}")
    assert not fails, "\n".join(fails)


def test_nothing_dropped_silently():
    # stacked compounds and nested name words reach the UGL
    for en in ("Its glass transition temperature is high.", "They met the United Nations Security Council."):
        lines = U.encode_lines(en)
        assert not any("was dropped" in i for l in lines for i in l.issues), lines


FAMILIES_3 = [
    # loose verbless side phrases are kept as their own statement
    ("Do not buy tickets yet - a lesson I learned.", lambda u, e: "lesson" in e),
    # adjective complements keep their dependents
    ("The Sunnis remain committed to nationalism.", lambda u, e: "nationalism" in e),
    # adjectives with complements become clauses about the head
    ("Please discuss the issues related to your area.", lambda u, e: "area" in e),
    # joined adjectives
    ("We need an interactive and lively discussion.", lambda u, e: "interactive" in e and "lively" in e),
    # words of address
    ("Hello Louise, the meeting moved.", lambda u, e: "Louise" in e),
    # negative commands
    ("Don't go there.", lambda u, e: e.startswith("Do not ") and " v" in " " + u and "ox" in u),
    # content clauses keep their own subject
    ("This is the fact that the company is overpriced.", lambda u, e: "company" in e),
    # verbless fragments keep their phrases
    ("Grateful for any help.", lambda u, e: "help" in e),
]


def test_families_3():
    fails = []
    for en, ok in FAMILIES_3:
        u = "\n".join(ENC.encode(en))
        e = DEC.decode(u)
        if not ok(u, e):
            fails.append(f"{en!r} -> {u!r} -> {e!r}")
    assert not fails, "\n".join(fails)


def test_source_spans():
    lines = U.encode_lines("My mother, who lives in Ohio, is ill.")
    src = lines[0].source
    assert [src[a:b] for a, b in lines[0].spans] == ["My mother", "is ill"]
    assert [src[a:b] for a, b in lines[1].spans] == ["lives in Ohio"]


def test_topic_resolves_pronouns():
    lines = U.encode_lines("It was founded in 1877.", topic="Sacred Heart Church")
    assert not any("refers to nothing" in i for l in lines for i in l.issues)


FAMILIES_4 = [
    # "as" without a subject of its own is a role, not a cause
    ("It was shown as extending to the river.", lambda u, e: "because" not in e),
    # multiword prepositions, whatever the parser tags their first word
    ("Thawai was a village prior to independence.", lambda u, e: "before" in e),
    # "Paris, France": a place inside the next one
    ("He was born in Paris, France.", lambda u, e: "in France" in e),
]


def test_families_4():
    fails = []
    for en, ok in FAMILIES_4:
        u = "\n".join(ENC.encode(en))
        e = DEC.decode(u)
        if not ok(u, e):
            fails.append(f"{en!r} -> {u!r} -> {e!r}")
    assert not fails, "\n".join(fails)


FAMILIES_5 = [
    # object + a second noun saying what it is: nothing dropped
    ("He made a jelly that he dubbed styrol oxide.", lambda u, e: "oxide" in e),
    ("We elected the woman president.", lambda u, e: "president" in e),
    # "it" never means a person
    ("Simon found that it had thickened.", lambda u, e: not e.startswith("Simon found that Simon")),
    ("Paris is large. It has many museums.", lambda u, e: e.count("Paris") == 2),
]


def test_families_5():
    fails = []
    for en, ok in FAMILIES_5:
        u = "\n".join(ENC.encode(en))
        e = DEC.decode(u)
        if not ok(u, e):
            fails.append(f"{en!r} -> {u!r} -> {e!r}")
    assert not fails, "\n".join(fails)

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
    ("Thawai was a village prior to independence.", lambda u, e: "before" in e.lower()),
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


FAMILIES_6 = [
    # dates: one canonical literal however English phrases them
    ("He was born on March 3, 1990.", lambda u, e: "#1990-03-03" in u),
    ("He was born on 3 March 1990.", lambda u, e: "#1990-03-03" in u),
    ("The violence lasted from February 2025 – February 2026.", lambda u, e: "#2025-02" in u and "#2026-02" in u),
    # reported clauses survive the passive; "by <date>" is not an agent
    ("By 14 February, it was reported that the houses had burned.", lambda u, e: "houses" in e and "#--02-14" in u),
]


def test_families_6():
    fails = []
    for en, ok in FAMILIES_6:
        u = "\n".join(ENC.encode(en))
        e = DEC.decode(u)
        if not ok(u, e):
            fails.append(f"{en!r} -> {u!r} -> {e!r}")
    assert not fails, "\n".join(fails)


def test_round_trip_stable():
    # decode -> encode gives the same UGL back (relations in canonical order,
    # possessives definite, relation phrases fronted when an object could take them)
    for en in ("I ate the apple at nine in the kitchen.", "In the kitchen, I ate the apple at nine.",
               "Simon read his book on the bank of the river.", "This caused an arson of houses."):
        a = U.encode(en)
        assert U.encode(U.decode("\n".join(a))) == a, en
    assert U.encode("I ate the apple at nine in the kitchen.") == U.encode("In the kitchen, I ate the apple at nine.")


FAMILIES_7 = [
    # coordinated modifiers are one group after the head: nothing dropped, nothing distributed
    ("I wrote down their cell and car numbers.",
     lambda u, e: re.search(r"n\d+st v4 n\d+j \\v4 n\d+j \\v4", u) and "cell and car numbers" in e and "of them" not in e),
    ("The red and blue cars left.", lambda u, e: re.search(r"v4 a\d+ \\v4 a\d+ \\v4", u) and "red and blue cars" in e),
    ("Red, white and blue flags flew.", lambda u, e: "red, white and blue" in e.lower()),
    ("We waited two or three days.", lambda u, e: r"v5 #2 \v5 #3 \v5" in u),
    # a coordinated clause after an object is still a clause, not a modifier group
    ("She told him that it rained and that he left.", lambda u, e: "rained" in e and "left" in e),
    # tense sits on the first word of the verb group, whichever tagger head says so
    ("The cars left.", lambda u, e: re.search(r"v\d+p$", u)),
    ("He has left.", lambda u, e: re.search(r"v\d+k$", u)),
    # grammatical prefixes: non- on an adjective is "not"; non- noun = b; anti- = i
    ("Non-toxic paint is safe.", lambda u, e: "?" not in u),
    ("Non-members cannot vote.", lambda u, e: "?" not in u and "member" in e.lower()),
    ("The anti-war protest grew.", lambda u, e: re.search(r"n\d+ij", u) and "anti-war protest" in e),
    ("He takes anti-inflammatory drugs.", lambda u, e: re.search(r"a\d+i\b", u) and "anti-inflammatory" in e),
]


def test_families_7():
    fails = []
    for en, ok in FAMILIES_7:
        u = "\n".join(ENC.encode(en))
        e = DEC.decode(u)
        if not ok(u, e):
            fails.append(f"{en!r} -> {u!r} -> {e!r}")
    assert not fails, "\n".join(fails)


def test_nested_closing_tags_pair_innermost_first():
    # like parentheses: the first \v104 closes the inner "said"
    p = Parser(tokenize(r"n4 v104p n4 v104p n3t v3182p \v104 \v104"))
    p.parse()
    assert p.opener == {6: 3, 7: 1}, p.opener


def test_modifier_groups_round_trip():
    for en in ("I wrote down their cell and car numbers.", "The red and blue cars left.",
               "We waited two or three days.", "The anti-war protest grew."):
        a = U.encode(en)
        assert U.encode(U.decode("\n".join(a))) == a, (en, a, U.decode("\n".join(a)))


# ---- round-trip families (tools/roundtrip_report.py on Wikipedia text)
def lines(text, topic=None):
    return U.encode_lines(text, topic=topic)


def test_adverb_before_main_verb():
    # between verb and object an adverb reads as part of the object: "bore frequently quotes"
    e = U.decode("\n".join(U.encode("His letters often bore quotes from Tukaram.")))
    assert re.search(r"often bore|frequently bore", e), e
    e = U.decode("\n".join(U.encode("She will quickly finish the report.")))
    assert re.search(r"will quickly \w+ the report", e), e
    e = U.decode("\n".join(U.encode("He picked the box up.")))   # particles stay after the verb
    assert not re.search(r"\bup pick", e), e


def test_adverb_word_means_the_concept():
    # "currently" is the adverb of 'present'; "presently" would read back as 'soon'
    e = U.decode("\n".join(U.encode("Postiglioni currently plays for Beauvais Rugby Club.")))
    assert "currently" in e, e


def test_canonical_modifier_and_relation_order():
    # one UGL per meaning, whatever the English order
    a = U.encode("Both parents died in the 1902 plague epidemic.")
    b = U.encode("Both parents died in the plague epidemic of 1902.")
    assert a == b, (a, b)
    a = U.encode("The tree grows in thickets in Brazil.")
    b = U.encode("The tree grows in Brazil in thickets.")
    assert a == b, (a, b)


def test_reserved_noun_keeps_noun_modifiers():
    # "all the Kuki tribals": the attribute noun / name on "things" must be written
    e = DEC.decode("n3s a760 'Kuki'j v1 a344")
    assert "Kuki" in e and "tribal" in e.lower(), e
    assert DEC.decode(r"n4 v107p n3se v3 n3s v319p \v3").count("that that") == 0


def test_pronoun_gender_and_chains():
    ls = lines("Rina Sawayama is a singer. Her older brother installed Ableton on her laptop, "
               "and she made her first track.", "Rina Sawayama")
    body = " ".join(l.ugl for l in ls[1:])
    assert "n390ty" not in body and "'Ableton'y" not in body, body     # not the brother's / Ableton's laptop
    assert body.count("'Rina_Sawayama'") >= 3, body


def test_nested_possessor_pronoun_resolves():
    ls = lines("K. B. Hedgewar was a doctor. His health deteriorated in later years of his life.")
    assert "n3ty" not in ls[1].ugl and ls[1].ugl.count("'K_B_Hedgewar'y") == 2, ls[1].ugl


def test_one_spelling_per_name():
    assert U.encode("K. B. Hedgewar was a doctor.") == U.encode("K B Hedgewar was a doctor.")


def test_verbless_lines_are_flagged():
    ls = lines("Pathétique Sonata – Ludwig van Beethoven")
    assert any("not a sentence" in i for l in ls for i in l.issues), ls


def test_calendar_prepositions():
    assert DEC.decode(r"#2017 v20 'Ann' v154p \v20").startswith("In 2017") or " in 2017" in DEC.decode(r"#2017 v20 'Ann' v154p \v20")
    assert " on 21 June 1940" in DEC.decode(r"#1940-06-21 v20 'Ann' v154p \v20") or \
        DEC.decode(r"#1940-06-21 v20 'Ann' v154p \v20").startswith("On 21 June 1940")


def test_same_reading_treats_homonyms_as_equal():
    # n545 / n1188 are both written "end": the round trip cannot tell them apart
    assert U.same_reading(["n545t v1 a5"], ["n1188t v1 a5"])
    assert not U.same_reading(["n545t v1 a5"], ["n545t v1p a5"])        # tense is grammar


def test_no_double_negation_or_degree_on_predicates():
    # "not" / "more" on a predicate adjective belong to one place only
    u = U.encode("Ferric minerals were not present.")[0]
    assert "v1px" in u and not re.search(r"a\d+[a-z]*x", u), u
    u = U.encode("She is more beautiful.")[0]
    assert re.search(r"v1 a\d+r$", u) and "n7" not in u, u


def test_lexical_comparatives_get_no_extra_degree():
    # "later" is its own word (subsequently); "faster" is fast + r
    assert not re.search(r"a\d+r$", U.encode("He later withdrew.")[0])
    assert re.search(r"a\d+r$", U.encode("He ran faster.")[0])


def test_comparatives_read_as_english():
    # a comparative of the concept, in the word English uses most for it
    # (beautiful and pretty are one concept in the dictionary; big and large too)
    for en, want in (("He ran faster.", ("faster",)), ("She is more beautiful.", ("more beautiful", "prettier")),
                     ("The bigger dog barked.", ("bigger", "larger")), ("She spoke more carefully.", ("more carefully",))):
        e = U.decode("\n".join(U.encode(en)))
        assert any(w in e for w in want), (en, e)


def test_clause_after_preposition_is_english():
    e = U.decode("\n".join(U.encode("Hedgewar travelled across India to set up shakhas and inspire young people.")))
    assert "across India to establish" in e and "for Hedgewar" not in e, e
    e = U.decode("\n".join(U.encode("After matriculating, he went to Kolkata.")))
    assert "after matriculating" in e, e


def test_split_off_remention_keeps_be():
    ls = U.encode("Access is from the third level of the tower, which is at a height of 21.6 meters.")
    assert any(re.search(r"n\d+t v1 \\v32$", l) for l in ls), ls


def test_relative_pronouns_for_people():
    e = DEC.decode(r"n4834 v1 n152 v3 n152t v52 n4834 v143 n11602 \v52 \v3")
    assert "with whom" in e, e


def test_dates_without_year_take_on():
    assert "On 11 March" in DEC.decode(r"#--03-11 v20 n4 v1424p n145s #2 \v20")


def test_modifiers_after_a_group_belong_to_the_group():
    # SPEC 13: "who served" is said of the writer AND the activist
    u = U.encode("Elizabeth Rowley is a Canadian writer and political activist who served as the leader of the party.")[0]
    # the clause follows the group's last closing tag, and its gap is the whole group
    assert re.search(r"n\d+ a\d+ \\v4 v3 .*v4 n\d+t \\v4 n\d+t \\v4 v\d+p", u), u
    e = U.decode(u)
    assert "writer and political activist who served" in e or "author and political activist who served" in e, e
    assert U.encode(e) == [u]


def test_conjuncts_share_the_determiner():
    u = U.encode("The realization requires the close involvement and cooperation of the knowing subject.")[0]
    assert re.search(r"v4 n\d+t a\d+ \\v4 n\d+t \\v4", u), u
    assert U.encode(U.decode(u)) == [u]


def test_name_modifiers_keep_every_word():
    for en, name in (("The Riverside High School mascot is Cyclones.", "Riverside_High_School"),):
        ls = lines(en)
        assert name in ls[0].ugl and not any("dropped" in i for i in ls[0].issues), ls


def test_number_ranges_and_counted_compounds():
    ls = lines("Sprouting occurs within 40–60 days.")
    assert "v51 #40" in ls[0].ugl and not any("dropped" in i for i in ls[0].issues), ls
    assert "40 to 60" in U.decode(ls[0].ugl)
    ls = lines("Obukhov signed a 2-year contract.")
    assert "y #2" in ls[0].ugl and not any("dropped" in i for i in ls[0].issues), ls
    for l in ("Sprouting occurs within 40–60 days.", "Obukhov signed a 2-year contract."):
        a = U.encode(l)
        assert U.encode(U.decode("\n".join(a))) == a


def test_entities_compare_by_identity():
    # coreference ranks candidate antecedents; value equality on entities
    # recursed forever once a relative clause pointed back at its own head
    # (OpenAssistant text, found by scripts/fetch_sources.py + roundtrip_report)
    para = ('To use this for "pushing" and "pulling", we\'ll put two images through the neural network. '
            "Let's say the images are of a cat and a dog, so we want to increase the distance between the two. "
            "We pick one neuron weight from the network and make it a variable `x`, then construct a function that "
            "calculates the output of the network based on it and all the other parameters; let's call it `N(x)`. "
            "The distance between the cat and dog outputs would be `f(x) = N(cat) - N(dog)`. (Of course, the real "
            "output would have more than 1 dimension, but we're simplifying.) We now want to nudge the weight such "
            "that it moves the two outputs slightly further apart. For that, we can simply take the derivative! "
            "If `f'(x)` is positive, that means that increasing the weight will move them further apart, so we "
            "should do that. If it's negative, then it'll move them closer, so we'll want to slightly decrease "
            "the weight instead. Apply this to all the neurons enough times and your network will soon converge "
            "to a pretty good cat-dog separator!")
    assert lines(para)
    from ugl.ir import Entity
    a, b = Entity("lex", "dog"), Entity("lex", "dog")
    assert a != b and a == a and len({a, b}) == 2


def test_without_clause_keeps_its_negation():
    # "without paying" = with + not, like the preposition "without a word"
    for en in ("She left without saying a word.", "She left without a word."):
        ugl_text = "\n".join(U.encode(en))
        assert "v52x" in ugl_text, ugl_text
        assert "without" in U.decode(ugl_text)


def test_over_a_time_is_during():
    assert "v23" in U.encode("Over the years, Microsoft has released hundreds of applications.")[0]
    assert "v34" in U.encode("The plane flew over the mountains.")[0]


def test_comparison_relation_is_said_than():
    assert "taller than" in U.decode(U.encode("He is taller than his brother.")[0])
    assert "effective than" in U.decode(U.encode("Showing can be more effective than telling.")[0])
    assert "against" in U.decode(U.encode("He fought against the king.")[0])


def test_date_heading_is_the_time_of_its_sentence():
    for en in ("April 29, 1991 - A cyclone in Bangladesh kills many people.",
               "August 17 1960: Gabon becomes independent."):
        ls = U.encode(en)
        assert len(ls) == 1 and ls[0].startswith("#") and " v20 " in ls[0], ls


def test_names_with_of_stay_one_name():
    ls = lines("He is a professor at the University of Michigan.")
    assert "'University_of_Michigan'" in ls[0].ugl, ls
    assert "n239y" in U.encode("The capital of France is Paris.")[0]     # ordinary "of" unchanged


def test_ellipsis_is_not_a_sentence():
    for en in ("It will not.", "Yes, it is.", "I can."):
        assert any("ellipsis" in i for l in lines(en) for i in l.issues), en
    assert not any("ellipsis" in i for l in lines("The dog is happy.") for i in l.issues)


def test_unresolved_pronoun_chain_does_not_jump_to_a_thing():
    # the earlier "they" refers to nothing known; the next "they" continues that
    # chain (flagged), it does not become "the rifles"
    ls = lines("They saw they were caught and threw down their rifles. They were terribly frightened.")
    assert any("refers to nothing" in i for i in ls[-1].issues), ls
    assert "n2246" not in ls[-1].ugl, ls
    ls = lines("The dogs barked. They were hungry.")          # a real antecedent still wins
    assert not ls[-1].issues and ls[-1].ugl.startswith("n909st"), ls


def test_linking_verbs_take_the_predicate():
    # "felt sad", not "felt to be sad"; "very" is the adjective's degree letter only
    for en, want in (("Lily felt sad and angry.", "felt sad and angry"),
                     ("Gabon becomes independent.", "becomes independent"),
                     ("Tom became a doctor.", "became a doctor"),
                     ("They felt very happy.", "felt very happy")):
        e = U.encode(en)
        d = U.decode("\n".join(e))
        assert want in d and U.encode(d) == e, (en, e, d)
    assert "n7 a29" not in U.encode("They felt very happy.")[0]        # no extra "with a very way"


def test_named_participle():
    # "a boy named Tom": the boy is named and is Tom (not "someone is Tom")
    e = U.encode("A little boy named Tom liked to play.")
    assert "n1210t n1210t v1 n9767" in e[0], e
    assert "boy named Tom" in U.decode(e[0])


def test_they_for_two_people_mentioned_apart():
    ls = lines("Tom found a box. Sue opened it. They laughed together.")
    assert not ls[-1].issues and r"v4 n9767 \v4 'Sue' \v4" in ls[-1].ugl, ls
    assert not any("refers to nothing" in i for l in lines("Tom felt sorry for what he did.") for i in l.issues)


def test_paragraph_break_ends_a_sentence():
    from ugl.parser.tokenize import split_sentences
    assert split_sentences("Lily went to the park\nShe was happy.") == ["Lily went to the park", "She was happy."]


def test_public_functions_not_shadowed_by_submodules():
    """ugl.humanize etc. stay the documented functions after the package has
    loaded its submodules (a submodule of the same name used to replace one)."""
    ugl.default()
    for name in ("encode", "decode", "humanize", "to_ids", "from_ids", "load"):
        f = getattr(ugl, name)
        assert callable(f) and not isinstance(f, type(ugl)), name


# families found by tools/fidelity_report.py (source English vs decoded English)
FIDELITY_FAMILIES = [
    # a frequent name numbered for frequency alone may be a person: "his" = Tim
    ("Tim wanted to play with his toys.", lambda u, e: "Tim's toys" in e),
    # a question inside quotes is still a question
    ('"Mom, can I play with my toys?"', lambda u, e: e.startswith("Can I play") and "?" in e),
    # set phrases: "once upon a time" = once; "one day" has no preposition
    ("Once upon a time, there was a family.", lambda u, e: e == "Once, there was a family."),
    ("One day, a dog barked.", lambda u, e: "one day" in e.lower() and "1 day" not in e),
    ("The next day, Tim went home.", lambda u, e: "the next day" in e and "at the next" not in e),
    # meaning adverbs are kept
    ("The astronomers can only see the separation.", lambda u, e: "only see" in e),
    ("Sam had dirty toys too.", lambda u, e: "also" in e and "excessively" not in e),
    # a kinship word used as a name is that person
    ("Mom said hello.", lambda u, e: e.startswith("Mom ") and "'Mom'" in u),
    # the word English uses most for the concept, among words that mean it first
    ("Her mom gave her a big hug.", lambda u, e: "mom" in e and "big" in e and "mummy" not in e),
    ("He started to cry.", lambda u, e: "cry" in e),
]


def test_fidelity_families():
    fails = []
    for en, ok in FIDELITY_FAMILIES:
        u = "\n".join(ENC.encode(en))
        e = DEC.decode(u)
        if not ok(u, e):
            fails.append(f"{en!r} -> {u!r} -> {e!r}")
    assert not fails, "\n".join(fails)

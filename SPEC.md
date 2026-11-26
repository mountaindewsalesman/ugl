# UGL specification v0.1

UGL ("universal grammatical language") is a written-out predicate–argument
tree. Every word is a numbered concept, and every grammatical fact is either
word order, a modifier letter, or an explicit open/close scope.

## 1. Tokens

| shape        | meaning                                            | example        |
|--------------|----------------------------------------------------|----------------|
| `n#` `v#` `a#` | noun / verb / adjective-or-adverb concept          | `n104`         |
| `nv#`        | verb used as a noun (the act of …)                 | `nv152` eating |
| `\v#`        | closes the scope opened by verb `#`                | `\v20`         |
| `#…`         | number literal                                     | `#9`, `#1990`  |
| `'…'`        | name literal (spaces → `_`)                        | `'New_York'`   |
| `n?w` `v?w` `a?w` | word missing from the lexicon (kept literally) | `n?blorft`     |
| `@k`         | back-reference: the k-th most recent referent (rule 15) | `@1`, `@2y` |

Modifier letters come right after the digits, in a fixed canonical order:
`n105st`.

## 2. Numbering

* **Lexical numbering is per meaning (WordNet sense), not per spelling.**
  `bank` (river) and `bank` (money) are different numbers, while `big` and
  `large` are the same number (one synset).
* The numbers follow **sense frequency** in the corpus (Wikipedia). Each
  lemma's count is split across its senses using SemCor sense counts.
* Nouns and verbs `1–99` are **reserved** (below), so lexical nouns and verbs
  start at 100. Adjectives start at 1.
* **Compounds before new words.** Senses are processed from most to least
  frequent. A sense gets **no number** when its dictionary gloss can be written
  with ≤3 more frequent words:
  * `puppy` "a young dog" → `n(dog) a(young)`
  * `destruction` "the act of destroying" → `nv(destroy)`
  * `stroll` "walk leisurely" → `v(walk) a(leisurely)`
  * adverbs → the adjective they derive from (`quickly` → `a(quick)`)
  * satellite adjectives without a usable gloss collapse into their head
    adjective when the two are close in embedding space (`huge` → `large`).
    This is the deliberate removal of connotation.
* Once built, `data/lexicon.json` is **frozen**. Rebuilding on a new corpus
  renumbers everything, so treat a lexicon as a versioned standard.

## 3. Reserved words

Nouns (they replace pronouns; no other pronouns exist):

| id | concept  | with modifiers |
|----|----------|----------------|
| n1 | speaker  | `n1s` = we |
| n2 | listener | you |
| n3 | thing    | `n3q` what, `n3x` nothing, `n3d` this, `n3e` that |
| n4 | person   | `n4q` who, `n4x` nobody |
| n5 | place    | `n5d` here, `n5e` there, `n5q` where |
| n6 | time     | `n6d` now, `n6e` then, `n6q` when |
| n7 | way      | `n7q` how |
| n8 | reason   | `n8q` why |
| n9 | amount   | `n9q` how much |

He, she, it and they are replaced by the noun they refer to.

Verbs:

| id | word | kind |
|----|------|------|
| v1 | be | core (identity / class / property) |
| v2 | have | core (possession) |
| v3 | that | relative-clause marker inside a noun phrase |
| v4 v5 v6 | and, or, but | logic (clause form only) |
| v7 | cause | relation (`A causes B`) |
| v8 | if | logic |
| v9 v10 v11 v12 | all, most, many, few | quantifiers |
| v20–v25 | time, before, after, during, until, since | relations |
| v30–v43 | in, on, at, under, above, near, between, around, through, across, along, behind, in-front-of, out-of | relations |
| v50–v61 | from, to, with, for, about, like, against, despite, except, as, of, by | relations |

## 4. Modifiers

| class | letter | meaning |
|-------|--------|---------|
| verb | `p` `f` | past, future (none = present / timeless) |
| verb | `g` `k` | ongoing, completed |
| verb | `c` `m` `u` | able (can), possible (may), required (must) |
| verb | `h` | hypothetical (would; `uh` = should, `ch` = could, `mh` = might) |
| verb | `q` `o` | yes/no question, command |
| any  | `x` | not / none / without (`v52x` = without) |
| noun | `s` | plural |
| noun | `t` `d` `e` | the, this/these, that/those (none = a / some) |
| noun | `q` | which / what |
| noun | `y` | possessor of the preceding head (`n150 'John'y` = John's dog) |
| noun | `j` | noun used as an attribute (`n200 n180j` = kitchen table) |
| noun | `b` | non- (something other than: `n_member_b` = non-member) |
| any  | `i` | anti- (against: `a_inflammatory_i`, `n_war_ij` = anti-war) |
| adj  | `r` `z` `l` `w` | more, most, less, very |

## 5. Grammar

```
clause  := np V adv* [np] (clause \V)*      core:  subject verb adverbs object complements
         | np R clause \R                   R relation/quantifier: "np is R of clause"
         | R clause \R clause \R            R relation/logic: R(A = landmark, B = event)
         | np                               bare noun phrase
np      := head mod*                        modifiers FOLLOW their head
         | L np \L np \L                     coordinated noun phrase, L = v4 (and) / v5 (or);
                                              as a possessor the y goes on L: v4y A \v4 B \v4
mod     := a# | n#j | #num | v3 clause \v3 | possessor-np(y)   (possessor is always last;
                                              after a coordinated np only these two: rule 13)
         | L mod+ \L mod+ \L                coordinated modifiers (rule 10), L = v4 / v5
```

Rules:

1. **Strict SVO**, at most one object. Recipients and every other argument use
   relation verbs (`gave the book to John`: `'John' v51 n1 v_give_p n_book_t \v51`).
2. **No passive.** `X was eaten by Y` → `Y ate X`. A passive without an agent
   uses `n4` (someone) as the subject.
3. **Adverbs go between the verb and the object.** Adjectives follow their
   noun. So `n1 v_eat a_quick n_apple` = "eat quickly the apple", and
   `n1 v_eat n_apple a_quick` = "eat the quick apple". This is never ambiguous.
4. **Prepositional phrases wrap the clause**: `#9 v20 n1 v_eat_p n_apple_t \v20`
   = I ate the apple at nine. Several around one clause all modify it, so their
   order carries no meaning and is canonical (rule 14), not the English order.
5. **Subordinate clauses** use the clause form, with the subordinate clause
   first: `v7 <it rained> \v7 <I stayed> \v7` = I stayed because it rained.
6. **Quantifiers scope explicitly** and repeat the noun:
   `n_dog v9 n_dog v_bark \v9` = every dog barks. Scope follows surface order
   (the first quantifier is outermost).
7. **Relative clauses** repeat the head noun instead of using who/which:
   `n_man_t v3 n_man_t v_leave_p \v3 v_see_p n1` = the man who left saw me.
8. **Clause complements** close with the verb that took them:
   `'Mary' v_say_p 'John' v_leave_p \v_say` = Mary said that John left.
   Infinitive complements restate the subject: `n1 v_want n1 v_eat \v_want`.
9. **Coordinated noun phrases are one group**: `v4 'John' \v4 'Mary' \v4 v_meet_p`
   = John and Mary met. "and" over two bare noun phrases always means a
   coordinated noun phrase (a group); followed by a verb, it is the subject.
   Re-mentioning the group ("they") repeats it. Nothing is distributed, so
   collective readings survive.
10. **Coordinated modifiers are one group too**, placed where a single modifier
   would go (after the head): `n_number_s v4 n_cell_j \v4 n_car_j \v4` = cell and
   car numbers; `n_car_ts v4 a_red \v4 a_blue \v4` = the red and blue cars;
   `n_day_s v5 #2 \v5 #3 \v5` = two or three days. Each member is a run of
   modifiers; three or more nest (`v4 A \v4 v4 B \v4 C \v4 \v4`). Right after a
   noun, and/or over modifiers only (no noun head, no verb) is always this group;
   anything else is a coordinated noun phrase or clause. As with rule 9, the
   group does not say whether each car is red and blue or some are red and some
   blue: English does not either.
11. **Closing tags pair like parentheses**: a closing tag closes the innermost
   open scope of that verb. `n4 v_say_p n4 v_say_p n3t v_rain_p \v_say \v_say`
   = someone said that someone said that it rained; the first `\v_say` closes
   the second "say".
12. **Prefixes are letters, not words.** A word the dictionary lacks, made of a
   grammatical prefix and a stem it has, is the stem plus a letter:
   non- on a noun is `b` (`n_member_bs` = non-members), anti- is `i` on any word
   (`a_inflammatory_i` = anti-inflammatory). non- on an adjective or verb is
   plain `x` (non-toxic = not toxic), so a meaning has one spelling. When the
   stem is a noun but the word works as an adjective, it is that noun used as
   an attribute: anti-war = `n_war_ij`. A word the dictionary has whole
   ("antibody", "nonsense", "nontoxic") is never split.
13. **Modifiers after a group belong to the group.** A relative clause, "of"
   phrase or apposition after the closing tag of a coordinated noun phrase
   modifies all of it: `v4 n_writer \v4 n_activist \v4 v3 v4 n_writer_t \v4
   n_activist_t \v4 v_serve_p \v3` = a writer and activist who served (both
   served). A modifier inside a member belongs to that member only. A later
   member without its own determiner shares the first one's: "the leader and
   founder" = the leader and the founder.
14. **One spelling per meaning: canonical order.** Where English order carries
   no meaning, UGL uses a fixed order: modifiers after their head go adjectives,
   then attribute nouns and names, then numbers ("the 1902 plague epidemic" =
   "the plague epidemic of 1902"); relations wrapped around one clause go by
   relation number, then by their landmark's tokens.
15. **Back-references.** A referent already mentioned in the same text may be
   written `@k`: the referent mentioned k-th most recently (k = 1 to 9), with a
   possessive `y` if needed (`@1y`). Referents are names, definite noun heads
   (`n_dog_t`, `n_cat_ts`, `n_box_d`) and two-member groups of those (rule 9).
   Every mention moves its referent to the front of the list; a group written in
   full counts its members first, then itself. `@k` means exactly the same as
   writing the referent out again, so the full form is always equivalent and
   either may be stored:
   `'Tom' v_land_p v30 n_back_t 'Tom'y \v30` = `'Tom' v_land_p v30 n_back_t @1y \v30`
   = Tom landed on his back. A text's first mention of anything is written in
   full. `ugl/refs.py` converts between the two forms exactly.

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
| adj  | `r` `z` `l` `w` | more, most, less, very |

## 5. Grammar

```
clause  := np V adv* [np] (clause \V)*      core:  subject verb adverbs object complements
         | np R clause \R                   R relation/quantifier: "np is R of clause"
         | R clause \R clause \R            R relation/logic: R(A = landmark, B = event)
         | np                               bare noun phrase
np      := head mod*                        modifiers FOLLOW their head
mod     := a# | n#j | #num | v3 clause \v3 | possessor-np(y)   (possessor is always last)
```

Rules:

1. **Strict SVO**, at most one object. Recipients and every other argument use
   relation verbs (`gave the book to John`: `'John' v51 n1 v_give_p n_book_t \v51`).
2. **No passive.** `X was eaten by Y` → `Y ate X`. A passive without an agent
   uses `n4` (someone) as the subject.
3. **Adverbs go between the verb and the object.** Adjectives follow their
   noun. So `n1 v_eat a_quick n_apple` = "eat quickly the apple", and
   `n1 v_eat n_apple a_quick` = "eat the quick apple". This is never ambiguous.
4. **Prepositional phrases wrap the clause**, outermost first in sentence
   order. This is your `nine has I ate \has`:
   `#9 v20 n1 v_eat_p n_apple_t \v20`.
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
9. **Coordinated noun phrases are distributed** over clauses (`v4`). The
   collective reading ("John and Mary met") is lost; see the limitations in
   the README.

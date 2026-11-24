# UGL: a universal grammatical language (v0.1, English only)

English â‡„ a numbered-concept notation whose grammar is fully explicit. See
[SPEC.md](SPEC.md) for the language itself.

```
EN  : At nine, I ate the apple.
UGL : #9 v20 n1 v444p n2344t \v20
BACK: I ate the apple at 9.

EN  : The man who left saw me.
UGL : n136t v3 n136t v153p \v3 v117p n1
BACK: The man that left saw me.

EN  : Every dog barks.
UGL : n988 v9 n988 v4520 \v9
```

Everything runs locally. There is no cloud service and no large language
model. The only neural parts are spaCy's small parser and word vectors
(`en_core_web_md`).

## Install

From the repo folder:

```
python -m venv .venv
.venv\Scripts\activate            # (Linux/macOS: source .venv/bin/activate)
pip install -e .                  # library + spaCy model (~45 MB download)
pip install -e .[dev]             # + tools to rebuild dictionaries / run tests
```

Or straight from GitHub, with no clone needed:
`pip install git+https://github.com/<you>/ugl.git`

## Use as a library

```python
import ugl

ugl.encode("At nine, I ate the apple.")      # ['#9 v20 n1 v444p n2344t \\v20']
ugl.decode("#9 v20 n1 v444p n2344t \\v20")    # 'I ate the apple at 9.'

ids = ugl.to_ids("Every dog barks.")          # ints for model training
ugl.from_ids(ids)                             # 'n988 v9 n988 v4520 \\v9'

u = ugl.load("ugl-dict-0.1.0.json.gz")        # pin a dictionary version
u.version, len(u.tokenizer)                   # ('0.1.0', 113468)
```

Tokenizer ids:
* One id per concept, in frequency order. Truncating the vocabulary keeps
  the most common meanings: `Tokenizer(lex, max_concepts=20000)`.
* Modifiers (`+p`, `+s`, …) are separate ids.
* Names, numbers and unknown words are spelled out character by character.
* Decoding the ids gives back exactly the same UGL text.

## Command line

```
ugl roundtrip "I stayed because it rained." --gloss
ugl encode "Mary said that John left."
ugl decode "'Mary' v105p 'John' v153p \v105"
ugl explain n988 v444
ugl sample --n 20 --gloss          # random sentences from data/raw
pytest tests
```

`--gloss` prints the tokens as English lemmas so you can read the structure.
Parenthetical `( … )` and `[ … ]` asides are dropped before encoding.

## Getting text

```
python scripts\fetch_wikipedia.py api  --count 500      # random articles via API
python scripts\fetch_wikipedia.py dump --count 50000    # streams the official dump
```

Dump mode streams and decompresses on the fly and stops after `--count`
articles, so nothing big is stored. 4,000 articles are about 89 MB of text.

## Dictionaries: frozen, versioned, one file

The dictionary is a single JSON file (gzipped), described in
[DICTIONARY_FORMAT.md](DICTIONARY_FORMAT.md).
* Released versions ship inside the package in `ugl/dictionary/`, and
  `default.txt` says which one loads by default.
* A released file is never changed, so the default always works and old
  encodings stay valid.

```
ugl build-lexicon                          # -> data/build/ugl-dict.json.gz (+ .tsv views)
ugl build-lexicon --recount --limit-chars 30000000
ugl promote --version 0.2.0                # freeze it into the package as the new default
ugl export --out dictionary-export         # plain indented JSON + TSV of the default
```

Dictionary 0.1.0 was built from 30M characters of Wikipedia (2.07M words):
* 80,998 noun, 13,737 verb and 18,296 adjective/adverb primitives.
* 4,628 senses expressed through other words: gloss compounds (`puppy` →
  young dog), connotation collapses (`mirthless` → `joyless`) and adverbs
  folded into their adjectives.
## How it works

| stage | file | method |
|---|---|---|
| corpus | `scripts/fetch_wikipedia.py` | MediaWiki API or dump streaming |
| numbering | `ugl/build_lexicon.py` | frequency per WordNet sense, gloss decomposition |
| English â†’ UGL | `ugl/encoder.py` | spaCy dependency parse + ~40 rewrite rules |
| UGL â†’ English | `ugl/decoder.py` | recursive-descent parser + realiser (`lemminflect`); no neural net |

## Measured on 121 random Wikipedia sentences

* **0.94** UGL tokens per English word. Some content is currently dropped (see
  below), so the true figure is a little higher.
* 3% of encodings failed to parse back, and 2% of tokens were words missing
  from the lexicon.

Token counts only save compute if a model uses UGL tokens as its vocabulary,
one token per concept. A standard tokenizer splits `n26988` into several
pieces.

## Known limitations (in order of how much they hurt)

1. **Word-sense choice is weak.** Senses are picked by most-frequent-sense
   plus gloss overlap, so `script` â†’ *book* and `particle` â†’ *atom*. This is
   the biggest source of meaning errors. The fix is a neural sense
   disambiguator (e.g. a small BERT trained on SemCor), which can still run
   locally.
2. **The parser's mistakes become UGL's mistakes.** Lists, parentheses, dates,
   headings and long sentences parse badly; some modifiers and phrases are
   silently dropped.
3. **Pronoun resolution is a recency heuristic.** It can pick the wrong noun.
4. **Distributed coordination.** "John and Mary met" becomes "John met and
   Mary met".
5. **Compounds use the first sense of each gloss word.** `baya`
   ("common Indian weaverbird") came out as *weaver* (the person). Review
   `compounds.tsv`.
6. **The decoder's English is stilted** ("impact negatively a human health").
   This is intended: the decoder is literal.
7. **Countability isn't known**, so you get "a water".
8. **Quantifier scope is always surface order**, and "some/any" are treated
   as plain indefinites.
9. **Hyphenated words** ("by-product") often come out as unknown literals.

# UGL dictionary file format (format_version 1)

A dictionary is **one UTF-8 JSON object**, normally gzipped
(`ugl-dict-<version>.json.gz`). Run `ugl export` to get a plain, indented
`.json` plus `.tsv` views of it.

Released dictionaries live in `ugl/dictionary/` and are **frozen**: a given
version always produces the same numbers. `ugl/dictionary/default.txt` names
the one the library loads by default. To pin a version in training code, use
`ugl.load("ugl-dict-0.1.0.json.gz")`.

```jsonc
{
  "format": "ugl-dictionary",
  "format_version": 1,
  "version": "0.1.0",                       // dictionary release (semver)
  "built": "2026-11-23 22:10",
  "source": {"corpus": "...", "corpus_tokens": 2074544, "sense_inventory": "Princeton WordNet 3.0"},
  "stats": {"primitives": {"n": 80998, "v": 13737, "a": 18296}, "senses": 117659, "compounds": 4628},
  "lexical_start": {"n": 100, "v": 100, "a": 1},

  "reserved":  {"nouns": {"n1": "speaker", ...},
                "verbs": {"v1": {"word": "be", "kind": "core"}, "v20": {"word": "time", "kind": "rel"}, ...}},
  "modifiers": {"verb": {"p": "past", ...}, "noun": {...}, "adjective": {...}},

  // every concept that has its own number, in frequency order
  "primitives": {
    "n100": {"lemma": "year", "sense": "year.n.01", "freq": 9123.4, "gloss": "..."}
  },

  // every word sense -> the UGL tokens that express it
  "senses": {
    "year.n.01":  {"tokens": ["n100"], "gloss": "..."},
    "puppy.n.01": {"tokens": ["n988", "a63"], "gloss": "a young dog", "why": "gloss: a young dog"}
  },

  // English lemma -> candidate senses, most frequent first.  "a" includes
  // satellite adjectives; "r" is adverbs.
  "index": {"n": {"dog": ["dog.n.01", "frump.n.01", ...]}, "v": {...}, "a": {...}, "r": {...}},

  // noun senses that denote people (used to resolve he/she)
  "person_senses": ["person.n.01", ...]
}
```

Compatibility rules:

* Within a `format_version`, fields can be added but never removed or renamed.
* Token numbers are only meaningful together with a dictionary `version`.
  Store the version next to any data you encode, for example in a training
  set's metadata.
* `ugl.Tokenizer` derives its integer ids from the dictionary alone, so the
  same dictionary version always gives the same ids.

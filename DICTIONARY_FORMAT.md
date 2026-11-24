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
* `ugl.Tokenizer` derives its integer ids from the dictionary plus its own
  layout version, `ugl.tokenizer.SCHEME`. The same (dictionary version,
  scheme) pair always gives the same ids, so record both next to tokenized
  data. Scheme 2 is current.

Fields added in later dictionaries (all optional when reading):

* `classes` (0.2.0+): `{"person"|"time"|"place": [senses]}`, the semantic
  classes the encoder uses.
* `names` (0.3.0+): `{"Barack Obama": "name:Barack Obama", "Paris": "paris.n.01"}`,
  mapping frequent proper names to senses. Name primitives carry `"proper": true`.
* `literal_pieces` (0.4.0+): subword pieces (BPE) for spelling literals.
* `symbols` (0.6.0+): `{"°C": ["degree_centigrade.n.01"], "°": ["degree.n.04", "degree.n.06"]}`,
  unit symbols mapped to the senses they can mean (one = exact; several = the
  sense model chooses among them).
* Headwords (0.6.0+): a primitive's `lemma` is chosen so that encoding it with
  no context gives the same tokens back (decode -> encode is stable), prefers
  real words over abbreviations ("metre", not "m"), then the spelling most used
  in that sense (0.5.0+).

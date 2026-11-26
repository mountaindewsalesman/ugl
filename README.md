# UGL: a universal grammatical language

UGL turns English into a notation of numbered concepts whose grammar is fully
explicit, and back again. Every word is a numbered meaning (`n909` = dog,
`v3861` = bark). Each relation opens a scope and closes it with a tag. There
are no pronouns: the thing referred to is repeated. The goal is precise,
low-ambiguity text for training language models. [SPEC.md](SPEC.md) defines
the language.

```
EN  : At nine, I ate the apple.
UGL : #9 v20 n1 v416p n4155t \v20
READ: nine time I ate the apple \time
BACK: At 9, I ate the apple.

EN  : The man who left saw me.
UGL : n145t v3 n145t v155p \v3 v114p n1
BACK: The man who left saw me.

EN  : Lily found a ball because she was looking under the bed.
UGL : v7 n1892t v33 'Lily' v315pg \v33 \v7 'Lily' v133p n580 \v7
BACK: Lily found a ball because Lily was looking under the bed.
```

`READ` is the same UGL with each number replaced by its English word, so you
can see the structure.

Everything runs locally. No cloud service and no outside language model are
involved. English is parsed by UGL's own transformer parser, trained on
Universal Dependencies treebanks. It runs on numpy alone, in Python or in
the browser.

* **Playground:** https://mountaindewsalesman.github.io/ugl/ encodes, decodes
  and explains UGL live in your browser, using this repo's Python code in
  Pyodide. It is rebuilt from `main` on every push and tested in Chromium and
  Firefox before it deploys.
* **Corpus page:** https://mountaindewsalesman.github.io/ugl/corpus.html shows
  live progress of the paired English/UGL training corpus: words, rates and
  recent sentence pairs.
* **Architecture, version by version:** [docs/architecture.html](docs/architecture.html).

## Install

```
pip install git+https://github.com/mountaindewsalesman/ugl.git          # numpy + lemminflect only
pip install "ugl[lm] @ git+https://github.com/mountaindewsalesman/ugl.git"   # + English story models
```

From a clone (Windows shown; on Linux/macOS use `source .venv/bin/activate`):

```
python -m venv .venv
.venv\Scripts\activate
pip install -e .          # the library
pip install -e .[lm]      # + tokenizers, for English story models
pip install -e .[dev]     # + everything to rebuild dictionaries, train models, run tests
```

## Use as a library

```python
import ugl

ugl.encode("At nine, I ate the apple.")      # ['#9 v20 n1 v416p n4155t \\v20']
ugl.decode("#9 v20 n1 v416p n4155t \\v20")    # 'At 9, I ate the apple.'
ugl.humanize("n909 v9 n909 v3861 \\v9")       # 'dog all dog barks \\all'

ids = ugl.to_ids("Every dog barks.")          # integer ids for model training
ugl.from_ids(ids)                             # 'n909 v9 n909 v3861 \\v9'

u = ugl.load("ugl-dict-0.6.0.json.gz")        # pin a dictionary version
u.version, len(u.tokenizer)                   # ('0.6.0', 121886)

# each sentence with the English it came from and anything the encoder is unsure of
for line in u.encode_lines(text, roundtrip=True):
    line.ugl, line.source, line.issues
```

Tokenizer ids:
* One id per concept, in frequency order. Modifier letters (`+p`, `+t`, …)
  and closing tags are ids of their own.
* Names, numbers and unknown words are spelled with subword pieces.
* Decoding ids always gives back exactly the same UGL text.
* Ids are only valid with their dictionary version and tokenizer scheme
  (`ugl/tokenizer.py`).

## Story models

`ugl.lm` runs small story-writing models trained on the paired corpus. Give
it some text and it writes the next sentences. There are two kinds of model:

* **English** models read and write English.
* **UGL** models read and write UGL. Your English prompt is encoded to UGL,
  the model continues in UGL, and the decoder turns the result back into
  English.

```python
import ugl.lm
m = ugl.lm.load("ugl-stories")           # a released model: downloaded once to ~/.cache/ugl
m = ugl.lm.load("path/to/model-folder")  # or a local folder
m.continue_text("Once upon a time, there was a little dog named Max.", sentences=3)
m.continue_text("...", details=True)     # [{"english": ..., "ugl": ...}, ...] per sentence
```

```
ugl continue "Once upon a time, there was a little dog named Max." --model ugl-stories --show-ugl
```

**No models are released yet.** The first pair, English and UGL trained on
the same ~50M words, comes after the current corpus build.

A model is a standard Hugging Face GPT-2 folder (`config.json`,
`model.safetensors`, `tokenizer.json`), so the same files also work in other
tools:

* `ugl.lm` runs them with numpy only, on any machine.
* `transformers` loads them: `GPT2LMHeadModel.from_pretrained(folder)`.
* `llama.cpp`'s `convert_hf_to_gguf.py` converts them to GGUF for llama.cpp,
  Ollama or LM Studio.
* `optimum` exports them to ONNX for the browser (WebGPU).

Models are released as GitHub releases named `lm-<name>` (see below).

## Command line

```
ugl encode "Mary said that John left." --gloss
ugl decode "n1862 v104p n720 v155p \v104"      # Mary said that John left.
ugl roundtrip "I stayed because it rained."
ugl explain n909 v3861
ugl sample --n 20 --gloss          # random sentences from data/raw
ugl continue "Once upon a time" --model ugl-stories
```

Parenthetical `( … )` and `[ … ]` asides are dropped before encoding.

## How it works

| stage | file | method |
|---|---|---|
| sentences + words | `ugl/parser/tokenize.py` | sentence splitter (newlines are boundaries), UD-style tokenizer |
| parse | `ugl/parser/`, `ugl/transformer.py` | ELECTRA-small transformer, biaffine heads, Eisner decoding (always a valid tree) |
| word meaning | `ugl/wsd_gloss.py` | sense head trained with the parser: the word in context against each sense's gloss (precomputed gloss table) |
| meaning graph | `ugl/analyze.py` | dependency tree → who-did-what-to-whom graph (`ugl/ir.py`) |
| pronouns | `ugl/coref.py` | pronoun → the thing referred to, across a whole document |
| simplify | `ugl/simplify.py` | long sentences split into simpler ones with the same meaning |
| English → UGL | `ugl/serialize.py` | the single place that writes UGL |
| UGL → English | `ugl/decoder.py` | recursive-descent parser + realiser (`lemminflect`); no neural net |

Measured accuracy:

| model | labelled attachment (LAS) | word sense |
|---|---|---|
| **ELECTRA-small (shipped, runs in the browser)** | **90.8** | 73.5% on held-out SemCor (most common sense: 66.8%) |
| ELECTRA-base (corpus building only, 206 MB) | 92.7 | higher than small |

Train the parsers with `tools/train_parser_tf.py`; score them with
`tools/eval_parser.py`.

## Dictionaries: frozen and versioned

Each dictionary is a single gzipped JSON file in `ugl/dictionary/`, described in
[DICTIONARY_FORMAT.md](DICTIONARY_FORMAT.md). `default.txt` names the default
(currently 0.6.0). A released file is never changed, so old encodings stay
valid.

```
ugl build-lexicon                     # -> data/build/ugl-dict.json.gz
ugl promote --version 0.7.0           # freeze it into the package as the new default
ugl export --out dictionary-export    # plain JSON + TSV views
```

After a dictionary or parser change, rebuild the gloss table with
`tools/build_gloss_table.py`.

## Building the training corpus

1. **Fetch** English text from open datasets into `data/raw/<source>/` with
   `scripts/fetch_sources.py`. Sources: TinyStories, Simple Wikipedia,
   FineWeb-Edu, Project Gutenberg, CC-News, OpenAssistant dialogue.
   `scripts/fetch_wikipedia.py` fetches Wikipedia.
2. **Encode and filter** with `tools/build_corpus.py`. It encodes whole
   documents, so pronouns resolve across paragraphs. It keeps only sentences
   that pass all of these checks:
   * The base and small parsers agree.
   * The parser is confident, at 0.6 or more, where it matters.
   * Decoding the UGL and re-encoding it gives the same UGL.

   The outputs are `passages.jsonl`, `ugl.txt`, `english.txt` and
   `rejected.tsv` (with reasons). Use `--device cuda` for batched GPU parsing
   (same output as on the CPU). Use `--resume` to continue exactly where the
   build stopped.
3. **Run unattended:** `tools/resume_builds.py <folder> --max 5 --watch 300`
   restarts any stopped build and keeps up to 5 running. Five is about what
   fits in 8 GB of GPU memory. `tools/corpus_monitor.py` publishes the stats
   behind the corpus page every 10 minutes.

## Training and releasing story models

```
python tools/train_lm.py data/build/corpus-v2 --lang ugl        # -> data/build/lm/ugl
python tools/train_lm.py data/build/corpus-v2 --lang english
python tools/export_lm.py data/build/lm/ugl dist/ugl-stories --check
gh release create lm-ugl-stories dist/ugl-stories/* --title "UGL story model"
```

Both languages train on the same passages, with the same split, model size
and settings. The headline number is **bits per English word** on the test
passages, which is comparable even though the two languages tokenize
differently. `--check` confirms that the exported model predicts the same as
the trained one in `ugl.lm` and in `transformers`.

## Tests

```
pytest tests
```

`tests/test_ugl.py` covers the bug families found so far (one regression test
per family). `tests/test_lm.py` covers the story-model runner and export.

## Known limitations

1. **Parser mistakes become UGL mistakes.** Lists, headings, dates and long
   sentences are the weak spots. The corpus filter keeps 58% of TinyStories
   sentences but only about 15% of Wikipedia sentences.
2. **Word sense is wrong about a quarter of the time** (for example *crying*
   → *calling*).
3. **Pronoun resolution is heuristic.** It can pick the wrong person.
4. **The decoder's English is literal and stilted** on purpose ("asked that
   …" for quoted questions; "named X").
5. **Countability isn't known** (you can get "a water").
6. **Quantifier scope always follows surface order.**

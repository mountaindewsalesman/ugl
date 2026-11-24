"""Build a UGL dictionary file (data/build/ugl-dict.json.gz).

Promote a build to the frozen, packaged default with:
    python -m ugl promote --version 0.2.0


Steps
 1. Count (lemma, class) frequencies in the raw corpus (data/raw/*.txt).
 2. Spread each lemma's count over its WordNet senses (SemCor sense counts +
    rank prior) -> a frequency per *sense*.
 3. Walk all senses from most to least frequent and decide, for each, whether
    it is a COMPOUND of already-decided, more frequent words (from its
    dictionary gloss) or a new PRIMITIVE.  Compounds get no index.
       noun  gloss "a young dog"          puppy   -> n(dog) a(young)
       noun  gloss "the act of destroying" destruction -> nv(destroy)
       verb  gloss "walk leisurely"        stroll  -> v(walk) a(leisurely)
       adj   gloss "very large"            -> a(large)w
       satellite adjective with no usable gloss -> its head adjective, if the
            two words are close in embedding space (huge -> large); this is
            the deliberate "remove connotation" collapse
       adverb -> the adjective it is derived from (quickly -> quick)
 4. Number the primitives per class in frequency order.
"""
import argparse
import collections
import glob
import json
import math
import os
import re
import sys
import time

from nltk.corpus import wordnet as wn

from .core import LEXICAL_START
from .core import RESERVED_NOUNS, RESERVED_VERBS, VERB_MODS, NOUN_MODS, ADJ_MODS


def synset_class(ss):
    return {"n": "n", "v": "v", "a": "a", "s": "a", "r": "a"}[ss.pos()]

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA = os.path.join(ROOT, "data")
BUILD = os.path.join(DATA, "build")
UPOS_CLASS = {"NOUN": "n", "VERB": "v", "ADJ": "a", "ADV": "r"}
WN_FOR_CLASS = {"n": ("n",), "v": ("v",), "a": ("a", "s"), "r": ("r",)}
INTENSIFIERS = {"very", "extremely", "highly", "exceedingly", "greatly", "unusually",
                "especially", "remarkably", "intensely", "most"}
MAX_MODS = 3


def log(*a):
    print(*a, file=sys.stderr, flush=True)


# ------------------------------------------------------------------ step 1
def count_corpus(nlp, files, limit_chars=None):
    counts = collections.Counter()
    total_chars = 0

    def texts():
        nonlocal total_chars
        for fp in files:
            with open(fp, encoding="utf-8", errors="ignore") as f:
                for para in f.read().split("\n"):
                    para = para.strip()
                    if len(para) < 20:
                        continue
                    total_chars += len(para)
                    if limit_chars and total_chars > limit_chars:
                        return
                    yield para[:100000]

    t0 = time.time()
    with nlp.select_pipes(enable=["tok2vec", "tagger", "attribute_ruler", "lemmatizer"]):
        for i, doc in enumerate(nlp.pipe(texts(), batch_size=256)):
            run = []
            for t in list(doc) + [None]:
                if t is not None and t.pos_ == "PROPN" and (t.is_alpha or "-" in t.text):
                    run.append(t.text)
                    continue
                if run:                                   # a maximal run of proper nouns = one name
                    counts[(" ".join(run), "name")] += 1
                    run = []
                if t is None:
                    break
                c = UPOS_CLASS.get(t.pos_)
                if c and t.is_alpha:
                    counts[(t.lemma_.lower(), c)] += 1
            if i % 2000 == 0 and i:
                log(f"  counted {i} paragraphs, {total_chars/1e6:.1f}M chars, {time.time()-t0:.0f}s")
    return counts


# ------------------------------------------------------------------ step 2
class NameSense:
    """A frequent proper name that WordNet does not have: a noun meaning of its own."""
    def __init__(self, text):
        self.text = text

    def name(self):
        return "name:" + self.text

    def pos(self):
        return "n"

    def definition(self):
        return "proper name"


def proper_sense(text):
    """The WordNet meaning of a proper name, if WordNet has it as a proper noun."""
    key = text.replace(" ", "_")
    for s in wn.synsets(key, pos="n"):
        if any(l.name() == key and l.name()[:1].isupper() for l in s.lemmas()):
            return s
    return None


def name_senses(counts, freq, min_count, max_names):
    """Frequent names -> senses.  Names WordNet knows (France, Aristotle) credit
    their count to that meaning; the rest become NameSense primitives."""
    names = sorted(((n, k[0]) for k, n in counts.items() if k[1] == "name" and n >= min_count), reverse=True)
    index, extra = {}, []
    for n, text in names[:max_names]:
        s = proper_sense(text)
        if s is not None:
            freq[s.name()] += n
            index[text] = s.name()
        else:
            ns = NameSense(text)
            freq[ns.name()] += n
            index[text] = ns.name()
            extra.append(ns)
    return index, extra


def sense_frequencies(counts):
    freq = collections.defaultdict(float)
    for (lemma, c), n in counts.items():
        if c == "name":
            continue
        senses = []
        for p in WN_FOR_CLASS[c]:
            senses += wn.synsets(lemma, pos=p)
        senses = list(dict.fromkeys(senses))
        # a lowercase corpus word ("north", "may") means a common sense, not a
        # proper one ("North" the Union, "May" the month): WordNet's lookup is
        # case-insensitive, so keep only senses that list the word in lowercase
        key = lemma.replace(" ", "_")
        common = [x for x in senses if any(l.name() == key for l in x.lemmas())]
        senses = common or senses
        if not senses:
            continue
        weights = []
        for rank, s in enumerate(senses):
            sc = sum(l.count() for l in s.lemmas() if l.name().lower() == lemma)
            weights.append((sc + 1.0) / (rank + 1.0))
        tot = sum(weights)
        for s, w in zip(senses, weights):
            freq[s.name()] += n * w / tot
    # every WordNet sense gets a tiny SemCor-based floor so unseen words still
    # get stable (late) positions
    for s in wn.all_synsets():
        sc = sum(l.count() for l in s.lemmas())
        freq[s.name()] += 1e-3 * (sc + 1) / (1 + int(s.name().rsplit(".", 1)[1]))
    return freq


# ------------------------------------------------------------------ step 3
class Builder:
    def __init__(self, nlp, freq, sim_threshold=0.30, collapse_threshold=0.55):
        self.nlp = nlp
        self.freq = freq
        self.sim_threshold = sim_threshold
        self.collapse_threshold = collapse_threshold
        # decided: synset name -> list of (synset_name_of_primitive, cls_override, mods)
        self.decided = {}
        self.how = {}

    # -- helpers
    def first_sense(self, lemma, c):
        for p in WN_FOR_CLASS[c]:
            ss = wn.synsets(lemma, pos=p)
            if ss:
                return ss[0]
        return None

    def vec_sim(self, a, b):
        va, vb = self.nlp.vocab[a], self.nlp.vocab[b]
        if not (va.has_vector and vb.has_vector):
            return None
        return float(va.similarity(vb))

    def expansion_of(self, ss, extra_mods=""):
        """Expansion of an already decided synset, or None."""
        exp = self.decided.get(ss.name())
        if exp is None:
            return None
        exp = [list(x) for x in exp]
        exp[0][2] = exp[0][2] + extra_mods
        return exp

    def adj_expansion(self, lemma, extra=""):
        """Adjective / adverb word -> expansion (adverbs map to their adjective)."""
        ss = self.first_sense(lemma, "a")
        if ss is None:
            r = self.first_sense(lemma, "r")
            if r is None:
                return None
            ss = r
        return self.expansion_of(ss, extra)

    # -- gloss patterns
    def decompose(self, ss, doc):
        """Try to express synset ss through its gloss.  Returns expansion or None."""
        c = synset_class(ss)
        own = ss.lemmas()[0].name().lower()
        toks = [t for t in doc if not t.is_punct]
        if not toks:
            return None
        root = [t for t in toks if t.dep_ == "ROOT"]
        if len(root) != 1:
            return None
        root = root[0]

        # "the act of destroying" -> nv(destroy)
        if c == "n":
            m = re.match(r"^(?:the |an? )?(?:act|process|action) of (\w+ing)\b(.*)$", doc.text.strip())
            if m and not m.group(2).strip():
                ing = [t for t in doc if t.text == m.group(1)]
                if ing:
                    v = self.first_sense(ing[0].lemma_, "v")
                    if v is not None:
                        e = self.expansion_of(v)
                        if e and e[0][1] == "v":
                            e[0][1] = "nv"
                            return e

        if c == "n" and root.pos_ == "NOUN":
            allowed = {"det", "amod", "compound", "conj", "cc", "advmod", "punct"}
        elif c == "v" and root.pos_ == "VERB":
            allowed = {"advmod", "conj", "cc", "punct"}
        elif c == "a" and root.pos_ == "ADJ":
            allowed = {"advmod"}
        else:
            return None
        # every token must hang (directly or via amod/conj) off the root with an allowed relation
        for t in toks:
            if t is root:
                continue
            if t.dep_ not in allowed:
                return None
            if t.dep_ == "advmod" and t.head is root and c == "n":
                return None

        head_lemma = root.lemma_.lower()
        if head_lemma == own:
            return None
        if c == "a":
            # gloss like "very large" / "not small"
            mods = ""
            for t in toks:
                if t is root:
                    continue
                w = t.lower_
                if w in INTENSIFIERS:
                    mods += "w"
                elif w in ("not", "un"):
                    mods += "x"
                else:
                    return None
            return self.adj_expansion(head_lemma, mods)

        # compound noun in the gloss ("ice cream"): try the joined lemma first
        head_ss = None
        comp = [t for t in toks if t.dep_ == "compound" and t.head is root]
        if c == "n" and comp:
            joined = "_".join([t.lemma_.lower() for t in comp] + [head_lemma])
            head_ss = self.first_sense(joined, "n")
            if head_ss is not None:
                comp = []
        if head_ss is None:
            head_ss = self.first_sense(head_lemma, c)
        if head_ss is None or head_ss.name() == ss.name():
            return None
        exp = self.expansion_of(head_ss)
        if exp is None:
            return None
        n_mods = 0
        for t in toks:
            if t is root or t.dep_ in ("det", "cc", "punct"):
                continue
            if t.dep_ == "compound":
                if t not in comp:
                    continue
                e = self.expansion_of(self.first_sense(t.lemma_.lower(), "n")) if self.first_sense(t.lemma_.lower(), "n") else None
                if e is None:
                    return None
                e[0][2] += "j"
                exp += e
            elif t.pos_ in ("ADJ", "ADV") or (t.pos_ == "VERB" and t.dep_ in ("amod", "conj", "advmod")):
                if t.lower_ in INTENSIFIERS and exp:
                    exp[-1][2] += "w"
                    continue
                e = self.adj_expansion(t.lemma_.lower())
                if e is None:
                    return None
                exp += e
            else:
                return None
            n_mods += 1
        if n_mods == 0 or n_mods > MAX_MODS:
            # n_mods == 0 means the gloss is a bare synonym ("a dog"): collapse
            if n_mods == 0:
                sim = self.vec_sim(own, head_lemma)
                if sim is None or sim < self.collapse_threshold:
                    return None
            else:
                return None
        sim = self.vec_sim(own, head_lemma)
        if sim is not None and sim < self.sim_threshold:
            return None
        return exp

    def run(self, order, gloss_docs):
        n_comp = n_coll = 0
        # phase A: nouns, verbs, head adjectives in frequency order
        # phase B: satellite adjectives;  phase C: adverbs
        phaseA = [s for s in order if s.pos() in ("n", "v", "a")]
        phaseB = [s for s in order if s.pos() == "s"]
        phaseC = [s for s in order if s.pos() == "r"]
        for phase, items in (("A", phaseA), ("B", phaseB), ("C", phaseC)):
            t0 = time.time()
            for i, ss in enumerate(items):
                name = ss.name()
                exp = None
                if phase in ("A", "B"):
                    exp = self.decompose(ss, gloss_docs[name])
                    if exp is not None:
                        self.how[name] = "gloss: " + ss.definition()
                        n_comp += 1
                if exp is None and phase == "B":
                    heads = ss.similar_tos()
                    if heads:
                        h = heads[0]
                        sim = self.vec_sim(ss.lemmas()[0].name().lower(), h.lemmas()[0].name().lower())
                        if sim is not None and sim >= self.collapse_threshold:
                            exp = self.expansion_of(h)
                            if exp is not None:
                                self.how[name] = f"collapsed into head adjective {h.name()} (sim {sim:.2f})"
                                n_coll += 1
                if exp is None and phase == "C":
                    for l in ss.lemmas():
                        for p in l.pertainyms():
                            exp = self.expansion_of(p.synset())
                            if exp is not None:
                                self.how[name] = f"adverb of {p.synset().name()}"
                                break
                        if exp is not None:
                            break
                if exp is None:
                    exp = [[name, "a" if ss.pos() in ("s", "r") else ss.pos(), ""]]
                self.decided[name] = exp
                if i % 20000 == 0 and i:
                    log(f"  phase {phase}: {i}/{len(items)} ({time.time()-t0:.0f}s)")
        log(f"  compounds from glosses: {n_comp}, connotation collapses: {n_coll}")


def build(raw_dir, out_path, limit_chars=None, recount=False, min_name=20, max_names=30000):
    import spacy
    nlp = spacy.load("en_core_web_md")
    os.makedirs(DATA, exist_ok=True)
    import gzip
    counts_path = os.path.join(DATA, "counts.json.gz")

    if os.path.exists(counts_path) and not recount:
        log("loading cached counts", counts_path)
        with gzip.open(counts_path, "rt", encoding="utf-8") as f:
            counts = collections.Counter({tuple(k.split("\t")): v for k, v in json.load(f).items()})
    else:
        files = sorted(glob.glob(os.path.join(raw_dir, "**", "*.txt"), recursive=True))
        if not files:
            log(f"no corpus files in {raw_dir}; run scripts/fetch_wikipedia.py first")
            sys.exit(1)
        log(f"step 1: counting {len(files)} files")
        counts = count_corpus(nlp, files, limit_chars)
        with gzip.open(counts_path, "wt", encoding="utf-8") as f:
            json.dump({f"{l}\t{c}": n for (l, c), n in counts.items()}, f)
    log(f"  {len(counts)} distinct (lemma, class) pairs, {sum(counts.values())} tokens")

    log("step 2: sense frequencies")
    freq = sense_frequencies(counts)
    name_index, name_extra = name_senses(counts, freq, min_name, max_names)
    log(f"  {len(name_index)} frequent names: {len(name_index) - len(name_extra)} already in WordNet, "
        f"{len(name_extra)} new")
    order = sorted(wn.all_synsets(), key=lambda s: (-freq[s.name()], s.name()))
    order = list(dict.fromkeys(order))

    log(f"step 3: parsing {len(order)} glosses")
    t0 = time.time()
    glosses = [s.definition().split(";")[0] for s in order]
    gloss_docs = {}
    with nlp.select_pipes(disable=["ner"]):
        for s, doc in zip(order, nlp.pipe(glosses, batch_size=512)):
            gloss_docs[s.name()] = doc
    log(f"  parsed in {time.time()-t0:.0f}s")
    b = Builder(nlp, freq)
    b.run(order, gloss_docs)
    for ns in name_extra:                       # names are always primitives (no gloss to decompose)
        b.decided[ns.name()] = [[ns.name(), "n", ""]]
    order = sorted(order + name_extra, key=lambda s: (-freq[s.name()], s.name()))

    log("step 4: numbering primitives")
    next_id = dict(LEXICAL_START)
    prim_id, ids = {}, {}
    for s in order:
        exp = b.decided[s.name()]
        if len(exp) == 1 and exp[0][0] == s.name():
            c = "a" if s.pos() in ("a", "s", "r") else s.pos()
            tok = f"{c}{next_id[c]}"
            next_id[c] += 1
            prim_id[s.name()] = tok
            if isinstance(s, NameSense):
                ids[tok] = {"sense": s.name(), "lemma": s.text, "freq": round(freq[s.name()], 4),
                            "gloss": s.definition(), "proper": True}
                continue
            ccls = {"n": "n", "v": "v", "a": "a", "s": "a", "r": "r"}[s.pos()]
            best = max(s.lemmas(), key=lambda l: (counts.get((l.name().lower().replace("_", " "), ccls), 0)
                                                  + counts.get((l.name().lower(), ccls), 0)
                                                  + counts.get((l.name().replace("_", " "), "name"), 0),
                                                  l.count(), -s.lemmas().index(l)))
            ids[tok] = {"sense": s.name(), "lemma": best.name().replace("_", " "),
                        "freq": round(freq[s.name()], 4), "gloss": s.definition()}
            # a proper noun: WordNet lists it as an instance (Paris is an instance of
            # a capital city), or every spelling is capitalized (January, Jan)
            if s.instance_hypernyms() or all(re.sub(r"^the_", "", l.name())[:1].isupper() for l in s.lemmas()):
                ids[tok]["proper"] = True

    def resolve(exp):
        out = []
        for name, cls, mods in exp:
            base = prim_id[name]
            if cls == "nv":
                base = "nv" + base[1:]
            out.append(base + mods)
        return out

    mapping = {s.name(): resolve(b.decided[s.name()]) for s in order}

    # lemma -> senses index (WordNet order = most frequent sense first), so the
    # runtime needs no WordNet install
    index = {"n": {}, "v": {}, "a": {}, "r": {}}
    for p, key in (("n", "n"), ("v", "v"), ("a", "a"), ("r", "r")):
        for lemma in wn.all_lemma_names(pos=p):
            ss = [x.name() for x in wn.synsets(lemma, pos=p)]
            if ss:
                index[key][lemma.replace("_", " ").lower()] = ss
    def closure(*roots):
        out = set()
        for r in roots:
            ss = wn.synset(r)
            out |= {x.name() for x in ss.closure(lambda s: s.hyponyms() + s.instance_hyponyms())} | {ss.name()}
        return sorted(out)
    # semantic classes the encoder needs (e.g. "at nine" = time, "at home" = place)
    classes = {
        "person": closure("person.n.01"),
        "time": closure("time_period.n.01", "time_unit.n.01", "calendar_month.n.01",
                        "day_of_the_week.n.01", "time.n.05", "season.n.02", "point.n.06",
                        "clock_time.n.01", "calendar_day.n.01"),
        "place": closure("location.n.01", "structure.n.01", "geographic_point.n.01"),
    }
    person_senses = classes["person"]

    senses = {}
    for s in order:
        e = {"tokens": mapping[s.name()], "gloss": s.definition()}
        if isinstance(s, NameSense):
            e["name"] = s.text
        if s.name() in b.how:
            e["why"] = b.how[s.name()]
        senses[s.name()] = e

    log("step 5: literal pieces")
    from .pieces import learn
    literal = collections.Counter()
    for (text, c), n in counts.items():
        if c == "name" and text not in name_index:
            literal[re.sub(r"\s+", "_", text)] += n            # rare names stay literals
        elif c in ("n", "v", "a", "r") and not any(wn.synsets(text.replace(" ", "_"), pos=p) for p in WN_FOR_CLASS[c]):
            literal[text.replace(" ", "_")] += n               # words WordNet lacks
    pieces = learn(literal, n_pieces=4000)
    log(f"  {len(pieces)} pieces from {len(literal)} literal strings")

    dictionary = {
        "format": "ugl-dictionary",
        "format_version": 1,
        "version": "unreleased",
        "built": time.strftime("%Y-%m-%d %H:%M"),
        "source": {"corpus": "English Wikipedia sample", "corpus_tokens": sum(counts.values()),
                   "sense_inventory": "Princeton WordNet 3.0"},
        "stats": {"primitives": {k: v - LEXICAL_START[k] for k, v in next_id.items()},
                  "senses": len(senses), "compounds": len(b.how)},
        "lexical_start": LEXICAL_START,
        "reserved": {"nouns": {f"n{i}": w for i, w in RESERVED_NOUNS.items()},
                     "verbs": {f"v{i}": {"word": w, "kind": k} for i, (w, k) in RESERVED_VERBS.items()}},
        "modifiers": {"verb": VERB_MODS, "noun": NOUN_MODS, "adjective": ADJ_MODS},
        "primitives": ids,
        "senses": senses,
        "index": index,
        "person_senses": person_senses,
        "classes": classes,
        "names": name_index,          # exact proper name -> sense (format addition, v1-compatible)
        "literal_pieces": pieces,     # subword pieces for spelling literals (tokenizer)
    }
    write_dictionary(dictionary, out_path)
    export_tsv(dictionary, os.path.dirname(out_path))
    log(f"done: {dictionary['stats']} -> {out_path}")


def write_dictionary(d, path):
    import gzip
    os.makedirs(os.path.dirname(path), exist_ok=True)
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "wt", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, separators=(",", ":"))


def export_tsv(d, folder):
    """Human-readable views of a dictionary."""
    ids = d["primitives"]
    with open(os.path.join(folder, "primitives.tsv"), "w", encoding="utf-8") as f:
        f.write("token\tlemma\tsense\tfreq\tgloss\n")
        for t, e in ids.items():
            f.write(f"{t}\t{e['lemma']}\t{e['sense']}\t{e['freq']}\t{e['gloss']}\n")
    with open(os.path.join(folder, "compounds.tsv"), "w", encoding="utf-8") as f:
        f.write("sense\texpansion\treadable\twhy\n")
        for name, e in d["senses"].items():
            if "why" not in e:
                continue
            readable = " ".join(ids[re.sub(r"[a-z]+$", "", "v" + t[2:] if t.startswith("nv") else t)]["lemma"]
                                for t in e["tokens"])
            f.write(f"{name}\t{' '.join(e['tokens'])}\t{readable}\t{e['why']}\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=os.path.join(DATA, "raw"))
    ap.add_argument("--out", default=os.path.join(BUILD, "ugl-dict.json.gz"))
    ap.add_argument("--limit-chars", type=int, default=None, help="stop counting after N characters")
    ap.add_argument("--recount", action="store_true", help="ignore cached data/counts.json.gz")
    ap.add_argument("--min-name", type=int, default=20, help="corpus count for a name to get its own number")
    ap.add_argument("--max-names", type=int, default=30000)
    a = ap.parse_args(argv)
    build(a.raw, a.out, a.limit_chars, a.recount, a.min_name, a.max_names)


if __name__ == "__main__":
    main()

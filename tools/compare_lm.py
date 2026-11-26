"""Where does the UGL model lose to the English one?  Sentence by sentence.

    python tools/compare_lm.py data/build/corpus-v2            # models from data/build/lm/{ugl,english}

Both models (tools/train_lm.py) score every test passage, token by token, with
the end-of-passage token before it as context (as in training, where passages
are packed after one another), so every token of every sentence is scored.
Each sentence's bits are its tokens' bits, its sentence break included.

Reported:
  * UGL's bits per English word split by token type: which part of UGL costs
    what (concept senses, relation words, closing tags, modifier letters,
    literal names/numbers, sentence breaks)
  * the per-sentence gap (UGL bits - English bits, per English word) by
    parser confidence (the corpus parser's least sure head in the sentence),
    source, length, pronouns (which UGL replaces with repeated names),
    literals and dialogue
  * the sentences with the biggest gaps either way

Writes data/build/lm/compare.json and compare-sentences.tsv.
"""
import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from train_lm import BRK, EOS, PAD, GPT, EnglishTokens, UGLTokens, english_words, passages, split_of  # noqa: E402
from ugl.core import RESERVED_NOUNS, RESERVED_VERBS  # noqa: E402

LOG2 = math.log(2)
PRONOUNS = re.compile(r"\b(he|she|it|they|him|her|them|his|its|their|hers|theirs|himself|herself|itself|themselves)\b", re.I)


def rss_gb():
    """This process's resident memory in GB (Windows API, else /proc), for the progress log."""
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + \
                           [(n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize",
                            "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                            "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
            c = PMC()
            c.cb = ctypes.sizeof(c)
            k32 = ctypes.WinDLL("kernel32")
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            get = ctypes.WinDLL("psapi").GetProcessMemoryInfo
            get.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]   # a 64-bit handle, not an int
            get(k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
            return c.WorkingSetSize / 2 ** 30
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2 ** 30
    except Exception:  # noqa: BLE001
        return float("nan")


def log(msg):
    print(f"{msg}  [{rss_gb():.1f} GB]", file=sys.stderr, flush=True)


def load_model(folder, device):
    ck = torch.load(os.path.join(folder, "model.pt"), map_location=device)
    c = ck["cfg"]
    m = GPT(c["vocab"], c["d"], c["layers"], c["heads"], c["ctx"]).to(device)
    m.load_state_dict(ck["model"])
    return m.eval()


@torch.no_grad()
def token_bits(model, seqs, device, bs=16):
    """Bits of every token of every sequence, each predicted from [EOS] + what precedes it."""
    out = [None] * len(seqs)
    order = sorted(range(len(seqs)), key=lambda k: len(seqs[k]))
    for i in range(0, len(order), bs):
        if i % (bs * 1000) == 0:
            log(f"  scoring {i:,}/{len(order):,}")
        idx = order[i:i + bs]
        part = [[EOS] + seqs[k][:model.ctx] for k in idx]
        T = max(len(s) for s in part)
        x = torch.full((len(part), T), PAD, dtype=torch.long)
        for b, s in enumerate(part):
            x[b, :len(s)] = torch.tensor(s)
        x = x.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logits = model(x[:, :-1])
        nll = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1),
                              ignore_index=PAD, reduction="none").view(len(part), -1) / LOG2
        for b, k in enumerate(idx):
            out[k] = nll[b, :len(part[b]) - 1].cpu().numpy()
    return out


def per_sentence(ids, bits):
    """Sum bits per sentence: a sentence's tokens and the break (or end) after it."""
    sums, cur = [], 0.0
    for t, b in zip(ids, bits):
        cur += float(b)
        if t in (BRK, EOS):
            sums.append(cur)
            cur = 0.0
    if cur:
        sums.append(cur)
    return sums


def ugl_kinds(tok):
    """model id -> token type, for the UGL vocabulary."""
    reserved = {f"n{i}" for i in RESERVED_NOUNS} | {f"v{i}" for i in RESERVED_VERBS}
    kinds = {BRK: "sentence break", EOS: "end of passage"}
    for orig, mid in tok.vocab.items():
        s = tok.tok.vocab[orig]
        if s in reserved:
            k = "relation/fixed word (v1-v61, n1-n9)"
        elif re.fullmatch(r"[nva]\d+", s):
            k = {"n": "noun concept", "v": "verb concept", "a": "adjective/adverb concept"}[s[0]]
        elif s.startswith("\\") or s == "<close>":
            k = "closing tag"
        elif s.startswith("+"):
            k = "modifier letter"
        else:
            k = "literal (names, numbers, unknown words)"
        kinds[mid] = k
    return kinds


def min_confidence(sentences, device):
    """The corpus parser's least sure head (non-punctuation) in each sentence."""
    from ugl.parser.model import ParserModel
    from ugl.parser.tokenize import tokenize
    from ugl.transformer import use_device
    from build_corpus import BASE
    if device == "cuda":
        use_device("cuda")
    m = ParserModel(BASE if os.path.exists(BASE) else None)
    words = [tokenize(s) for s in sentences]
    out = []
    for i in range(0, len(words), 2000):
        chunk = words[i:i + 2000]
        m.prefetch(chunk)
        for w in chunk:
            st = m.score(w)
            tree = m.tree(st, m.decode_tree(st["S"]))
            out.append(min([t.get("conf", 1.0) for t in tree if t["upos"] != "PUNCT"] or [1.0]))
        log(f"  parsed {len(out)}/{len(words)}")
    return out


def table(rows, key, label, buckets=None):
    """Mean gap (UGL - English bits per word, word-weighted) per value of key."""
    g = defaultdict(lambda: [0.0, 0.0, 0.0, 0])
    for r in rows:
        v = key(r)
        g[v][0] += r["ugl"]
        g[v][1] += r["english"]
        g[v][2] += r["words"]
        g[v][3] += 1
    print(f"\n{label:28} sentences   words    UGL  English   gap (bits/word)")
    out = []
    for v in (buckets or sorted(g)):
        if v not in g:
            continue
        u, e, w, n = g[v]
        print(f"  {str(v):26} {n:9,} {int(w):8,} {u / w:6.3f} {e / w:7.3f}  {(u - e) / w:+.3f}")
        out.append({"value": str(v), "sentences": n, "words": w, "ugl": u / w, "english": e / w, "gap": (u - e) / w})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--lm", default=os.path.join(ROOT, "data", "build", "lm"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-parse", action="store_true", help="skip the parser-confidence breakdown")
    a = ap.parse_args()
    test = [p for p in passages(a.corpus) if split_of(p) == "test"]
    log(f"{len(test):,} test passages")
    utok = UGLTokens()
    with open(os.path.join(a.lm, "ugl", "vocab.json")) as f:
        utok.vocab = {int(k): v for k, v in json.load(f).items()}
    etok = EnglishTokens(os.path.join(a.lm, "english", "bpe.json"))
    kinds = ugl_kinds(utok)

    rows, by_kind = [], defaultdict(float)
    sides = {}
    for lang, tok in (("ugl", utok), ("english", etok)):
        model = load_model(os.path.join(a.lm, lang), a.device)
        seqs = [tok.ids(p) for p in test]
        log(f"{lang}: tokenized")
        bits = token_bits(model, seqs, a.device)
        log(f"{lang}: scored")
        sides[lang] = (seqs, bits)
        del model
        torch.cuda.empty_cache() if a.device == "cuda" else None
    useqs, ubits = sides["ugl"]
    eseqs, ebits = sides["english"]
    skipped = 0
    for p, us, ub, es, eb in zip(test, useqs, ubits, eseqs, ebits):
        su, se = per_sentence(us, ub), per_sentence(es, eb)
        if len(us) > len(ub) or len(es) > len(eb) or len(su) != len(p["english"]) or len(se) != len(su):
            skipped += 1                      # longer than the window: not every sentence was scored
            continue
        for t, b in zip(us, ub):
            by_kind[kinds.get(t, "unknown id")] += float(b)
        for k, (s, u_line) in enumerate(zip(p["english"], p["ugl"])):
            rows.append({"english": s, "ugl_text": u_line, "source": p.get("source", "?"), "first": k == 0,
                         "words": max(1, len(s.split())), "ugl": su[k], "english_bits": se[k]})
    for r in rows:
        r["english_text"] = r.pop("english")
        r["english"] = r.pop("english_bits")
    words = sum(r["words"] for r in rows)
    U, E = sum(r["ugl"] for r in rows), sum(r["english"] for r in rows)
    print(f"\n{len(rows):,} test sentences ({skipped} passages longer than the window left out), {words:,} words")
    print(f"overall: UGL {U / words:.3f}, English {E / words:.3f} bits per English word; gap {(U - E) / words:+.3f}")
    print("\nUGL bits per English word, by token type:")
    kind_rows = []
    for k, b in sorted(by_kind.items(), key=lambda x: -x[1]):
        print(f"  {k:42} {b / words:6.3f}  ({b / U:5.1%})")
        kind_rows.append({"kind": k, "bits_per_word": b / words, "share": b / U})

    report = {"sentences": len(rows), "words": words, "ugl": U / words, "english": E / words, "by_kind": kind_rows}
    for r in rows:
        r["pronouns"] = len(PRONOUNS.findall(r["english_text"]))
        r["literal"] = "'" in r["ugl_text"] or "#" in r["ugl_text"] or "?" in r["ugl_text"]
        r["quote"] = '"' in r["english_text"]
    report["source"] = table(rows, lambda r: r["source"], "source")
    lb = ["1-5", "6-10", "11-15", "16-25", "26+"]
    report["length"] = table(rows, lambda r: lb[next(i for i, x in enumerate([5, 10, 15, 25, 10 ** 9])
                                                     if r["words"] <= x)], "length (words)", lb)
    report["pronouns"] = table(rows, lambda r: min(r["pronouns"], 3), "pronouns in English (3 = 3+)")
    report["literal"] = table(rows, lambda r: r["literal"], "has a literal (name/number)")
    report["quote"] = table(rows, lambda r: r["quote"], "dialogue (has quotes)")
    report["first"] = table(rows, lambda r: r["first"], "first sentence of passage")
    if not a.no_parse:
        conf = min_confidence([r["english_text"] for r in rows], a.device)
        cb = ["<0.7", "0.7-0.8", "0.8-0.9", "0.9-0.95", "0.95-0.99", ">=0.99"]
        edges = [0.7, 0.8, 0.9, 0.95, 0.99, 2.0]
        for r, c in zip(rows, conf):
            r["confidence"] = c
            r["conf_bucket"] = cb[next(i for i, e in enumerate(edges) if c < e)]
        report["confidence"] = table(rows, lambda r: r["conf_bucket"], "parser's least sure head", cb)
    for r in rows:
        r["gap"] = (r["ugl"] - r["english"]) / r["words"]
    rows.sort(key=lambda r: r["gap"])
    print("\nUGL much better than English:")
    for r in rows[:8]:
        print(f"  {r['gap']:+6.2f}  {r['english_text'][:90]}")
    print("UGL much worse than English:")
    for r in rows[-12:]:
        print(f"  {r['gap']:+6.2f}  {r['english_text'][:90]}")
    with open(os.path.join(a.lm, "compare.json"), "w") as f:
        json.dump(report, f, indent=1)
    with open(os.path.join(a.lm, "compare-sentences.tsv"), "w", encoding="utf-8") as f:
        f.write("gap\tugl_bits\tenglish_bits\twords\tconfidence\tsource\tenglish\tugl\n")
        for r in rows:
            f.write(f"{r['gap']:.3f}\t{r['ugl']:.2f}\t{r['english']:.2f}\t{r['words']}\t{r.get('confidence', '')}\t"
                    f"{r['source']}\t{r['english_text']}\t{r['ugl_text'].replace(chr(10), ' / ')}\n")
    print("\nwrote", os.path.join(a.lm, "compare.json"))


if __name__ == "__main__":
    main()

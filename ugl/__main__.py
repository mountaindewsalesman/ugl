"""Command line:
  python -m ugl encode "At nine, I ate the apple."
  python -m ugl decode "n6 ..."
  python -m ugl roundtrip "Some English text."      English -> UGL -> English
  python -m ugl sample --n 20                       roundtrip random corpus sentences
  python -m ugl build-lexicon [--limit-chars N]
  python -m ugl explain n1234                       what a token means
"""
import argparse
import glob
import os
import random
import sys


def load():
    from . import UGL
    u = UGL()
    return u.lexicon, u.encoder, u.decoder


def promote(version, src=None, make_default=True):
    import re
    from .build_lexicon import BUILD, write_dictionary
    from .lexicon import DICT_DIR, read_dictionary
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        sys.exit("version must look like 1.2.3")
    src = src or os.path.join(BUILD, "ugl-dict.json.gz")
    name = f"ugl-dict-{version}.json.gz"
    dst = os.path.join(DICT_DIR, name)
    if os.path.exists(dst):
        sys.exit(f"{name} already exists; released dictionaries are frozen - pick a new version")
    d = read_dictionary(src)
    d["version"] = version
    write_dictionary(d, dst)
    if make_default:
        with open(os.path.join(DICT_DIR, "default.txt"), "w", encoding="utf-8") as f:
            f.write(name + "\n")
    print(f"wrote {dst}" + (" (now the default)" if make_default else ""))


def export(dictionary, out):
    import json
    from .build_lexicon import export_tsv
    from .lexicon import Lexicon
    lex = Lexicon(dictionary)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f"ugl-dict-{lex.version}.json"), "w", encoding="utf-8") as f:
        json.dump(lex.data, f, ensure_ascii=False, indent=1)
    export_tsv(lex.data, out)
    print("exported to", out)


def show(enc, dec, text, gloss=False):
    for ugl in enc.encode(text, keep_memory=True):
        print("UGL :", ugl)
        if gloss:
            print("READ:", HUMANIZER(ugl))
        try:
            print("BACK:", dec.decode(ugl))
        except Exception as e:
            print("BACK: <decode error:", e, ">")


HUMANIZER = None


def gloss_line(lex, ugl):
    from .core import WORD_RE
    out = []
    for t in ugl.split():
        m = WORD_RE.match(t)
        if m:
            cls = "v" if m.group(1) == "nv" else m.group(1)
            out.append(f"{lex.lemma_of(cls, m.group(2)).replace(' ', '_')}{'.' + m.group(3) if m.group(3) else ''}")
        else:
            out.append(t)
    return " ".join(out)


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("encode", "decode", "roundtrip"):
        p = sub.add_parser(name)
        p.add_argument("text", nargs="+")
        p.add_argument("--gloss", action="store_true", help="also print the readable view (tokens as English words)")
    s = sub.add_parser("sample")
    s.add_argument("--n", type=int, default=15)
    s.add_argument("--seed", type=int, default=None)
    s.add_argument("--gloss", action="store_true")
    e = sub.add_parser("explain")
    e.add_argument("tokens", nargs="+")
    pr = sub.add_parser("promote", help="freeze data/build/ugl-dict.json.gz as a packaged, versioned default")
    pr.add_argument("--version", required=True)
    pr.add_argument("--src", default=None)
    pr.add_argument("--no-default", action="store_true", help="ship it but keep the current default")
    ex = sub.add_parser("export", help="write a dictionary as plain JSON + TSV views")
    ex.add_argument("--dictionary", default=None)
    ex.add_argument("--out", default="dictionary-export")
    b = sub.add_parser("build-lexicon")
    b.add_argument("rest", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)

    if a.cmd == "build-lexicon":
        from .build_lexicon import main as bmain
        return bmain(a.rest)

    if a.cmd == "promote":
        return promote(a.version, a.src, not a.no_default)
    if a.cmd == "export":
        return export(a.dictionary, a.out)

    global HUMANIZER
    lex, enc, dec = load()
    from .humanize import Humanizer
    HUMANIZER = Humanizer(lex, dec)
    if a.cmd == "encode":
        for line in enc.encode(" ".join(a.text)):
            print(line)
            if a.gloss:
                print("  ", HUMANIZER(line))
    elif a.cmd == "decode":
        print(dec.decode(" ".join(a.text)))
    elif a.cmd == "roundtrip":
        show(enc, dec, " ".join(a.text), a.gloss)
    elif a.cmd == "explain":
        for t in a.tokens:
            ent = lex.entry(t)
            print(t, "->", ent if ent else gloss_line(lex, t))
    elif a.cmd == "sample":
        root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
        files = glob.glob(os.path.join(root, "*.txt"))
        rng = random.Random(a.seed)
        sents = []
        while len(sents) < a.n:
            with open(rng.choice(files), encoding="utf-8") as f:
                paras = [p for p in f.read().split("\n") if len(p) > 60]
            from .parser.tokenize import split_sentences
            cand = [x for x in split_sentences(rng.choice(paras)[:2000]) if 6 <= len(x.split()) <= 30]
            if cand:
                sents.append(rng.choice(cand))
        for s_ in sents:
            print("\nEN  :", s_)
            enc.memory = []
            show(enc, dec, s_, a.gloss)
        if enc.unknown:
            print("\nwords not in lexicon:", sorted(enc.unknown)[:40])


if __name__ == "__main__":
    main()

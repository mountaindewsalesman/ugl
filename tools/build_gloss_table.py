"""Precompute the sense head's gloss vectors (see ugl/wsd_gloss.py).

    python tools/build_gloss_table.py                 # shipped parser, default dictionary
    python tools/build_gloss_table.py --parser data/build/parser-en-base.npz

Every sense that can compete with another (a lemma with two or more senses in
some class) gets its vector, quantized to 8 bits with one scale per row.
Senses missing from the table are still computed on demand, so the table
only changes speed, never which sense wins (beyond 8-bit rounding, checked
here on a sample).
Rebuild whenever the parser weights or the dictionary change: a table whose
recorded parser size or dictionary version differs is ignored.
"""
import argparse
import json
import os
import sys
import time

# one BLAS thread per process: gloss matrices are small (<= 64 x 256), so
# threads barely help while worker processes scale with the cores
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ugl  # noqa: E402
from ugl.parser.model import DEFAULT_PATH, ParserModel  # noqa: E402
from ugl.wsd_gloss import GlossWSD, table_meta, table_path  # noqa: E402

# choose() looks at the first 12 candidates, after names are filtered out of a
# lemma's list; 24 per lemma covers that with room to spare
PER_LEMMA = 24


def competing_senses(lex):
    need = []
    seen = set()
    for lemmas in lex.index.values():
        for ss in lemmas.values():
            ss = [s for s in ss if s in lex.senses_]
            if len(ss) >= 2:
                for s in ss[:PER_LEMMA]:
                    if s not in seen:
                        seen.add(s)
                        need.append(s)
    return need


_wsd = None


def _init(parser_path):
    global _wsd
    _wsd = GlossWSD(ugl.UGL(wsd=False).lexicon, ParserModel(parser_path))
    _wsd.table = None                                  # compute everything afresh


def _vectors(senses):
    return np.stack([_wsd.compute_gloss_vec(s) for s in senses])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parser", default=DEFAULT_PATH)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = ap.parse_args()
    u = ugl.UGL(wsd=False)
    lex = u.lexicon
    senses = competing_senses(lex)
    print(f"{len(senses)} competing senses (dictionary {lex.version}, parser {os.path.basename(a.parser)}, "
          f"{a.workers} workers)", flush=True)
    import multiprocessing
    chunks = [senses[i:i + 500] for i in range(0, len(senses), 500)]
    t, parts = time.time(), []
    with multiprocessing.Pool(a.workers, _init, (a.parser,)) as pool:
        for k, v in enumerate(pool.imap(_vectors, chunks)):
            parts.append(v)
            if k % 10 == 9:
                done = sum(len(p) for p in parts)
                print(f"  {done}/{len(senses)}  ({(time.time() - t) / done * 1000:.2f} ms each)", flush=True)
    vecs = np.concatenate(parts)
    scale = np.abs(vecs).max(axis=1) / 127.0
    scale[scale == 0] = 1.0
    q = np.round(vecs / scale[:, None]).astype(np.int8)
    err = np.abs(q * scale[:, None] - vecs).max() / np.abs(vecs).max()
    path = table_path(a.parser, lex.version)
    np.savez_compressed(path, senses=np.array("\n".join(senses)), q=q, scale=scale.astype(np.float32),
                        meta=np.array(json.dumps(table_meta(a.parser, lex.version))))
    print(f"wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB; worst 8-bit error {err:.2%} of the largest value)")


if __name__ == "__main__":
    main()

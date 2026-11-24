"""Build sense centroids for ugl.wsd_context from SemCor.

    python tools/train_wsd.py                 # all of SemCor except the eval slice
    python tools/train_wsd.py --max-sents 5000

The sentences that tools/eval_wsd.py tests on (offset 20000, 400 sentences
by default) are held out.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ugl.wsd_context import DEFAULT_MODEL, DEFAULT_PATH, make_embedder
from eval_wsd import items  # noqa: E402  (same folder)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default=DEFAULT_PATH)
    ap.add_argument("--max-sents", type=int, default=None)
    ap.add_argument("--holdout-offset", type=int, default=20000)
    ap.add_argument("--holdout-size", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--prior-weight", type=float, default=0.05,
                    help="stored in the file; pick it with tools/eval_wsd.py on the held-out slice")
    a = ap.parse_args()

    from nltk.corpus import semcor
    total = len(semcor.tagged_sents(tag="sem"))
    emb = make_embedder(a.model)          # "ugl-parser:A-B" = the parser's own transformer layers
    sums, counts = {}, {}
    ranges = [(0, a.holdout_offset), (a.holdout_offset + a.holdout_size, total)]
    batch, n, t0 = [], 0, time.time()

    def flush():
        vecs = emb.embed_batch([w for w, _ in batch])
        for (words, targets), v in zip(batch, vecs):
            for i, _lemma, sense, _cls in targets:
                if i < len(words):
                    if sense in sums:
                        sums[sense] += v[i]
                        counts[sense] += 1
                    else:
                        sums[sense] = v[i].copy()
                        counts[sense] = 1
        batch.clear()

    for start, end in ranges:
        for words, targets in items(end - start, start):
            if not targets:
                continue
            batch.append((words, targets))
            n += 1
            if len(batch) >= a.batch:
                flush()
            if n % 1000 == 0:
                print(f"  {n} sentences, {len(sums)} senses, {time.time()-t0:.0f}s", file=sys.stderr, flush=True)
            if a.max_sents and n >= a.max_sents:
                break
        if a.max_sents and n >= a.max_sents:
            break
    if batch:
        flush()

    names = sorted(sums)
    mat = np.stack([sums[s] / counts[s] for s in names]).astype(np.float16)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez_compressed(a.out, senses=np.array(names), vectors=mat, model=np.array(a.model),
                        prior_weight=np.array(a.prior_weight),
                        counts=np.array([counts[s] for s in names], dtype=np.int32))
    print(f"saved {len(names)} sense centroids from {n} sentences -> {a.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()

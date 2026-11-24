"""How trustworthy are the parser's head probabilities?  (sets ugl.encoder.UNSURE)

    python tools/calibrate_parser.py [--split dev]

1. Calibration: of the words whose chosen head got probability p, what share
   really have the right head?  (well calibrated: about p)
2. Thresholds, at the sentence level, which is what the corpus filter uses: a
   sentence is flagged when any word is below the threshold.  For each
   threshold: share of sentences kept, and share of kept sentences that still
   contain a wrong head (the errors that get into training data).
Punctuation is skipped, as in the encoder's flag.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ugl.parser.conllu import read_conllu  # noqa: E402
from ugl.parser.model import ParserModel  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def fit(m, sents):
    """Temperature scaling: the T minimising the negative log-likelihood of the
    gold heads under softmax(scores / T); written into the model's metadata."""
    import json
    import numpy as np
    data = []
    for s in sents:
        S = m.arc_scores(m.encode([t["form"] for t in s]))
        data.append((S, np.array([t["head"] for t in s])))
    best = None
    for T in np.arange(0.5, 6.01, 0.1):
        nll = 0.0
        for S, gold in data:
            P = m.head_probs(S, T)
            nll -= np.log(P[np.arange(len(gold)), gold] + 1e-12).sum()
        if best is None or nll < best[1]:
            best = (float(T), nll)
    print(f"fitted arc temperature T = {best[0]:.1f}")
    z = np.load(m.path, allow_pickle=False)
    arrays = {k: z[k] for k in z.files if k != "meta"}
    meta = json.loads(str(z["meta"]))
    meta["arc_temperature"] = round(best[0], 2)
    z.close()
    np.savez_compressed(m.path, meta=np.array(json.dumps(meta)), **arrays)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--fit", action="store_true", help="fit the arc temperature and store it in the model file")
    a = ap.parse_args()
    m = ParserModel()
    if a.fit:
        fit(m, list(read_conllu(os.path.join(ROOT, "data", "ud", f"en_ewt-ud-{a.split}.conllu")))[: a.n])
        m = ParserModel()
    sents = list(read_conllu(os.path.join(ROOT, "data", "ud", f"en_ewt-ud-{a.split}.conllu")))[: a.n]
    words = []        # (conf, correct)
    per_sent = []     # list of (conf, correct) per sentence
    for s in sents:
        out = m.parse([t["form"] for t in s])
        row = [(p["conf"], p["head"] == g["head"]) for g, p in zip(s, out) if g["upos"] != "PUNCT"]
        words += row
        per_sent.append(row)
    print(f"{len(sents)} sentences, {len(words)} words ({a.split})\n")
    print("calibration: confidence bin -> share of heads that are right")
    bins = [0, .3, .5, .6, .7, .8, .9, .95, .99, 1.01]
    for lo, hi in zip(bins, bins[1:]):
        b = [c for p, c in words if lo <= p < hi]
        if b:
            print(f"  {lo:4.2f}-{min(hi, 1):4.2f}  {len(b):6} words  {sum(b) / len(b):6.1%} right")
    print("\nthreshold   sentences kept   kept sentences with a wrong head   errors caught")
    errs = sum(1 for r in per_sent if not all(c for _, c in r))
    for th in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        kept = [r for r in per_sent if all(p >= th for p, _ in r)]
        bad = sum(1 for r in kept if not all(c for _, c in r))
        caught = errs - bad
        print(f"  {th:4.2f}       {len(kept) / len(per_sent):6.1%}          {bad / max(1, len(kept)):6.1%}"
              f"                         {caught / max(1, errs):6.1%}")


if __name__ == "__main__":
    main()

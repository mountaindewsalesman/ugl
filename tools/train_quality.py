"""Train the sentence-quality grader: P(the UGL is right) for one corpus sentence.

    python tools/train_quality.py                        # labels from tools/make_quality_labels.py
    python tools/train_quality.py --skip en_ctetex en_lines

Input text (what the grader sees):
    <English source> [SEP] <the decoder's English for the UGL> [SEP] flags: <issue kinds>
The decoder's English lets a pretrained English model compare the two
meanings; the flags are the corpus builder's own checks.

Reports, on EWT test (never trained on) and on the hand-graded review sample
(data/review/grades-*.jsonl: ok / minor / wrong by a person):
  AUC                  how well the score ranks right above wrong (0.5 = chance)
  right at keep rate   precision when keeping the best-scored x% (compare with
                       the rule-based filter at its own keep rate)
Saves the model to data/build/quality-grader/ (torch; corpus building only).
"""
import argparse
import glob
import json
import os
import random
import sys

import numpy as np
import torch
from torch import nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(ROOT, "data", "build")
sys.path.insert(0, os.path.join(ROOT, "tools"))

from quality import DEFAULT_PATH, Grader, save, score, text_of  # noqa: E402


def auc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def precision_at(scores, labels, keep):
    order = np.argsort(-np.asarray(scores))
    k = max(1, int(round(keep * len(scores))))
    return float(np.mean(np.asarray(labels)[order[:k]]))


def report(name, scores, labels, rule_kept):
    """rule_kept: which rows the rule-based filter keeps (for comparison at its keep rate)."""
    keep = sum(rule_kept) / len(rule_kept)
    rule = sum(l for l, k in zip(labels, rule_kept) if k) / max(1, sum(rule_kept))
    print(f"  {name}: {len(labels)} sentences, {np.mean(labels):.0%} right; AUC {auc(scores, labels):.3f}")
    print(f"    keeping {keep:.0%}:  rule filter {rule:.0%} right   grader {precision_at(scores, labels, keep):.0%} right")
    for k in (0.2, 0.4, 0.6):
        print(f"    keeping {k:.0%}:  grader {precision_at(scores, labels, k):.0%} right")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=os.path.join(BUILD, "quality-labels.jsonl"))
    ap.add_argument("--model", default="google/electra-base-discriminator")
    # left out by default: their annotation conventions differ from what the parser
    # learned, so many "wrong" labels are convention mismatches, not errors (with them
    # the grader ranks EWT test at AUC 0.80; without, 0.83)
    ap.add_argument("--skip", nargs="*", default=["en_ctetex", "en_lines", "en_partut", "en_childes", "en_atis"],
                    help="treebanks left out of training")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--out", default=DEFAULT_PATH)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    rows = [json.loads(l) for l in open(a.labels, encoding="utf-8")]
    train = [r for r in rows if r["split"] == "train" and r["treebank"] not in a.skip]
    test = [r for r in rows if r["split"] == "test"]
    review = [json.loads(l) for f in sorted(glob.glob(os.path.join(ROOT, "data", "review", "grades-*.jsonl")))
              for l in open(f, encoding="utf-8")]
    print(f"train {len(train)} ({np.mean([r['label'] for r in train]):.0%} right), test {len(test)}, "
          f"hand-graded {len(review)}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    random.seed(0)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = Grader(a.model).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * ((len(train) + a.bs - 1) // a.bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / (0.1 * steps)) * max(0.0, 1 - s / steps))
    scaler = torch.amp.GradScaler(enabled=device == "cuda")
    lossf = nn.BCEWithLogitsLoss()
    for ep in range(a.epochs):
        model.train()
        random.shuffle(train)
        total = 0.0
        for i in range(0, len(train), a.bs):
            batch = train[i:i + a.bs]
            x, y = zip(*[text_of(r) for r in batch])
            enc = tok(list(x), list(y), truncation=True, max_length=256, padding=True, return_tensors="pt").to(device)
            lab = torch.tensor([float(r["label"]) for r in batch], device=device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                loss = lossf(model(**enc).float(), lab)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item() * len(batch)
        s = score(model, tok, test, device)
        print(f"epoch {ep + 1}: train loss {total / len(train):.3f}; EWT test AUC {auc(s, [r['label'] for r in test]):.3f}",
              flush=True)
    print("results")
    s = score(model, tok, test, device)
    test_scores = s
    report("EWT test (right = same UGL as the gold tree)", s, [r["label"] for r in test],
           [not r["kinds"] and r["roundtrip"] for r in test])
    if review:
        s = score(model, tok, review, device)
        report("hand-graded review (right = ok or minor)", s, [r["grade"] != "wrong" for r in review],
               [r["bucket"] == "kept" for r in review])
    # score cuts on the held-out test: keep share -> (score, share right), for build_corpus
    labels = np.asarray([r["label"] for r in test])
    order = np.argsort(-np.asarray(test_scores))
    cuts = []
    for keep in [x / 20 for x in range(1, 20)]:
        k = max(1, int(round(keep * len(order))))
        cuts.append({"keep": keep, "score": float(test_scores[order[k - 1]]), "right": float(labels[order[:k]].mean())})
    save(model, tok, a.out, {"base": a.model, "skip": a.skip, "epochs": a.epochs, "cuts": cuts,
                             "test_auc": auc(test_scores, labels.tolist())})
    print("saved", a.out)


if __name__ == "__main__":
    main()

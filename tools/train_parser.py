"""Train the UGL tagger + dependency parser on Universal Dependencies English.

    python tools/train_parser.py                       # all data/ud/*-ud-train.conllu, GPU if present
    python tools/train_parser.py --train en_ewt        # one treebank
    python tools/train_parser.py --epochs 2 --quick    # smoke test

Exports ugl/parser/weights/parser-ewt.npz, which ugl.parser.model runs with
numpy alone.  Prints dev UPOS / UAS / LAS each epoch and test scores at the end.
Pretrained vectors come from spaCy's en_core_web_md (training-time only),
reduced to 128 dims by PCA and frozen.
"""
import argparse
import collections
import glob
import json
import os
import random
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ugl.parser.conllu import read_conllu
from ugl.parser.features import AFFIX_KEYS, affixes, norm, shape

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UD = os.path.join(ROOT, "data", "ud")
OUT = os.path.join(ROOT, "ugl", "parser", "weights", "parser-ewt.npz")


def build_vocab(items, min_count=1, extra=()):
    c = collections.Counter(items)
    return ["<pad>", "<unk>"] + list(extra) + sorted(k for k, v in c.items() if v >= min_count and k not in extra)


def pretrained(train_forms, n_max=60000, dim=128):
    """Frozen word vectors for frequent forms (train + Wikipedia sample)."""
    import spacy
    nlp = spacy.load("en_core_web_md")
    counts = collections.Counter(train_forms)
    raw = sorted(glob.glob(os.path.join(ROOT, "data", "raw", "*.txt")))[:6]
    for fp in raw:
        with open(fp, encoding="utf-8") as f:
            counts.update(norm(w) for w in re.findall(r"[A-Za-z][A-Za-z'-]*", f.read()))
    forms, vecs = [], []
    for w, _ in counts.most_common():
        if len(forms) >= n_max:
            break
        if nlp.vocab.has_vector(w):
            forms.append(w)
            vecs.append(nlp.vocab.get_vector(w))
    X = np.stack(vecs).astype(np.float32)
    X -= X.mean(0)
    U, S, Vt = np.linalg.svd(X[: min(len(X), 20000)], full_matrices=False)
    P = X @ Vt[:dim].T
    P /= P.std() + 1e-6
    table = np.concatenate([np.zeros((2, dim), np.float32), P.astype(np.float32)])
    return ["<pad>", "<unk>"] + forms, table


class Net(nn.Module):
    def __init__(self, sizes, pre_table, wdim=100, adim=32, sdim=16, hidden=200, layers=2,
                 arc=400, lab=100, n_lab=40, drop=0.33):
        super().__init__()
        self.emb = nn.ModuleDict({"word": nn.Embedding(sizes["word"], wdim, padding_idx=0),
                                  "shape": nn.Embedding(sizes["shape"], sdim, padding_idx=0)})
        for k in AFFIX_KEYS:
            self.emb[k] = nn.Embedding(sizes[k], adim, padding_idx=0)
        self.pre = nn.Embedding.from_pretrained(torch.tensor(pre_table), freeze=True, padding_idx=0)
        din = wdim + pre_table.shape[1] + len(AFFIX_KEYS) * adim + sdim
        self.lstm = nn.LSTM(din, hidden, num_layers=layers, bidirectional=True, batch_first=True, dropout=drop)
        d = 2 * hidden
        self.root = nn.Parameter(torch.randn(d) * 0.1)
        self.upos = nn.Linear(d, sizes["upos"])
        self.xpos = nn.Linear(d, sizes["xpos"])
        self.feats = nn.Linear(d, sizes["feats"])
        self.arc_dep, self.arc_head = nn.Linear(d, arc), nn.Linear(d, arc)
        self.arc_W = nn.Parameter(torch.zeros(arc, arc))
        self.arc_u = nn.Parameter(torch.zeros(arc))
        self.lab_dep, self.lab_head = nn.Linear(d, lab), nn.Linear(d, lab)
        self.lab_W = nn.Parameter(torch.zeros(n_lab, lab, lab))
        self.lab_lin = nn.Linear(2 * lab, n_lab)
        self.drop = nn.Dropout(drop)

    def forward(self, b):
        parts = [self.emb["word"](b["word"]), self.pre(b["pre"])]
        parts += [self.emb[k](b[k]) for k in AFFIX_KEYS] + [self.emb["shape"](b["shape"])]
        x = self.drop(torch.cat(parts, -1))
        h, _ = self.lstm(x)
        h = self.drop(h)
        B = h.shape[0]
        H = torch.cat([self.root.expand(B, 1, -1), h], 1)
        dep = self.drop(F.relu(self.arc_dep(H)))
        head = self.drop(F.relu(self.arc_head(H)))
        S = dep @ self.arc_W @ head.transpose(1, 2) + (head @ self.arc_u).unsqueeze(1)
        ld = self.drop(F.relu(self.lab_dep(H)))
        lh = self.drop(F.relu(self.lab_head(H)))
        return h, S, ld, lh

    def label_scores(self, ld, lh, heads):
        # ld, lh: (B, n+1, a); heads: (B, n+1) -> (B, n+1, L)
        hsel = torch.gather(lh, 1, heads.unsqueeze(-1).expand(-1, -1, lh.shape[-1]))
        bil = torch.einsum("bia,lac,bic->bil", ld, self.lab_W, hsel)
        return bil + self.lab_lin(torch.cat([ld, hsel], -1))


def encode_sent(s, voc):
    n = [norm(t["form"]) for t in s]
    ids = lambda v, keys: [voc[v].get(k, 1) for k in keys]
    aff = [affixes(t["form"]) for t in s]
    e = {"word": ids("word", n), "pre": ids("pre", n), "shape": ids("shape", [shape(t["form"]) for t in s])}
    for k in AFFIX_KEYS:
        e[k] = ids(k, [a[k] for a in aff])
    e["upos"] = ids("upos", [t["upos"] for t in s])
    e["xpos"] = [IGNORE if t["xpos"] is None else voc["xpos"].get(t["xpos"], 1) for t in s]
    e["feats"] = ids("feats", [t["feats"] for t in s])
    e["head"] = [t["head"] for t in s]
    e["deprel"] = ids("deprel", [t["deprel"] for t in s])
    return e


DEVICE = torch.device("cpu")
TREEBANKS = []
# Penn Treebank XPOS tags (VBD, VBN, NNS...) are only annotated in these; the
# XPOS head learns from them alone, every other layer learns from all treebanks
PENN_XPOS = {"en_ewt", "en_gum"}
IGNORE = -100


def batchify(encs, word_drop=0.0, word_freq=None):
    n = max(len(e["word"]) for e in encs)
    B = {}
    for k in encs[0]:
        B[k] = torch.zeros(len(encs), n, dtype=torch.long)
        for i, e in enumerate(encs):
            B[k][i, : len(e[k])] = torch.tensor(e[k])
    if word_drop:
        r = torch.rand(B["word"].shape)
        f = word_freq.cpu()[B["word"]]
        drop = (r < word_drop / (word_drop + f)) & (B["word"] > 1)
        B["word"] = B["word"].masked_fill(drop, 1)
    B["mask"] = B["word"].ne(0) | B["pre"].ne(0) | B["shape"].ne(0)
    return {k: v.to(DEVICE) for k, v in B.items()}


def evaluate(net, encs, bs=64):
    net.eval()
    tot = up = ua = la = 0
    with torch.no_grad():
        for i in range(0, len(encs), bs):
            b = batchify(encs[i:i + bs])
            h, S, ld, lh = net(b)
            m = b["mask"]
            pu = net.upos(h).argmax(-1)
            # greedy heads for speed during training (exact decoding at export)
            Sm = S[:, 1:, :].clone()
            n = Sm.shape[1]
            pad = torch.cat([torch.ones_like(m[:, :1]), m], 1)
            Sm = Sm.masked_fill(~pad.unsqueeze(1), -1e9)
            ph = Sm.argmax(-1)
            heads_full = torch.cat([torch.zeros_like(ph[:, :1]), ph], 1)
            pl = net.label_scores(ld, lh, heads_full)[:, 1:].argmax(-1)
            tot += m.sum().item()
            up += ((pu == b["upos"]) & m).sum().item()
            ua += ((ph == b["head"]) & m).sum().item()
            la += ((ph == b["head"]) & (pl == b["deprel"]) & m).sum().item()
    net.train()
    return up / tot, ua / tot, la / tot


def export(net, vocabs, path):
    sd = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
    out = {}
    for k, v in sd.items():
        if k.startswith("emb."):
            out[k.replace(".weight", "")] = v
        elif k == "pre.weight":
            out["emb.pre"] = v
        else:
            out[k] = v
    out = {k: v.astype(np.float16) if v.ndim >= 2 and v.size > 50000 else v.astype(np.float32) for k, v in out.items()}
    meta = {"vocabs": vocabs, "lstm_layers": net.lstm.num_layers, "trained": time.strftime("%Y-%m-%d %H:%M"),
            "treebanks": TREEBANKS}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, meta=np.array(json.dumps(meta)), **out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--quick", action="store_true", help="tiny subset for a smoke test")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--train", nargs="+", default=None,
                    help="treebank prefixes (e.g. en_ewt en_gum); default: every *-ud-train.conllu in data/ud")
    ap.add_argument("--hidden", type=int, default=300)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    global DEVICE
    DEVICE = torch.device(a.device)
    torch.manual_seed(a.seed); random.seed(a.seed)
    torch.set_num_threads(max(1, os.cpu_count() - 1))

    files = ([os.path.join(UD, f"{t}-ud-train.conllu") for t in a.train] if a.train
             else sorted(glob.glob(os.path.join(UD, "*-ud-train.conllu"))))
    train = []
    for f in files:
        penn = os.path.basename(f).split("-ud-")[0] in PENN_XPOS
        for sent in read_conllu(f):
            if not penn:
                for t in sent:
                    t["xpos"] = None
            train.append(sent)
    print("training on", [os.path.basename(f) for f in files], "device", DEVICE, flush=True)
    dev = list(read_conllu(os.path.join(UD, "en_ewt-ud-dev.conllu")))
    test = list(read_conllu(os.path.join(UD, "en_ewt-ud-test.conllu")))
    if a.quick:
        random.shuffle(train)
        train, dev = train[:800], dev[:200]
    toks = [t for s in train for t in s]
    print(f"{len(train)} train sentences, {len(toks)} tokens", flush=True)

    vocabs = {"word": build_vocab([norm(t["form"]) for t in toks], 2),
              "shape": build_vocab([shape(t["form"]) for t in toks])}
    for k in AFFIX_KEYS:
        vocabs[k] = build_vocab([affixes(t["form"])[k] for t in toks], 3)
    for k in ("upos", "xpos", "feats", "deprel"):
        vocabs[k] = build_vocab([t[k] for t in toks if t[k] is not None])
    pre_forms, pre_table = pretrained([norm(t["form"]) for t in toks])
    vocabs["pre"] = pre_forms
    voc = {k: {s: i for i, s in enumerate(v)} for k, v in vocabs.items()}
    print("vocab sizes:", {k: len(v) for k, v in vocabs.items()}, flush=True)

    wc = collections.Counter(norm(t["form"]) for t in toks)
    word_freq = torch.tensor([wc.get(w, 0) for w in vocabs["word"]], dtype=torch.float)

    E_train = [encode_sent(s, voc) for s in train]
    E_dev = [encode_sent(s, voc) for s in dev]
    E_test = [encode_sent(s, voc) for s in test]
    sizes = {k: len(v) for k, v in vocabs.items()}
    net = Net(sizes, pre_table, n_lab=sizes["deprel"], hidden=a.hidden, layers=a.layers).to(DEVICE)
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=a.lr, betas=(0.9, 0.9))
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, 0.95)

    best, best_state = -1, None
    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        order = sorted(range(len(E_train)), key=lambda i: len(E_train[i]["word"]) + random.random() * 4)
        batches = [order[i:i + a.bs] for i in range(0, len(order), a.bs)]
        random.shuffle(batches)
        tot_loss = 0.0
        for bi in batches:
            b = batchify([E_train[i] for i in bi], word_drop=0.25, word_freq=word_freq)
            h, S, ld, lh = net(b)
            m = b["mask"]
            pad = torch.cat([torch.ones_like(m[:, :1]), m], 1)
            Sm = S[:, 1:, :].masked_fill(~pad.unsqueeze(1), -1e9)
            loss = F.cross_entropy(Sm[m], b["head"][m])
            heads_full = torch.cat([torch.zeros_like(b["head"][:, :1]), b["head"]], 1)
            L = net.label_scores(ld, lh, heads_full)[:, 1:]
            loss = loss + F.cross_entropy(L[m], b["deprel"][m])
            for k in ("upos", "xpos", "feats"):
                gold = b[k][m]
                if (gold != IGNORE).any():          # a batch may hold no Penn-tagged sentence
                    loss = loss + F.cross_entropy(getattr(net, k)(h)[m], gold, ignore_index=IGNORE)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            tot_loss += loss.item()
        sched.step()
        up, ua, la = evaluate(net, E_dev)
        print(f"epoch {ep:2}  loss {tot_loss/len(batches):.3f}  dev UPOS {up:.4f} UAS {ua:.4f} LAS {la:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        if la > best:
            best = la
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    up, ua, la = evaluate(net, E_test)
    print(f"TEST  UPOS {up:.4f}  UAS {ua:.4f}  LAS {la:.4f}  (greedy heads)", flush=True)
    global TREEBANKS
    TREEBANKS = [os.path.basename(f).split("-ud-")[0] for f in files]
    export(net.cpu(), vocabs, a.out)
    print("saved", a.out, f"{os.path.getsize(a.out)/1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main()

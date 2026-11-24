"""Train the transformer UGL tagger + dependency parser (v0.3 parser).

    python tools/train_parser_tf.py                       # all treebanks in data/ud, GPU
    python tools/train_parser_tf.py --epochs 1 --quick    # smoke test

Encoder: a small pretrained transformer (ELECTRA-small discriminator, 12
layers x 256 dims, 14M parameters), fine-tuned.  Each word is represented by
a learned mix of all the transformer's layers at the word's first word piece.
Heads are the same as the BiLSTM parser's (tools/train_parser.py): linear
UPOS / XPOS / FEATS taggers, a biaffine arc scorer and a biaffine label scorer.

Exports ugl/parser/weights/parser-en-tf.npz, which ugl.parser.model runs with
numpy alone (ugl/transformer.py), then checks numpy == torch predictions.
"""
import argparse
import glob
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ugl.parser.conllu import read_conllu

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UD = os.path.join(ROOT, "data", "ud")
OUT = os.path.join(ROOT, "ugl", "parser", "weights", "parser-en-tf.npz")
MODEL = "google/electra-small-discriminator"
PENN_XPOS = {"en_ewt", "en_gum"}
IGNORE = -100
MAX_PIECES = 250
DEVICE = torch.device("cpu")


def build_vocab(items):
    return ["<pad>", "<unk>"] + sorted(set(items))


class TfNet(nn.Module):
    def __init__(self, model_name, sizes, arc=400, lab=100, drop=0.33):
        super().__init__()
        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained(model_name)
        d = self.bert.config.hidden_size
        nl = self.bert.config.num_hidden_layers + 1
        self.mix = nn.Parameter(torch.zeros(nl))         # softmax weights over layers 0..12
        self.gamma = nn.Parameter(torch.ones(1))
        self.root = nn.Parameter(torch.randn(d) * 0.1)
        self.upos = nn.Linear(d, sizes["upos"])
        self.xpos = nn.Linear(d, sizes["xpos"])
        self.feats = nn.Linear(d, sizes["feats"])
        self.arc_dep, self.arc_head = nn.Linear(d, arc), nn.Linear(d, arc)
        self.arc_W = nn.Parameter(torch.zeros(arc, arc))
        self.arc_u = nn.Parameter(torch.zeros(arc))
        self.lab_dep, self.lab_head = nn.Linear(d, lab), nn.Linear(d, lab)
        self.lab_W = nn.Parameter(torch.zeros(sizes["deprel"], lab, lab))
        self.lab_lin = nn.Linear(2 * lab, sizes["deprel"])
        self.drop = nn.Dropout(drop)

    def words(self, b):
        out = self.bert(input_ids=b["ids"], attention_mask=b["att"], output_hidden_states=True)
        hs = torch.stack(out.hidden_states, 0)                         # (L, B, T, d)
        w = torch.softmax(self.mix, 0)
        h = self.gamma * (w[:, None, None, None] * hs).sum(0)          # (B, T, d)
        idx = b["first"].unsqueeze(-1).expand(-1, -1, h.shape[-1])     # first piece of each word
        return torch.gather(h, 1, idx)                                 # (B, n, d)

    def forward(self, b):
        h = self.drop(self.words(b))
        B = h.shape[0]
        H = torch.cat([self.root.expand(B, 1, -1), h], 1)
        dep = self.drop(F.relu(self.arc_dep(H)))
        head = self.drop(F.relu(self.arc_head(H)))
        S = dep @ self.arc_W @ head.transpose(1, 2) + (head @ self.arc_u).unsqueeze(1)
        ld = self.drop(F.relu(self.lab_dep(H)))
        lh = self.drop(F.relu(self.lab_head(H)))
        return h, S, ld, lh

    def label_scores(self, ld, lh, heads):
        hsel = torch.gather(lh, 1, heads.unsqueeze(-1).expand(-1, -1, lh.shape[-1]))
        bil = torch.einsum("bia,lac,bic->bil", ld, self.lab_W, hsel)
        return bil + self.lab_lin(torch.cat([ld, hsel], -1))


def encode_sent(s, voc, wp):
    """wp: ugl.transformer.WordPiece (the same tokenizer numpy inference uses)."""
    ids, first = wp.encode_words([t["form"] for t in s])
    if len(ids) > MAX_PIECES:
        return None
    ids_ = lambda v, keys: [voc[v].get(k, 1) for k in keys]
    return {"ids": ids, "first": first,
            "upos": ids_("upos", [t["upos"] for t in s]),
            "xpos": [IGNORE if t["xpos"] is None else voc["xpos"].get(t["xpos"], 1) for t in s],
            "feats": ids_("feats", [t["feats"] for t in s]),
            "head": [t["head"] for t in s],
            "deprel": ids_("deprel", [t["deprel"] for t in s])}


def batchify(encs):
    T = max(len(e["ids"]) for e in encs)
    n = max(len(e["first"]) for e in encs)
    B = {"ids": torch.zeros(len(encs), T, dtype=torch.long), "att": torch.zeros(len(encs), T, dtype=torch.long)}
    for k in ("first", "upos", "xpos", "feats", "head", "deprel"):
        B[k] = torch.zeros(len(encs), n, dtype=torch.long)
    B["mask"] = torch.zeros(len(encs), n, dtype=torch.bool)
    for i, e in enumerate(encs):
        B["ids"][i, : len(e["ids"])] = torch.tensor(e["ids"])
        B["att"][i, : len(e["ids"])] = 1
        m = len(e["first"])
        for k in ("first", "upos", "xpos", "feats", "head", "deprel"):
            B[k][i, :m] = torch.tensor(e[k])
        B["mask"][i, :m] = True
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
            pad = torch.cat([torch.ones_like(m[:, :1]), m], 1)
            ph = S[:, 1:, :].masked_fill(~pad.unsqueeze(1), -1e9).argmax(-1)
            heads_full = torch.cat([torch.zeros_like(ph[:, :1]), ph], 1)
            pl = net.label_scores(ld, lh, heads_full)[:, 1:].argmax(-1)
            tot += m.sum().item()
            up += ((pu == b["upos"]) & m).sum().item()
            ua += ((ph == b["head"]) & m).sum().item()
            la += ((ph == b["head"]) & (pl == b["deprel"]) & m).sum().item()
    net.train()
    return up / tot, ua / tot, la / tot


def export(net, vocabs, wp, path, treebanks):
    from ugl.transformer import export_names
    sd = {k: v.detach().cpu().float().numpy() for k, v in net.state_dict().items()}
    out = {}
    for k, v in sd.items():
        if k.startswith("bert."):
            name = export_names(k[5:])
            if name is not None:
                out["tf." + name] = v
        else:
            out[k] = v
    out["mix"] = (torch.softmax(net.mix, 0) * net.gamma).detach().cpu().numpy()
    out.pop("gamma", None)
    out = {k: v.astype(np.float16) if v.ndim >= 2 and v.size > 50000 else v.astype(np.float32)
           for k, v in out.items()}
    cfg = net.bert.config
    meta = {"encoder": "transformer", "vocabs": vocabs, "wordpiece": wp.vocab_list,
            "lowercase": wp.lowercase,
            "transformer": {"layers": cfg.num_hidden_layers, "heads": cfg.num_attention_heads,
                            "eps": cfg.layer_norm_eps, "base": MODEL},
            "trained": time.strftime("%Y-%m-%d %H:%M"), "treebanks": treebanks}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, meta=np.array(json.dumps(meta)), **out)


def parity_check(net, path, sents, voc, wp, n=60):
    from ugl.parser.model import ParserModel
    m = ParserModel(path)
    net.eval()
    same_u = same_h = tot = 0
    with torch.no_grad():
        for sent in sents[:n]:
            e = encode_sent(sent, voc, wp)
            if e is None:
                continue
            b = batchify([e])
            h, S, _, _ = net(b)
            tu = net.upos(h).argmax(-1)[0].tolist()
            th = S[0, 1:, :].argmax(-1).tolist()
            words = [t["form"] for t in sent]
            Hs = m.encode(words)
            nu = m.lin("upos", Hs).argmax(1).tolist()
            Sn = m.arc_scores(Hs)
            nh = Sn[1:].argmax(1).tolist()
            same_u += sum(a == c for a, c in zip(tu, nu))
            same_h += sum(a == c for a, c in zip(th, nh))
            tot += len(words)
    print(f"parity numpy vs torch on {n} sentences: UPOS {same_u/tot:.4f}  heads {same_h/tot:.4f}", flush=True)
    if same_u / tot < 0.995 or same_h / tot < 0.99:
        raise SystemExit("numpy inference does not match the trained model")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4, help="transformer learning rate")
    ap.add_argument("--head-lr", type=float, default=2e-3)
    ap.add_argument("--model", default="google/electra-small-discriminator")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    global DEVICE, MODEL
    DEVICE, MODEL = torch.device(a.device), a.model
    torch.manual_seed(a.seed); random.seed(a.seed)

    from ugl.transformer import WordPiece
    from transformers import AutoTokenizer
    hf = AutoTokenizer.from_pretrained(a.model)
    wp = WordPiece([t for t, _ in sorted(hf.get_vocab().items(), key=lambda kv: kv[1])],
                   lowercase=getattr(hf, "do_lower_case", True))

    files = sorted(glob.glob(os.path.join(UD, "*-ud-train.conllu")))
    train = []
    for f in files:
        penn = os.path.basename(f).split("-ud-")[0] in PENN_XPOS
        for sent in read_conllu(f):
            if not penn:
                for t in sent:
                    t["xpos"] = None
            train.append(sent)
    dev = list(read_conllu(os.path.join(UD, "en_ewt-ud-dev.conllu")))
    test = list(read_conllu(os.path.join(UD, "en_ewt-ud-test.conllu")))
    if a.quick:
        random.shuffle(train)
        train, dev = train[:800], dev[:200]
    toks = [t for s in train for t in s]
    vocabs = {k: build_vocab([t[k] for t in toks if t[k] is not None]) for k in ("upos", "xpos", "feats", "deprel")}
    voc = {k: {s: i for i, s in enumerate(v)} for k, v in vocabs.items()}
    E_train = [e for e in (encode_sent(s, voc, wp) for s in train) if e is not None]
    E_dev = [e for e in (encode_sent(s, voc, wp) for s in dev) if e is not None]
    E_test = [e for e in (encode_sent(s, voc, wp) for s in test) if e is not None]
    print(f"training on {[os.path.basename(f) for f in files]}: {len(E_train)} sentences, device {DEVICE}", flush=True)

    sizes = {k: len(v) for k, v in vocabs.items()}
    net = TfNet(a.model, sizes).to(DEVICE)
    bert_p = list(net.bert.parameters())
    ids = {id(p) for p in bert_p}
    head_p = [p for p in net.parameters() if id(p) not in ids]
    opt = torch.optim.AdamW([{"params": bert_p, "lr": a.lr}, {"params": head_p, "lr": a.head_lr}],
                            betas=(0.9, 0.98), weight_decay=0.01)
    steps = a.epochs * ((len(E_train) + a.bs - 1) // a.bs)
    warm = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (steps - s) / max(1, steps - warm)))

    best, best_state = -1, None
    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        order = sorted(range(len(E_train)), key=lambda i: len(E_train[i]["ids"]) + random.random() * 6)
        batches = [order[i:i + a.bs] for i in range(0, len(order), a.bs)]
        random.shuffle(batches)
        tot_loss = 0.0
        for bi in batches:
            b = batchify([E_train[i] for i in bi])
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
                if (gold != IGNORE).any():
                    loss = loss + F.cross_entropy(getattr(net, k)(h)[m], gold, ignore_index=IGNORE)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            sched.step()
            tot_loss += loss.item()
        up, ua, la = evaluate(net, E_dev)
        print(f"epoch {ep:2}  loss {tot_loss/len(batches):.3f}  dev UPOS {up:.4f} UAS {ua:.4f} LAS {la:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        if la > best:
            best = la
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    up, ua, la = evaluate(net, E_test)
    print(f"TEST  UPOS {up:.4f}  UAS {ua:.4f}  LAS {la:.4f}  (greedy heads)", flush=True)
    DEVICE = torch.device("cpu")
    net = net.cpu()
    export(net, vocabs, wp, a.out, [os.path.basename(f).split("-ud-")[0] for f in files])
    parity_check(net, a.out, dev, voc, wp)
    print("saved", a.out, f"{os.path.getsize(a.out)/1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main()

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
WSD_PRIOR = 0.05
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
        # sense head (ugl/wsd_gloss.py): its own layer mix, context and gloss projections
        self.wmix = nn.Parameter(torch.zeros(nl))
        self.wgamma = nn.Parameter(torch.ones(1))
        self.wsd_ctx, self.wsd_gloss = nn.Linear(d, 256), nn.Linear(d, 256)

    def wsd_states(self, ids, att):
        out = self.bert(input_ids=ids, attention_mask=att, output_hidden_states=True)
        hs = torch.stack(out.hidden_states, 0)
        return self.wgamma * (torch.softmax(self.wmix, 0)[:, None, None, None] * hs).sum(0)   # (B, T, d)

    def wsd_logits(self, w):
        """Scores of every target word against every gloss in the batch
        (non-candidates masked out)."""
        h = self.wsd_states(w["ids"], w["att"])
        ctx = self.wsd_ctx(self.drop(h[w["tb"], w["tp"]]))                     # (targets, k)
        g = self.wsd_states(w["gids"], w["gatt"])[:, 0]                          # [CLS] of each gloss
        scores = ctx @ self.wsd_gloss(g).T                                       # (targets, glosses)
        return scores.masked_fill(~w["cmask"], -1e9)

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


# ------------------------------------------------------------------ sense data
def wsd_examples(lex, wp, ranges):
    """SemCor sentences -> (piece ids, [(first piece, candidate senses, gold index)])."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from eval_wsd import items
    out = []
    for start, end in ranges:
        for words, targets in items(end - start, start):
            ids, first = wp.encode_words(words)
            if len(ids) > MAX_PIECES:
                continue
            tg = []
            for i, lemma, sense, cls in targets:
                cands = lex.senses(lemma, cls)[:12]
                if i < len(first) and len(cands) >= 2 and sense in cands:
                    tg.append((first[i], cands, cands.index(sense)))
            if tg:
                out.append((ids, tg))
    return out


GLOSS_IDS = {}


def gloss_ids(lex, wp, sense):
    from ugl.wsd_gloss import gloss_words, MAX_GLOSS_PIECES
    if sense not in GLOSS_IDS:
        ids, _ = wp.encode_words(gloss_words(lex, sense))
        GLOSS_IDS[sense] = ids[: MAX_GLOSS_PIECES - 1] + [ids[-1]]
    return GLOSS_IDS[sense]


def wsd_batch(exs, lex, wp):
    T = max(len(e[0]) for e in exs)
    ids = torch.zeros(len(exs), T, dtype=torch.long)
    att = torch.zeros(len(exs), T, dtype=torch.long)
    tb, tp, targets, senses = [], [], [], {}
    for bi, (pid, tg) in enumerate(exs):
        ids[bi, : len(pid)] = torch.tensor(pid)
        att[bi, : len(pid)] = 1
        for pos, cands, gold in tg:
            tb.append(bi)
            tp.append(pos)
            targets.append((cands, gold))
            for c in cands:
                senses.setdefault(c, len(senses))
    G = [gloss_ids(lex, wp, s) for s in senses]
    gT = max(len(g) for g in G)
    gids = torch.zeros(len(G), gT, dtype=torch.long)
    gatt = torch.zeros(len(G), gT, dtype=torch.long)
    for i, g in enumerate(G):
        gids[i, : len(g)] = torch.tensor(g)
        gatt[i, : len(g)] = 1
    cmask = torch.zeros(len(targets), len(G), dtype=torch.bool)
    gold = torch.zeros(len(targets), dtype=torch.long)
    for t, (cands, g) in enumerate(targets):
        for c in cands:
            cmask[t, senses[c]] = True
        gold[t] = senses[cands[g]]
    d = {"ids": ids, "att": att, "tb": torch.tensor(tb), "tp": torch.tensor(tp), "gids": gids, "gatt": gatt,
         "cmask": cmask, "gold": gold}
    return {k: v.to(DEVICE) for k, v in d.items()}


def wsd_eval(net, exs, lex, wp, bs=8):
    net.eval()
    hit = tot = 0
    with torch.no_grad():
        for i in range(0, len(exs), bs):
            w = wsd_batch(exs[i:i + bs], lex, wp)
            hit += (net.wsd_logits(w).argmax(-1) == w["gold"]).sum().item()
            tot += len(w["gold"])
    net.train()
    return hit / max(1, tot)


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
    out["wsd.mix"] = (torch.softmax(net.wmix, 0) * net.wgamma).detach().cpu().numpy()
    for k in ("gamma", "wmix", "wgamma"):
        out.pop(k, None)
    out = {k: v.astype(np.float16) if v.ndim >= 2 and v.size > 50000 else v.astype(np.float32)
           for k, v in out.items()}
    cfg = net.bert.config
    meta = {"encoder": "transformer", "vocabs": vocabs, "wordpiece": wp.vocab_list,
            "lowercase": wp.lowercase,
            "transformer": {"layers": cfg.num_hidden_layers, "heads": cfg.num_attention_heads,
                            "eps": cfg.layer_norm_eps, "base": MODEL},
            "trained": time.strftime("%Y-%m-%d %H:%M"), "treebanks": treebanks,
            "wsd_prior": WSD_PRIOR}           # retune with tools/eval_wsd.py --gloss
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
    ap.add_argument("--wsd-bs", type=int, default=8, help="SemCor sentences per sense batch (one per parser batch)")
    ap.add_argument("--no-wsd", action="store_true")
    ap.add_argument("--amp", action="store_true", help="bf16 mixed precision on the GPU")
    ap.add_argument("--resume", action="store_true", help="continue from <out>.ckpt.pt after an interruption")
    ap.add_argument("--ckpt-min", type=float, default=15.0,
                    help="also checkpoint mid-epoch every this many minutes, so a killed run loses little")
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

    lex = W_train = W_dev = None
    if not a.no_wsd:
        from ugl.lexicon import Lexicon
        lex = Lexicon()
        # SemCor 20000..22000 is held out: eval_wsd.py tests on 20000..20400, dev here is 21000..21600
        W_train = wsd_examples(lex, wp, [(0, 300 if a.quick else 20000), (22000, 22000 if a.quick else 40000)])
        W_dev = wsd_examples(lex, wp, [(21000, 21100 if a.quick else 21600)])
        print(f"sense data: {len(W_train)} SemCor sentences, dev {len(W_dev)}", flush=True)
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

    best, best_state, start = -1, None, 1
    ckpt = a.out + ".ckpt.pt"
    mid = None                                     # where to pick up inside an epoch
    if a.resume and os.path.exists(ckpt):          # continue an interrupted run
        c = torch.load(ckpt, map_location=DEVICE, weights_only=False)
        net.load_state_dict(c["net"])
        opt.load_state_dict(c["opt"])
        sched.load_state_dict(c["sched"])
        best, best_state = c["best"], c["best_state"]
        random.setstate(c["rng"])
        if c.get("step") is not None:              # saved mid-epoch: redo the rest of that epoch
            start, mid = c["epoch"], c
            print(f"resumed in epoch {c['epoch']} at step {c['step']}/{len(c['batches'])}", flush=True)
        else:
            start = c["epoch"] + 1
            print(f"resumed after epoch {c['epoch']} (best dev score {best:.4f})", flush=True)

    def save(ep, **extra):
        # written to a temporary file, then renamed: a checkpoint is never half-written
        torch.save({"epoch": ep, "net": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "best": best, "best_state": best_state, "rng": random.getstate(), "step": None, **extra},
                   ckpt + ".tmp")
        os.replace(ckpt + ".tmp", ckpt)

    for ep in range(start, a.epochs + 1):
        t0 = last_save = time.time()
        if mid is not None:
            batches, wperm, first, tot_loss = mid["batches"], mid["wperm"], mid["step"], mid["tot_loss"]
            mid = None
        else:
            order = sorted(range(len(E_train)), key=lambda i: len(E_train[i]["ids"]) + random.random() * 6)
            batches = [order[i:i + a.bs] for i in range(0, len(order), a.bs)]
            random.shuffle(batches)
            wperm = list(range(len(W_train))) if W_train else []
            random.shuffle(wperm)
            first, tot_loss = 0, 0.0
        W_ep = [W_train[i] for i in wperm]
        for step in range(first, len(batches)):
            b = batchify([E_train[i] for i in batches[step]])
            # bf16 mixed precision (--amp): same maths, about twice as fast for base-size encoders
            with torch.autocast(DEVICE.type, dtype=torch.bfloat16, enabled=a.amp):
                h, S, ld, lh = net(b)
                m = b["mask"]
                pad = torch.cat([torch.ones_like(m[:, :1]), m], 1)
                Sm = S[:, 1:, :].float().masked_fill(~pad.unsqueeze(1), -1e9)
                loss = F.cross_entropy(Sm[m], b["head"][m])
                heads_full = torch.cat([torch.zeros_like(b["head"][:, :1]), b["head"]], 1)
                L = net.label_scores(ld, lh, heads_full)[:, 1:].float()
                loss = loss + F.cross_entropy(L[m], b["deprel"][m])
                for k in ("upos", "xpos", "feats"):
                    gold = b[k][m]
                    if (gold != IGNORE).any():
                        loss = loss + F.cross_entropy(getattr(net, k)(h)[m].float(), gold, ignore_index=IGNORE)
                if W_ep:                        # one sense batch per parser batch, same optimizer step
                    k0 = (step * a.wsd_bs) % len(W_ep)
                    w = wsd_batch(W_ep[k0:k0 + a.wsd_bs] or W_ep[: a.wsd_bs], lex, wp)
                    loss = loss + F.cross_entropy(net.wsd_logits(w).float(), w["gold"])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            sched.step()
            tot_loss += loss.item()
            if time.time() - last_save > a.ckpt_min * 60 and step + 1 < len(batches):
                save(ep, step=step + 1, batches=batches, wperm=wperm, tot_loss=tot_loss)
                last_save = time.time()
                print(f"  checkpoint: epoch {ep} step {step + 1}/{len(batches)}", flush=True)
        up, ua, la = evaluate(net, E_dev)
        wa = wsd_eval(net, W_dev, lex, wp) if W_dev else 0.0
        print(f"epoch {ep:2}  loss {tot_loss/len(batches):.3f}  dev UPOS {up:.4f} UAS {ua:.4f} LAS {la:.4f}"
              f"  sense {wa:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        if la + wa > best:                  # both jobs count when picking the epoch
            best = la + wa
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        save(ep)                            # checkpoint every epoch
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

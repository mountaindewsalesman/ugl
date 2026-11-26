"""Train a small GPT on the paired corpus, in English or in UGL, for comparison.

    python tools/train_lm.py data/build/corpus-v2 --lang ugl
    python tools/train_lm.py data/build/corpus-v2 --lang english
    python tools/train_lm.py data/build/corpus-v2 --lang ugl --steps 200   # quick check

Both models see the same passages (passages.jsonl from tools/build_corpus.py,
runs of consecutive kept sentences), split the same way (by a hash of the
passage, 2% validation, 3% test), with the same architecture and settings.

Tokens
  ugl      UGL's own tokenizer ids (one id per concept, modifier letters and
           closing tags as their own ids), remapped to the ids that occur in
           the training passages; ids never seen in training are <unk>
  english  a byte-level BPE trained on the training passages, with the same
           vocabulary size as the UGL side (so neither gets more symbols)
  sentence breaks are a token on both sides; <eos> ends a passage

The headline number is bits per English word on the test passages: the total
negative log-likelihood of each test passage, in bits, divided by the number
of English words in it.  It is comparable across the two languages even though
they tokenize differently.  Also written: loss per token, samples (UGL samples
decoded back to English), and model.pt, which tools/export_lm.py packages
as a standard GPT-2 folder for ugl.lm / `ugl continue`.
"""
import argparse
import glob
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PAD, EOS, BRK, UNK = 0, 1, 2, 3            # padding, end of passage, sentence break, unknown
SPECIALS = ["<pad>", "<eos>", "<brk>", "<unk>"]


# ------------------------------------------------------------------ data
def passages(folder):
    """Every kept passage of every build under folder (files may still be growing)."""
    for path in sorted(glob.glob(os.path.join(folder, "*", "passages.jsonl"))):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    p = json.loads(line)
                except json.JSONDecodeError:
                    continue                        # a line still being written
                if p.get("english") and len(p["english"]) == len(p.get("ugl", [])):
                    yield p


def split_of(p):
    """train / valid / test, fixed per passage (the same for both languages)."""
    h = int(hashlib.md5(("\n".join(p["english"])).encode("utf-8")).hexdigest()[:8], 16) % 100
    return "test" if h < 3 else "valid" if h < 5 else "train"


def english_words(p):
    return sum(len(s.split()) for s in p["english"])


class UGLTokens:
    """UGL tokenizer ids, remapped to the ids seen in training."""

    def __init__(self, vocab=None):
        import ugl
        self.tok = ugl.UGL(wsd=False).tokenizer
        self.vocab = vocab                        # original id -> model id
        self.inv = None

    def pieces(self, p):
        out = []
        for k, s in enumerate(p["ugl"]):
            if k:
                out.append(None)                  # sentence break
            for line in s.split("\n"):
                out += self.tok.encode(line, bos=False, eos=False)
        return out

    def fit(self, train):
        seen = {}
        for p in train:
            for i in self.pieces(p):
                if i is not None:
                    seen[i] = seen.get(i, 0) + 1
        self.vocab = {i: n + len(SPECIALS) for n, i in enumerate(sorted(seen))}
        return len(self.vocab) + len(SPECIALS)

    def ids(self, p):
        return [BRK if i is None else self.vocab.get(i, UNK) for i in self.pieces(p)] + [EOS]

    def text(self, ids):
        """model ids -> UGL text (one line per sentence)."""
        if self.inv is None:
            self.inv = {v: k for k, v in self.vocab.items()}
        lines, cur = [], []
        for i in ids:
            if i == EOS:
                break
            if i == BRK:
                lines.append(cur)
                cur = []
            elif i in self.inv:
                cur.append(self.inv[i])
        lines.append(cur)
        return [self.tok.decode(l) for l in lines if l]


class EnglishTokens:
    """Byte-level BPE trained on the training passages."""

    def __init__(self, path=None):
        from tokenizers import Tokenizer
        self.bpe = Tokenizer.from_file(path) if path else None

    def fit(self, train, size):
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
        self.bpe = Tokenizer(models.BPE())
        self.bpe.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        self.bpe.decoder = decoders.ByteLevel()
        tr = trainers.BpeTrainer(vocab_size=size, special_tokens=SPECIALS, show_progress=False,
                                 initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
        self.bpe.train_from_iterator((" ".join(p["english"]) for p in train), tr)
        return self.bpe.get_vocab_size()

    def ids(self, p):
        out = []
        for k, s in enumerate(p["english"]):
            if k:
                out.append(BRK)
            out += self.bpe.encode(s).ids
        return out + [EOS]

    def text(self, ids):
        sents, cur = [], []
        for i in ids:
            if i == EOS:
                break
            if i == BRK:
                sents.append(cur)
                cur = []
            else:
                cur.append(i)
        sents.append(cur)
        return [self.bpe.decode(s).strip() for s in sents if s]


# ------------------------------------------------------------------ model
class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        B, T, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, d))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, vocab, d=384, layers=6, heads=6, ctx=256):
        super().__init__()
        self.ctx = ctx
        self.tok, self.pos = nn.Embedding(vocab, d), nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight            # tied input / output embeddings
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, x):
        T = x.shape[1]
        h = self.tok(x) + self.pos(torch.arange(T, device=x.device))
        for b in self.blocks:
            h = b(h)
        return self.head(self.ln(h))

    @torch.no_grad()
    def generate(self, ids, n, temperature=0.8, top_k=40, stop=EOS):
        x = torch.tensor([ids], device=self.head.weight.device)
        for _ in range(n):
            logits = self(x[:, -self.ctx:])[0, -1] / temperature
            v, _ = logits.topk(top_k)
            logits[logits < v[-1]] = -float("inf")
            nxt = torch.multinomial(logits.softmax(-1), 1)
            x = torch.cat([x, nxt[None]], 1)
            if nxt.item() == stop:
                break
        return x[0, len(ids):].tolist()


# ------------------------------------------------------------------ training
def batches(stream, ctx, bs, rng):
    """Random windows of ctx+1 tokens from the packed training stream."""
    starts = rng.integers(0, len(stream) - ctx - 1, size=bs)
    x = np.stack([stream[s:s + ctx + 1] for s in starts])
    return torch.from_numpy(x[:, :-1]).long(), torch.from_numpy(x[:, 1:]).long()


@torch.no_grad()
def passage_nll(model, seqs, ctx, device, bs=32):
    """Total negative log-likelihood (nats) of each passage, predicted from its start."""
    model.eval()
    out = []
    for i in range(0, len(seqs), bs):
        part = [s[:ctx + 1] for s in seqs[i:i + bs]]      # (passages longer than ctx: scored to ctx)
        T = max(len(s) for s in part)
        x = torch.full((len(part), T), PAD, dtype=torch.long)
        for b, s in enumerate(part):
            x[b, :len(s)] = torch.tensor(s)
        x = x.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logits = model(x[:, :-1]).float()
        nll = F.cross_entropy(logits.transpose(1, 2), x[:, 1:], ignore_index=PAD, reduction="none").sum(1)
        out += nll.tolist()
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", help="folder of build folders (tools/build_corpus.py outputs)")
    ap.add_argument("--lang", choices=["ugl", "english"], required=True)
    ap.add_argument("--out", default=None, help="default data/build/lm/<lang>")
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--micro-bs", type=int, default=16,
                    help="sequences per forward pass; the batch's gradients are summed over these slices. "
                         "Same training as one pass of --bs, but the (sequences x positions x vocabulary) "
                         "output table is a quarter the size: a full one overfills an 8 GB GPU")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=float, default=3.0, help="passes over the training tokens")
    ap.add_argument("--steps", type=int, default=None, help="overrides --epochs")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    out = a.out or os.path.join(ROOT, "data", "build", "lm", a.lang)
    os.makedirs(out, exist_ok=True)
    device = a.device
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    log = open(os.path.join(out, "train.log"), "a", encoding="utf-8")

    def say(*x):
        msg = " ".join(str(y) for y in x)
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    t0 = time.time()
    data = {"train": [], "valid": [], "test": []}
    for p in passages(a.corpus):
        data[split_of(p)].append(p)
    say(f"passages: {', '.join(f'{k} {len(v)}' for k, v in data.items())}; "
        f"English words in train {sum(english_words(p) for p in data['train']):,}")
    # the vocabulary size comes from the UGL side either way, so both models match
    ugl_tok = UGLTokens()
    vocab = ugl_tok.fit(data["train"])
    if a.lang == "ugl":
        tok = ugl_tok
        with open(os.path.join(out, "vocab.json"), "w") as f:
            json.dump({str(k): v for k, v in ugl_tok.vocab.items()}, f)
    else:
        tok = EnglishTokens()
        vocab = tok.fit(data["train"], vocab)
        tok.bpe.save(os.path.join(out, "bpe.json"))
    seqs = {k: [tok.ids(p) for p in v] for k, v in data.items()}
    stream = np.fromiter((i for s in seqs["train"] for i in s), dtype=np.int32)
    say(f"{a.lang}: vocabulary {vocab:,}; train tokens {len(stream):,} "
        f"({len(stream) / max(1, sum(english_words(p) for p in data['train'])):.2f} per English word); "
        f"prepared in {time.time() - t0:.0f} s")

    model = GPT(vocab, a.d, a.layers, a.heads, a.ctx).to(device)
    params = sum(p.numel() for p in model.parameters())
    steps = a.steps or max(1, int(a.epochs * len(stream) / (a.bs * a.ctx)))
    say(f"model: {params / 1e6:.1f}M parameters (d {a.d}, {a.layers} layers, ctx {a.ctx}); {steps} steps")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1)
    warm = max(1, steps // 50)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps)))))
    valid_words = sum(english_words(p) for p in data["valid"])
    best = float("inf")
    cfg = {"lang": a.lang, "vocab": vocab, "d": a.d, "layers": a.layers, "heads": a.heads, "ctx": a.ctx}
    t1 = time.time()
    for step in range(1, steps + 1):
        x, y = batches(stream, a.ctx, a.bs, rng)
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        # the batch in slices: summed token losses / all counted tokens = the full batch's mean loss
        counted = (y != PAD).sum().clamp(min=1)
        loss = 0.0
        for s in range(0, a.bs, a.micro_bs):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                part = F.cross_entropy(model(x[s:s + a.micro_bs]).float().view(-1, vocab),
                                       y[s:s + a.micro_bs].reshape(-1), ignore_index=PAD, reduction="sum") / counted
            part.backward()
            loss = loss + part.detach()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % a.eval_every == 0 or step == steps:
            bpw = sum(passage_nll(model, seqs["valid"], a.ctx, device)) / math.log(2) / max(1, valid_words)
            rate = step * a.bs * a.ctx / (time.time() - t1)
            say(f"step {step}/{steps}: train loss {loss.item():.3f}; valid {bpw:.3f} bits per English word; "
                f"{rate / 1e3:.0f}k tokens/s")
            if bpw < best:
                best = bpw
                torch.save({"cfg": cfg, "model": model.state_dict()}, os.path.join(out, "model.pt"))
    # final: the best checkpoint on the test passages
    model.load_state_dict(torch.load(os.path.join(out, "model.pt"), map_location=device)["model"])
    nll = passage_nll(model, seqs["test"], a.ctx, device)
    test_words = sum(english_words(p) for p in data["test"])
    test_tokens = sum(min(len(s), a.ctx + 1) - 1 for s in seqs["test"])
    result = {"lang": a.lang, "params": params, "vocab": vocab, "train_tokens": int(len(stream)),
              "train_english_words": sum(english_words(p) for p in data["train"]),
              "test_bits_per_english_word": sum(nll) / math.log(2) / max(1, test_words),
              "test_loss_per_token": sum(nll) / max(1, test_tokens), "valid_best": best,
              "minutes": (time.time() - t0) / 60}
    say(json.dumps(result))
    with open(os.path.join(out, "result.json"), "w") as f:
        json.dump(result, f, indent=1)
    # samples: continue the first sentence of some test passages
    u = None
    if a.lang == "ugl":
        import ugl
        u = ugl.UGL(wsd=False)
    with open(os.path.join(out, "samples.txt"), "w", encoding="utf-8") as f:
        for p, s in list(zip(data["test"], seqs["test"]))[:20]:
            first = s[:s.index(BRK) + 1] if BRK in s else s[:-1]
            cont = tok.text(model.generate(first, 120))
            f.write(f"PROMPT  {p['english'][0]}\n")
            if u is not None:
                f.write("UGL     " + " / ".join(cont) + "\n")
                back = []
                for c in cont:                    # each sentence on its own: one bad line spoils no other
                    try:
                        back.append(u.decode(c))
                    except Exception:  # noqa: BLE001
                        back.append("<not valid UGL>")
                cont = back
            f.write("MODEL   " + " ".join(cont) + "\n\n")
    say("wrote", out)


if __name__ == "__main__":
    main()

"""The sentence-quality grader (trained by tools/train_quality.py): P(the UGL
of one corpus sentence is right).  Torch; used when building corpora, never
in the browser.

It reads
    <English source> [SEP] <the decoder's English for the UGL> [SEP] flags: ...
so it compares the two meanings with a pretrained English model and sees the
corpus builder's own checks.
"""
import json
import os
import sys

import torch
from torch import nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PATH = os.path.join(ROOT, "data", "build", "quality-grader")


def text_of(r):
    """(source, second segment) for a row with src, eng and kinds + roundtrip
    (or the issue texts, as tools/review_sample.py writes them)."""
    kinds = r.get("kinds")
    if kinds is None:
        sys.path.insert(0, os.path.join(ROOT, "tools"))
        from build_corpus import kind
        kinds = sorted({kind(i) for i in r["issues"]})
    flags = ", ".join(k for k in kinds if k != "round trip unstable") or "none"
    rt = r.get("roundtrip")
    if rt is None:
        rt = not any("round trip" in i for i in r["issues"])
    return r["src"], f"{r['eng']} [SEP] flags: {flags}; round trip {'stable' if rt else 'changed'}"


class Grader(nn.Module):
    def __init__(self, name):
        super().__init__()
        from transformers import AutoConfig, AutoModel
        if os.path.isdir(name):        # a saved grader: architecture only, weights come from grader.pt
            self.enc = AutoModel.from_config(AutoConfig.from_pretrained(name))
        else:                          # a pretrained encoder to fine-tune
            self.enc = AutoModel.from_pretrained(name)
        self.head = nn.Linear(self.enc.config.hidden_size, 1)

    def forward(self, **x):
        return self.head(self.enc(**x).last_hidden_state[:, 0]).squeeze(-1)


def score(model, tok, rows, device, bs=64):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), bs):
            a, b = zip(*[text_of(r) for r in rows[i:i + bs]])
            x = tok(list(a), list(b), truncation=True, max_length=256, padding=True, return_tensors="pt").to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                out += torch.sigmoid(model(**x).float()).tolist()
    return out


def save(model, tok, path, meta):
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(path, "grader.pt"))
    tok.save_pretrained(path)
    model.enc.config.save_pretrained(path)
    with open(os.path.join(path, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)


class QualityGrader:
    """grader = QualityGrader(); grader.score(rows) -> probabilities."""

    def __init__(self, path=DEFAULT_PATH):
        from transformers import AutoTokenizer
        with open(os.path.join(path, "meta.json")) as f:
            self.meta = json.load(f)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = Grader(path)
        self.model.load_state_dict(torch.load(os.path.join(path, "grader.pt"), map_location="cpu"))
        self.model.to(self.device)

    def score(self, rows):
        return score(self.model, self.tok, rows, self.device)

    def threshold(self, keep):
        """The score cut that kept this share of the held-out test sentences."""
        cuts = self.meta["cuts"]
        best = min(cuts, key=lambda c: abs(c["keep"] - keep))
        return best["score"]

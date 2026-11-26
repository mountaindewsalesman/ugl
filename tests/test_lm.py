"""ugl.lm: the numpy GPT-2 runner and the export from tools/train_lm.py."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from ugl.lm import GPT, read_safetensors, write_safetensors  # noqa: E402

CFG = {"n_layer": 2, "n_head": 2, "n_positions": 16, "n_embd": 8, "vocab_size": 11}


def random_weights(seed=0):
    r = np.random.default_rng(seed)
    d, V, P = CFG["n_embd"], CFG["vocab_size"], CFG["n_positions"]
    w = {"transformer.wte.weight": r.normal(0, .5, (V, d)), "transformer.wpe.weight": r.normal(0, .5, (P, d)),
         "transformer.ln_f.weight": 1 + r.normal(0, .1, d), "transformer.ln_f.bias": r.normal(0, .1, d)}
    for i in range(CFG["n_layer"]):
        p = f"transformer.h.{i}."
        for n, shape in (("attn.c_attn", (d, 3 * d)), ("attn.c_proj", (d, d)),
                         ("mlp.c_fc", (d, 4 * d)), ("mlp.c_proj", (4 * d, d))):
            w[p + n + ".weight"] = r.normal(0, .3, shape)
            w[p + n + ".bias"] = r.normal(0, .1, shape[1])
        for n in ("ln_1", "ln_2"):
            w[p + n + ".weight"] = 1 + r.normal(0, .1, d)
            w[p + n + ".bias"] = r.normal(0, .1, d)
    return {k: v.astype(np.float32) for k, v in w.items()}


def test_safetensors_round_trip(tmp_path):
    w = random_weights()
    w["half"] = w["transformer.wte.weight"].astype(np.float16)
    write_safetensors(tmp_path / "m.safetensors", w, {"format": "pt"})
    back = read_safetensors(tmp_path / "m.safetensors")
    assert set(back) == set(w)
    for k in w:
        assert np.allclose(back[k], w[k].astype(np.float32), atol=0), k


def test_cached_steps_match_full_forward():
    """Feeding tokens one at a time through the key/value cache gives the same
    logits as one pass over the whole sequence."""
    g = GPT(random_weights(), CFG)
    ids = [1, 5, 7, 2, 9, 4, 4, 3]
    full = g.forward(ids, [])
    cache, step = [], []
    step.append(g.forward(ids[:3], cache))
    for i in ids[3:]:
        step.append(g.forward([i], cache))
    assert np.allclose(np.concatenate(step), full, atol=1e-5)


def test_generation_past_the_context_window():
    """Generating more tokens than the context holds keeps going (the window
    restarts on its second half) and returns exactly the tokens generated."""
    g = GPT(random_weights(), CFG)
    out = g.generate([1, 5], 40, temperature=1.0, rng=np.random.default_rng(0), stop=())
    assert len(out) == 40
    assert g.generate([1, 5], 5, temperature=0, stop=()) == g.generate([1, 5], 5, temperature=0, stop=())


def test_export_matches_torch(tmp_path):
    """A train_lm.GPT exported by tools/export_lm.py predicts the same logits
    in ugl.lm (numpy) as in PyTorch."""
    torch = pytest.importorskip("torch")
    from train_lm import GPT as TorchGPT
    from export_lm import hf_tensors
    torch.manual_seed(0)
    m = TorchGPT(11, d=8, layers=2, heads=2, ctx=16).eval()
    for p in m.parameters():                       # beyond init scale, so a transposed weight shows
        p.data.normal_(0, 0.3)
    write_safetensors(tmp_path / "m.safetensors", hf_tensors(m.state_dict(), 2, np.float32))
    g = GPT(read_safetensors(tmp_path / "m.safetensors"), CFG)
    ids = [1, 4, 6, 2, 8, 10, 3]
    with torch.no_grad():
        ref = m(torch.tensor([ids]))[0].numpy()
    assert np.allclose(g.forward(ids, []), ref, atol=1e-4)

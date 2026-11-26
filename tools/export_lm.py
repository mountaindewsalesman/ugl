"""Package a model trained by tools/train_lm.py as a standard GPT-2 folder.

    python tools/export_lm.py data/build/lm/ugl dist/ugl-stories
    python tools/export_lm.py data/build/lm/english dist/english-stories --check

Writes config.json, model.safetensors and tokenizer.json in the Hugging Face
GPT-2 layout (see ugl/lm.py), plus a short README.md.  The same folder:
  * runs with ugl.lm.load(folder) (numpy only),
  * loads in transformers: GPT2LMHeadModel.from_pretrained(folder),
  * converts with llama.cpp's convert_hf_to_gguf.py (English model) or
    optimum's ONNX export for the browser.

--check compares next-token logits from the trained PyTorch model, the numpy
runner in ugl/lm.py and (if installed) transformers, on held-out text.
Publish with:  gh release create lm-<name> <folder>/* --title "..."
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from ugl.lm import EOS, BRK, PAD, UNK, write_safetensors  # noqa: E402
from train_lm import GPT, SPECIALS  # noqa: E402


def hf_tensors(state, layers, dtype):
    """train_lm.GPT state dict -> Hugging Face GPT-2 tensor names.
    nn.Linear stores (out, in); GPT-2's Conv1D stores (in, out)."""
    t = {"transformer.wte.weight": state["tok.weight"], "transformer.wpe.weight": state["pos.weight"],
         "transformer.ln_f.weight": state["ln.weight"], "transformer.ln_f.bias": state["ln.bias"]}
    for i in range(layers):
        a, b = f"blocks.{i}.", f"transformer.h.{i}."
        for ours, theirs in (("ln1", "ln_1"), ("ln2", "ln_2")):
            t[b + theirs + ".weight"] = state[a + ours + ".weight"]
            t[b + theirs + ".bias"] = state[a + ours + ".bias"]
        for ours, theirs in (("qkv", "attn.c_attn"), ("proj", "attn.c_proj"),
                             ("mlp.0", "mlp.c_fc"), ("mlp.2", "mlp.c_proj")):
            t[b + theirs + ".weight"] = state[a + ours + ".weight"].T
            t[b + theirs + ".bias"] = state[a + ours + ".bias"]
    return {k: v.float().cpu().numpy().astype(dtype) for k, v in t.items()}


def ugl_tokenizer_json(vocab, tok):
    """A Hugging Face WordLevel tokenizer whose entries are UGL token texts
    (n988, +s, \\v20, literal pieces ...) at their model ids."""
    entries = {s: i for i, s in enumerate(SPECIALS)}
    for orig, model_id in vocab.items():
        entries[tok.vocab[int(orig)]] = model_id
    return {"version": "1.0", "truncation": None, "padding": None,
            "added_tokens": [{"id": i, "content": s, "single_word": False, "lstrip": False, "rstrip": False,
                              "normalized": False, "special": True} for i, s in enumerate(SPECIALS)],
            "normalizer": None, "pre_tokenizer": {"type": "WhitespaceSplit"}, "post_processor": None,
            "decoder": None, "model": {"type": "WordLevel", "vocab": entries, "unk_token": "<unk>"}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trained", help="tools/train_lm.py output folder (model.pt + vocab.json or bpe.json)")
    ap.add_argument("out")
    ap.add_argument("--dtype", choices=["float16", "float32"], default="float16",
                    help="float16 halves the download; generation is the same to within rounding")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    ck = torch.load(os.path.join(a.trained, "model.pt"), map_location="cpu")
    cfg = ck["cfg"]
    os.makedirs(a.out, exist_ok=True)
    result = {}
    if os.path.exists(os.path.join(a.trained, "result.json")):
        with open(os.path.join(a.trained, "result.json")) as f:
            result = json.load(f)
    info = {"format": 1, "lang": cfg["lang"], "break_token_id": BRK, "unk_token_id": UNK,
            "trained_english_words": result.get("train_english_words"),
            "test_bits_per_english_word": result.get("test_bits_per_english_word")}
    if cfg["lang"] == "ugl":
        import ugl
        tok = ugl.UGL(wsd=False).tokenizer
        with open(os.path.join(a.trained, "vocab.json")) as f:
            vocab = json.load(f)
        info.update(dictionary=f"ugl-dict-{tok.version}.json.gz", tokenizer_scheme=tok.scheme)
        with open(os.path.join(a.out, "tokenizer.json"), "w", encoding="utf-8") as f:
            json.dump(ugl_tokenizer_json(vocab, tok), f, ensure_ascii=False)
    else:
        shutil.copy(os.path.join(a.trained, "bpe.json"), os.path.join(a.out, "tokenizer.json"))
    config = {"model_type": "gpt2", "architectures": ["GPT2LMHeadModel"],
              "vocab_size": cfg["vocab"], "n_embd": cfg["d"], "n_layer": cfg["layers"], "n_head": cfg["heads"],
              "n_positions": cfg["ctx"], "n_ctx": cfg["ctx"], "n_inner": 4 * cfg["d"],
              "activation_function": "gelu",          # exact GELU, as nn.GELU()
              "layer_norm_epsilon": 1e-5, "tie_word_embeddings": True,
              "resid_pdrop": 0.0, "embd_pdrop": 0.0, "attn_pdrop": 0.0,
              "bos_token_id": EOS, "eos_token_id": EOS, "pad_token_id": PAD, "ugl": info}
    with open(os.path.join(a.out, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=1)
    write_safetensors(os.path.join(a.out, "model.safetensors"),
                      hf_tensors(ck["model"], cfg["layers"], np.dtype(a.dtype)), {"format": "pt"})
    params = sum(v.numel() for k, v in ck["model"].items() if k != "head.weight")
    with open(os.path.join(a.out, "README.md"), "w", encoding="utf-8") as f:
        f.write(f"# UGL project story model ({cfg['lang']})\n\n"
                f"GPT-2 architecture, {params / 1e6:.1f}M parameters, context {cfg['ctx']} tokens, "
                f"trained by tools/train_lm.py in https://github.com/mountaindewsalesman/ugl "
                f"on {info['trained_english_words'] or '?'} English words of paired text"
                f"{' (as UGL)' if cfg['lang'] == 'ugl' else ''}.\n\n"
                "```python\nimport ugl.lm\nm = ugl.lm.load(\"path/to/this/folder\")\n"
                "print(m.continue_text(\"Once upon a time, there was a little dog.\"))\n```\n")
    print("wrote", a.out, f"({params / 1e6:.1f}M parameters, {a.dtype})")
    if a.check:
        check(ck, a.out)


def check(ck, folder):
    """Same next-token predictions from PyTorch, ugl.lm (numpy) and transformers."""
    import ugl.lm
    cfg = ck["cfg"]
    m = GPT(cfg["vocab"], cfg["d"], cfg["layers"], cfg["heads"], cfg["ctx"])
    m.load_state_dict(ck["model"])
    m.eval()
    lm = ugl.lm.LM(folder)
    ids = lm.prompt_ids("Once upon a time, there was a little girl named Lily. She loved to play outside.")
    with torch.no_grad():
        ref = m(torch.tensor([ids]))[0].numpy()
    got = lm.model.forward(ids, [])
    print(f"numpy vs torch: max logit difference {np.abs(got - ref).max():.4f}; "
          f"same top token at {np.mean(got.argmax(-1) == ref.argmax(-1)):.0%} of {len(ids)} positions")
    try:
        from transformers import GPT2LMHeadModel
    except ImportError:
        return
    hf = GPT2LMHeadModel.from_pretrained(folder).float().eval()
    with torch.no_grad():
        out = hf(torch.tensor([ids])).logits[0].numpy()
    print(f"transformers vs torch: max logit difference {np.abs(out - ref).max():.4f}; "
          f"same top token at {np.mean(out.argmax(-1) == ref.argmax(-1)):.0%}")
    print("sample:", lm.continue_text("Once upon a time, there was a little girl named Lily.", sentences=2, seed=0))


if __name__ == "__main__":
    main()

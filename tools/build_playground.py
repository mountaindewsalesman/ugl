"""Build the browser playground (GitHub Pages site) from the current repo.

    python tools/build_playground.py            # -> _site/
    python tools/build_playground.py --out DIR
    python -m http.server -d _site 8000         # then open http://localhost:8000
    python tools/build_playground.py --compare smoke.json   # see tools/playground_smoke.js

What it produces:
  * the static files from site/
  * ugl-py.zip      the ugl package sources + the default dictionary, unpacked
                    into Pyodide so the browser runs this repo's own decoder,
                    tokenizer and lexicon
  * wheels/         the lemminflect wheel the decoder imports (pure Python)
  * gallery.json    the English examples in site/gallery.txt, encoded by the
                    REAL encoder (spaCy en_core_web_md) at build time
  * manifest.json   versions and file names the page loads

spaCy cannot run in Pyodide (no WebAssembly builds of spaCy, thinc, blis,
cymem, preshed or murmurhash exist), so English -> UGL happens here, at
deploy time, and the page decodes/tokenizes/explains live.
"""
import argparse
import datetime
import importlib.metadata as md
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import ugl  # noqa: E402
from ugl.lexicon import DICT_DIR, default_path  # noqa: E402

SITE = os.path.join(ROOT, "site")
SKIP = {"gallery.txt"}          # build inputs, not served


def read_gallery():
    with open(os.path.join(SITE, "gallery.txt"), encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip() and not l.lstrip().startswith("#")]


def build_bundle(path):
    """ugl/*.py + dictionary/default.txt + the default dictionary, as a zip
    whose top-level folder is `ugl/` (importable once unpacked on sys.path)."""
    pkg = os.path.join(ROOT, "ugl")
    dict_file = default_path()
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name in sorted(os.listdir(pkg)):
            if name.endswith(".py"):
                z.write(os.path.join(pkg, name), f"ugl/{name}")
        z.write(os.path.join(DICT_DIR, "default.txt"), "ugl/dictionary/default.txt")
        # already gzipped: store it rather than compressing twice
        z.write(dict_file, f"ugl/dictionary/{os.path.basename(dict_file)}",
                compress_type=zipfile.ZIP_STORED)


def fetch_wheel(name, out_dir):
    """Download the exact installed version of a pure-Python dependency."""
    version = md.version(name)
    os.makedirs(out_dir, exist_ok=True)
    subprocess.run([sys.executable, "-m", "pip", "download", f"{name}=={version}", "--no-deps",
                    "--only-binary", ":all:", "--dest", out_dir, "--quiet",
                    "--disable-pip-version-check"], check=True)
    wheels = [f for f in os.listdir(out_dir)
              if f.lower().startswith(name.lower() + "-" + version) and f.endswith(".whl")]
    if len(wheels) != 1:
        sys.exit(f"expected one {name} {version} wheel in {out_dir}, found {wheels}")
    if not wheels[0].endswith("-none-any.whl"):
        sys.exit(f"{wheels[0]} is not a pure-Python wheel; Pyodide cannot install it")
    return wheels[0]


def encode_gallery(sentences):
    u = ugl.UGL(wsd=False)      # most-frequent-sense, same as a plain `pip install`
    items = []
    for en in sentences:
        items.append({"en": en, "ugl": u.encode(en)})
        print(f"  {en}\n    -> " + "\n    -> ".join(items[-1]["ugl"]))
    nlp = u.encoder.nlp
    model = f"{nlp.meta['lang']}_{nlp.meta['name']}-{nlp.meta['version']}"
    return items, model


CHECK = r"""
import json, os, sys
lib, site = sys.argv[1], sys.argv[2]
sys.path[:0] = [lib, site]
import ugl, playground
assert os.path.dirname(ugl.__file__) == os.path.join(lib, "ugl"), ugl.__file__
playground.init()
gallery = json.load(open(os.path.join(site, "gallery.json"), encoding="utf-8"))
print(json.dumps([json.loads(playground.analyze("\n".join(g["ugl"]))) for g in gallery]))
"""


def site_results(out):
    """Run the built site's own files (bundle + playground.py) under an
    isolated CPython, the same way the worker runs them in Pyodide."""
    with tempfile.TemporaryDirectory() as lib:
        with zipfile.ZipFile(os.path.join(out, "ugl-py.zip")) as z:
            z.extractall(lib)
        r = subprocess.run([sys.executable, "-I", "-B", "-c", CHECK, lib, out],
                           capture_output=True, text=True, encoding="utf-8")
    if r.returncode:
        sys.exit("site check failed:\n" + r.stderr)
    return json.loads(r.stdout)


def check(out, items):
    """Every gallery entry must tokenize, decode and round-trip its ids."""
    problems = []
    for g, res in zip(items, site_results(out)):
        for s in res["sentences"]:
            if "error" in s or "back_error" in s or not s["roundtrip_ok"]:
                problems.append(f"{g['en']!r}: {s}")
    for p in problems:
        print("  PROBLEM", p)
    print(f"site check: {len(items)} gallery entries decoded by the bundled package, "
          f"{len(problems)} problems")
    return not problems


def compare(out, smoke_path):
    """Pyodide results (from tools/playground_smoke.js) == CPython results."""
    with open(smoke_path, encoding="utf-8") as f:
        smoke = json.load(f)["gallery"]
    ref = site_results(out)
    diffs = 0
    for g, r in zip(smoke, ref):
        if g["sentences"] != r["sentences"] or g["ids"] != r["ids"]:
            diffs += 1
            print("  DIFF", g["en"])
    print(f"compare: {len(smoke)} entries, {diffs} differ between Pyodide and CPython")
    return diffs == 0 and len(smoke) == len(ref)


def git_commit():
    sha = os.environ.get("GITHUB_SHA")
    if sha:
        return sha
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(ROOT, "_site"))
    ap.add_argument("--compare", metavar="SMOKE_JSON",
                    help="don't build; compare playground_smoke.js output with CPython")
    a = ap.parse_args(argv)
    out = os.path.abspath(a.out)
    if a.compare:
        sys.exit(0 if compare(out, a.compare) else 1)

    if os.path.exists(out):
        shutil.rmtree(out)
    shutil.copytree(SITE, out, ignore=lambda d, names: [n for n in names if n in SKIP or n == "__pycache__"])

    print("bundling ugl sources + dictionary")
    build_bundle(os.path.join(out, "ugl-py.zip"))
    print("downloading lemminflect wheel")
    wheel = fetch_wheel("lemminflect", os.path.join(out, "wheels"))

    print("encoding gallery with the real encoder")
    items, model = encode_gallery(read_gallery())
    with open(os.path.join(out, "gallery.json"), "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=1)

    commit = git_commit()
    manifest = {
        "ugl_version": ugl.__version__,
        "dictionary": os.path.basename(default_path()),
        "bundle": "ugl-py.zip",
        "bundle_bytes": os.path.getsize(os.path.join(out, "ugl-py.zip")),
        "wheels": [f"wheels/{wheel}"],
        "gallery": {"file": "gallery.json", "spacy": md.version("spacy"), "model": model,
                    "sense_choice": "most frequent sense"},
        "commit": commit,
        "built": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    print(f"built {out}  ({len(items)} gallery entries, model {model})")
    if not check(out, items):
        sys.exit("site check failed")


if __name__ == "__main__":
    main()

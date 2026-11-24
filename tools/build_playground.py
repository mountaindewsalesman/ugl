"""Build the browser playground (GitHub Pages site) from the current repo.

    python tools/build_playground.py            # -> _site/
    python tools/build_playground.py --out DIR
    python -m http.server -d _site 8000         # then open http://localhost:8000
    python tools/build_playground.py --compare smoke.json   # see tools/playground_smoke.mjs

What it produces:
  * the static files from site/
  * ugl-py.zip      the ugl package sources + the default dictionary, unpacked
                    into Pyodide so the browser runs this repo's own decoder,
                    tokenizer and lexicon
  * wheels/         the lemminflect wheel the decoder imports (pure Python)
  * pyodide/        the Pyodide runtime + numpy, served from the same origin
                    (browsers refuse cross-origin module loads inside workers,
                    and the site then has no third-party runtime dependency)
  * gallery.json    the English examples in site/gallery.txt with their UGL from
                    CPython; the page encodes them again live in Pyodide and the
                    tests require both to be identical
  * manifest.json   versions and file names the page loads

The encoder runs UGL's own numpy parser, so English -> UGL, decoding,
tokenizing and explanations all run live in the browser.
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
    """The ugl package (all .py files, the parser weights) + the default
    dictionary, as a zip whose top-level folder is `ugl/`."""
    pkg = os.path.join(ROOT, "ugl")
    dict_file = default_path()
    from ugl.parser.model import DEFAULT_PATH as PARSER
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for dirpath, dirs, files in os.walk(pkg):
            dirs[:] = [d for d in dirs if d not in ("__pycache__", "models")]
            for name in sorted(files):
                if name.endswith(".py"):
                    full = os.path.join(dirpath, name)
                    z.write(full, "ugl/" + os.path.relpath(full, pkg).replace(os.sep, "/"))
        z.write(PARSER, "ugl/" + os.path.relpath(PARSER, pkg).replace(os.sep, "/"),
                compress_type=zipfile.ZIP_STORED)
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


PYODIDE_CORE = ["pyodide.js", "pyodide.mjs", "pyodide.asm.mjs", "pyodide.asm.wasm",
                "python_stdlib.zip", "pyodide-lock.json"]


def pyodide_version():
    """The single source of truth is the constant in site/worker.js."""
    import re
    with open(os.path.join(SITE, "worker.js"), encoding="utf-8") as f:
        m = re.search(r'PYODIDE_VERSION = "([^"]+)"', f.read())
    if not m:
        sys.exit("PYODIDE_VERSION not found in site/worker.js")
    return m.group(1)


def fetch_pyodide(version, out_dir, packages=("numpy",)):
    """Download the Pyodide runtime and the named packages (+ dependencies)."""
    import urllib.request
    base = f"https://cdn.jsdelivr.net/pyodide/v{version}/full/"
    os.makedirs(out_dir, exist_ok=True)

    def get(name):
        dst = os.path.join(out_dir, name)
        with urllib.request.urlopen(base + name, timeout=120) as r, open(dst, "wb") as f:
            shutil.copyfileobj(r, f)
        return dst

    for name in PYODIDE_CORE:
        get(name)
    with open(os.path.join(out_dir, "pyodide-lock.json"), encoding="utf-8") as f:
        lock = json.load(f)["packages"]
    todo, seen = list(packages), set()
    while todo:
        p = todo.pop()
        if p in seen:
            continue
        seen.add(p)
        get(lock[p]["file_name"])
        todo += lock[p].get("depends", [])
    return sorted(seen)


def encode_gallery(sentences):
    u = ugl.UGL(wsd=False)      # most-frequent-sense: the browser has no BERT sense model
    items = []
    for en in sentences:
        items.append({"en": en, "ugl": u.encode(en)})
        print(f"  {en}\n    -> " + "\n    -> ".join(items[-1]["ugl"]))
    meta = u.encoder.parser.meta
    model = f"{os.path.basename(u.encoder.parser.path)} ({', '.join(meta.get('treebanks', []))})"
    return items, model


CHECK = r"""
import json, os, sys
lib, site = sys.argv[1], sys.argv[2]
sys.path[:0] = [lib, site]
import ugl, playground
assert os.path.dirname(ugl.__file__) == os.path.join(lib, "ugl"), ugl.__file__
playground.init()
gallery = json.load(open(os.path.join(site, "gallery.json"), encoding="utf-8"))
out = []
for g in gallery:
    enc = json.loads(playground.encode(g["en"]))
    assert enc["ugl"] == g["ugl"], (g["en"], enc["ugl"], g["ugl"])
    out.append(json.loads(playground.analyze("\n".join(g["ugl"]))))
print(json.dumps(out))
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
    """Pyodide results (from tools/playground_smoke.mjs) == CPython results."""
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
                    help="don't build; compare playground_smoke.mjs output with CPython")
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

    pv = pyodide_version()
    print(f"downloading Pyodide {pv} runtime + numpy")
    py_pkgs = fetch_pyodide(pv, os.path.join(out, "pyodide"))

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
        "pyodide": {"version": pv, "path": "pyodide/", "packages": py_pkgs},
        "gallery": {"file": "gallery.json", "parser": model, "sense_choice": "most frequent sense"},
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

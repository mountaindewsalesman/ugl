"""Mass-download plain English text from Wikipedia into data/raw/.

Two modes
  api   (default)  random articles through the MediaWiki API, one plain-text
                   article per request.  Good for a few thousand articles.
        python scripts/fetch_wikipedia.py api --count 500

  dump             streams an official Wikipedia XML dump over HTTP,
                   decompressing on the fly, and stops after --count articles
                   (nothing big is saved to disk).  Good for 10k-1M articles.
        python scripts/fetch_wikipedia.py dump --count 20000

Output: one .jsonl file per batch of articles, one article per line:
    {"title": "Polystyrene", "text": "paragraphs separated by newlines"}
(read with ugl.corpus.documents; titles give coreference its document topic).
"""
import argparse
import bz2
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "raw")
API = "https://en.wikipedia.org/w/api.php"
DUMP = "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles-multistream.xml.bz2"
HEADERS = {"User-Agent": "ugl-corpus-fetcher/0.1 (research text sampler; python-requests)"}

SKIP_SECTIONS = re.compile(r"^==+\s*(References|External links|See also|Notes|Further reading|Bibliography|Sources|Citations)\s*==+", re.M)


def clean_extract(text):
    text = SKIP_SECTIONS.split(text)[0]
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("=="):
            continue
        if len(line) < 40:          # headings, list stubs, captions
            continue
        lines.append(line)
    return "\n".join(lines)


class Writer:
    def __init__(self, prefix, per_file=200):
        os.makedirs(OUT, exist_ok=True)
        self.prefix, self.per_file = prefix, per_file
        self.buf, self.n, self.files, self.chars = [], 0, 0, 0
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.stamp = stamp

    def add(self, title, text):
        if len(text) < 200:
            return False
        self.buf.append(json.dumps({"title": title, "text": text}, ensure_ascii=False))
        self.n += 1
        self.chars += len(text)
        if len(self.buf) >= self.per_file:
            self.flush()
        return True

    def flush(self):
        if not self.buf:
            return
        fp = os.path.join(OUT, f"{self.prefix}-{self.stamp}-{self.files:04d}.jsonl")
        with open(fp, "w", encoding="utf-8") as f:
            f.write("\n".join(self.buf) + "\n")
        self.files += 1
        self.buf = []


def fetch_api(count, delay):
    s = requests.Session()
    s.headers.update(HEADERS)
    w = Writer("api")
    params = {"action": "query", "format": "json", "generator": "random", "grnnamespace": 0,
              "grnlimit": 1, "prop": "extracts", "explaintext": 1, "exsectionformat": "wiki",
              "maxlag": 5}
    t0 = time.time()
    while w.n < count:
        try:
            r = s.get(API, params=params, timeout=30)
            if r.status_code == 429 or "maxlag" in r.text[:200]:
                time.sleep(5)
                continue
            r.raise_for_status()
            for page in r.json().get("query", {}).get("pages", {}).values():
                w.add(page.get("title", ""), clean_extract(page.get("extract", "")))
        except Exception as e:  # network hiccup: back off and continue
            print("  error:", e, file=sys.stderr)
            time.sleep(5)
        if w.n % 25 == 0:
            print(f"  {w.n}/{count} articles, {w.chars/1e6:.1f}M chars, {time.time()-t0:.0f}s", file=sys.stderr)
        time.sleep(delay)
    w.flush()
    return w


# --- crude but fast wikitext -> plain text (used in dump mode)
_RE = [
    (re.compile(r"<!--.*?-->", re.S), ""),
    (re.compile(r"<ref[^>]*/>"), ""),
    (re.compile(r"<ref.*?</ref>", re.S), ""),
    (re.compile(r"<[^>]+>"), ""),
    (re.compile(r"\[\[(?:File|Image|Category):[^\]]*\]\]", re.I), ""),
    (re.compile(r"\[\[[^\]|]*\|([^\]]*)\]\]"), r"\1"),
    (re.compile(r"\[\[([^\]]*)\]\]"), r"\1"),
    (re.compile(r"\[https?://[^\s\]]+\s?([^\]]*)\]"), r"\1"),
    (re.compile(r"'{2,}"), ""),
    (re.compile(r"&nbsp;"), " "),
]


def strip_templates(t):
    out, depth, i = [], 0, 0
    while i < len(t):
        if t.startswith("{{", i) or t.startswith("{|", i):
            depth += 1; i += 2; continue
        if depth and (t.startswith("}}", i) or t.startswith("|}", i)):
            depth -= 1; i += 2; continue
        if not depth:
            out.append(t[i])
        i += 1
    return "".join(out)


def wikitext_to_text(t):
    t = strip_templates(t)
    for rx, rep in _RE:
        t = rx.sub(rep, t)
    lines = []
    for line in t.split("\n"):
        line = line.strip()
        if not line or line[0] in "=*#:;|!{":
            continue
        if len(line) < 40:
            continue
        lines.append(line)
    return clean_extract("\n".join(lines))


def fetch_dump(count, url):
    w = Writer("dump")
    r = requests.get(url, stream=True, headers=HEADERS, timeout=60)
    r.raise_for_status()

    class Stream:
        def __init__(self, it):
            self.it, self.dec, self.buf = it, bz2.BZ2Decompressor(), b""

        def read(self, n=-1):
            while len(self.buf) < max(n, 1 << 16):
                try:
                    chunk = next(self.it)
                except StopIteration:
                    break
                while chunk:
                    if self.dec.eof:   # multistream: start a new decompressor
                        rest = self.dec.unused_data
                        self.dec = bz2.BZ2Decompressor()
                        chunk = rest + chunk if rest else chunk
                    try:
                        self.buf += self.dec.decompress(chunk)
                        chunk = b""
                    except EOFError:
                        break
            out, self.buf = (self.buf, b"") if n < 0 else (self.buf[:n], self.buf[n:])
            return out

    t0 = time.time()
    ns = None
    for ev, el in ET.iterparse(Stream(r.iter_content(1 << 20)), events=("end",)):
        tag = el.tag.rsplit("}", 1)[-1]
        if tag != "page":
            continue
        if ns is None:
            ns = el.tag[: el.tag.index("}") + 1] if "}" in el.tag else ""
        title = el.findtext(f"{ns}title") or ""
        pns = el.findtext(f"{ns}ns")
        text = el.findtext(f"{ns}revision/{ns}text") or ""
        el.clear()
        if pns != "0" or text.lower().startswith("#redirect"):
            continue
        if w.add(title, wikitext_to_text(text)) and w.n % 500 == 0:
            print(f"  {w.n}/{count} articles, {w.chars/1e6:.1f}M chars, {time.time()-t0:.0f}s", file=sys.stderr)
        if w.n >= count:
            break
    w.flush()
    r.close()
    return w


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode")
    a = sub.add_parser("api"); a.add_argument("--count", type=int, default=200); a.add_argument("--delay", type=float, default=0.2)
    d = sub.add_parser("dump"); d.add_argument("--count", type=int, default=5000); d.add_argument("--url", default=DUMP)
    args = ap.parse_args()
    if args.mode == "dump":
        w = fetch_dump(args.count, args.url)
    else:
        w = fetch_api(getattr(args, "count", 200), getattr(args, "delay", 0.2))
    print(f"saved {w.n} articles ({w.chars/1e6:.1f}M chars) in {w.files} files under {OUT}")


if __name__ == "__main__":
    main()

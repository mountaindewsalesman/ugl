"""List EWT test sentences whose GOLD-tree encoding crashes or fails to decode."""
import os
import random
import sys
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8")

import ugl  # noqa: E402
from ugl.analyze import Analyzer  # noqa: E402
from ugl.coref import resolve  # noqa: E402
from ugl.parser.conllu import read_conllu  # noqa: E402
from ugl.serialize import Serializer  # noqa: E402

n = int(sys.argv[1]) if len(sys.argv) > 1 else 400
u = ugl.load(wsd=False)
sents = list(read_conllu(os.path.join(ROOT, "data", "ud", "en_ewt-ud-test.conllu")))
random.Random(0).shuffle(sents)
sents = [s for s in sents if 4 <= len(s) <= 30][:n]
for s in sents:
    words = [t["form"] for t in s]
    try:
        ev = Analyzer(u.lexicon).sentence(s, question=words[-1] == "?")
        resolve([ev])
        out = Serializer(u.lexicon).sentence(ev, words)
    except Exception:
        print("CRASH:", " ".join(words))
        print("   ", traceback.format_exc().strip().splitlines()[-3:])
        continue
    try:
        u.decode(out)
    except Exception as e:
        print("DECODE:", " ".join(words))
        print("    ugl:", out)
        print("    err:", e)

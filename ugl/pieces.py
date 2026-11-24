"""Subword pieces for literals (rare names, unknown words, numbers).

Learned with byte-pair encoding from the corpus strings that stay literals,
stored in the dictionary (`literal_pieces`), and applied by greedy
longest-match, so a literal costs a few ids instead of one per character:

    'Kuttiyeri'  ->  Kutt iy eri        (3 ids instead of 9)
"""
import collections
import heapq


def learn(strings, n_pieces=4000, min_freq=3):
    """strings: {text: count}.  Returns pieces (multi-character), most useful first."""
    words = []          # list of [symbols], count
    for text, c in strings.items():
        if text:
            words.append([list(text), c])
    pairs = collections.Counter()
    where = collections.defaultdict(set)
    for wi, (sym, c) in enumerate(words):
        for a, b in zip(sym, sym[1:]):
            pairs[(a, b)] += c
            where[(a, b)].add(wi)
    heap = [(-c, p) for p, c in pairs.items()]
    heapq.heapify(heap)
    pieces = []
    while heap and len(pieces) < n_pieces:
        negc, pair = heapq.heappop(heap)
        if pairs.get(pair, 0) != -negc:
            continue                                   # stale entry
        if -negc < min_freq:
            break
        a, b = pair
        merged = a + b
        pieces.append(merged)
        touched = collections.Counter()
        for wi in list(where[pair]):
            sym, c = words[wi]
            i, out = 0, []
            changed = False
            while i < len(sym):
                if i + 1 < len(sym) and sym[i] == a and sym[i + 1] == b:
                    out.append(merged)
                    i += 2
                    changed = True
                else:
                    out.append(sym[i])
                    i += 1
            if not changed:
                continue
            for x, y in zip(sym, sym[1:]):              # remove old pairs of this word
                pairs[(x, y)] -= c
                touched[(x, y)] += 0
            for x, y in zip(out, out[1:]):              # add new pairs
                pairs[(x, y)] += c
                where[(x, y)].add(wi)
                touched[(x, y)] += 0
            words[wi][0] = out
        pairs.pop(pair, None)
        for p in touched:
            if pairs.get(p, 0) > 0:
                heapq.heappush(heap, (-pairs[p], p))
    return pieces


class Segmenter:
    """Greedy longest-match over a fixed piece inventory (falls back to characters)."""

    def __init__(self, pieces):
        self.pieces = set(pieces)
        self.maxlen = max((len(p) for p in pieces), default=1)

    def __call__(self, text):
        out, i = [], 0
        while i < len(text):
            for L in range(min(self.maxlen, len(text) - i), 1, -1):
                if text[i:i + L] in self.pieces:
                    out.append(text[i:i + L])
                    i += L
                    break
            else:
                out.append(text[i])
                i += 1
        return out

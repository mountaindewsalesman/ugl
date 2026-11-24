"""Maximum spanning arborescence (Chu-Liu/Edmonds) with a single root.

scores[i, j] = score of word j being the head of word i; index 0 is ROOT.
Returns heads[i] for i in 0..n (heads[0] = -1).
"""
import numpy as np


def _cle(scores):
    n = scores.shape[0]
    s = scores.copy()
    np.fill_diagonal(s, -np.inf)
    s[0, :] = -np.inf                       # ROOT has no head
    heads = s.argmax(axis=1)
    heads[0] = -1
    cycle = _find_cycle(heads)
    if cycle is None:
        return heads
    # contract the cycle into one node and solve recursively
    cyc = sorted(cycle)
    cset = set(cyc)
    rest = [v for v in range(n) if v not in cset]       # includes ROOT (index 0)
    m = len(rest)
    new = np.full((m + 1, m + 1), -np.inf)
    idx = {v: k for k, v in enumerate(rest)}
    c = m                                               # index of contracted node
    cyc_score = sum(s[v, heads[v]] for v in cyc)
    enter = {}   # outside head u -> (best dep in cycle)
    leave = {}   # outside dep v -> (best head in cycle)
    for u in rest:
        for v in rest:
            if u != v:
                new[idx[v], idx[u]] = s[v, u]
        # arc u -> cycle: replace the cycle arc into the entered node
        best, bv = -np.inf, None
        for v in cyc:
            val = s[v, u] - s[v, heads[v]] + cyc_score
            if val > best:
                best, bv = val, v
        new[c, idx[u]] = best
        enter[u] = bv
        # arc cycle -> u
        best, bh = -np.inf, None
        for h in cyc:
            if s[u, h] > best:
                best, bh = s[u, h], h
        new[idx[u], c] = best
        leave[u] = bh
    sub = _cle(new)
    out = np.full(n, -1)
    for v in rest:
        if v == 0:
            continue
        h = sub[idx[v]]
        out[v] = leave[v] if h == c else rest[h]
    for v in cyc:
        out[v] = heads[v]
    u = rest[sub[c]]
    out[enter[u]] = u
    out[0] = -1
    return out


def _find_cycle(heads):
    n = len(heads)
    color = [0] * n
    for start in range(1, n):
        path, v = [], start
        while v > 0 and color[v] == 0:
            color[v] = 1
            path.append(v)
            v = heads[v]
        if v > 0 and color[v] == 1:
            return path[path.index(v):]
        for p in path:
            color[p] = 2
    return None


def chu_liu_edmonds(scores):
    """Best tree in which exactly one word attaches to ROOT."""
    scores = np.asarray(scores, dtype=np.float64)
    heads = _cle(scores)
    roots = [i for i in range(1, len(heads)) if heads[i] == 0]
    if len(roots) <= 1:
        return heads
    best, best_heads = -np.inf, heads
    for r in roots:
        s = scores.copy()
        s[1:, 0] = -np.inf
        s[r, 0] = scores[r, 0]
        h = _cle(s)
        total = sum(scores[i, h[i]] for i in range(1, len(h)))
        if total > best:
            best, best_heads = total, h
    return best_heads

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import build_corpus  # noqa: E402


def _jsonl(tmp_path, docs):
    p = tmp_path / "src" / "docs.jsonl"
    p.parent.mkdir()
    p.write_text("".join(json.dumps({"title": None, "text": t}) + "\n" for t in docs), encoding="utf-8")
    return str(p)


def test_long_document_encoded_in_paragraph_pieces(tmp_path):
    # a whole book in one piece held 5 GB and starved the other builds
    para = "The ship sailed on through the long night and the crew kept watch. " * 10
    book = "\n\n".join([para] * 100)
    units = list(build_corpus.document_units([_jsonl(tmp_path, ["Lily saw a dog. She ran home.", book])]))
    assert units[0][2] == ["Lily saw a dog. She ran home."]          # a short document stays whole
    pieces = units[1:]
    assert len(pieces) > 1
    assert all(sum(len(p) + 1 for p in u[2]) <= build_corpus.MAX_UNIT_CHARS + 1 for u in pieces)
    assert sum(len(u[2]) for u in pieces) == 100                      # no paragraph lost or cut

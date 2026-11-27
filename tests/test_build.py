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


def test_scheduler_matches_build_by_exact_folder():
    import resume_builds as r
    c = r"C:\x\python.exe tools/build_corpus.py a.jsonl --out C:\d\corpus\tinystories-8000-b --resume --device cuda"
    assert r.out_folder(c) == r"C:\d\corpus\tinystories-8000-b"
    assert os.path.normcase(r.out_folder(c)) != os.path.normcase(r"C:\d\corpus\tinystories-8000")
    assert r.out_folder(r'python tools/build_corpus.py x --out "C:\a b\c"') == r"C:\a b\c"


def test_finished_build_not_resumed_when_document_count_changes(tmp_path):
    # a finished build says so in run.json; recounting its documents (now split
    # into pieces) must not make the scheduler resume it and write a tail twice
    import resume_builds as r
    b = tmp_path / "corpus" / "done"
    b.mkdir(parents=True)
    src = _jsonl(tmp_path, ["A dog ran. " * 3000])          # one document, now several pieces
    (b / "run.json").write_text(json.dumps({"files": [src], "finished": 1.0}), encoding="utf-8")
    (b / "progress.json").write_text(json.dumps({"docs": 1}), encoding="utf-8")
    [(name, _, done, total)] = r.state(str(tmp_path / "corpus"))
    assert (name, done, total) == ("done", 1, 1)

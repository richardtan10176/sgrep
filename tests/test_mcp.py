"""Tests for the MCP layer (src/sgrep_mcp.py).

The formatting/capping logic is pure and runs anywhere. The one test that
actually embeds needs the full runtime (onnxruntime + the downloaded model), so
it is skipped when that isn't importable rather than failing the suite.
"""

import os
import sys
import importlib.util

import pytest

SRC = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import sgrep_mcp  # noqa: E402


def _make_hit(code, path="pkg/mod.py", start=10, end=20, score=0.5,
              context="Function: do_thing"):
    return {
        "score": score,
        "display_path": path,
        "line_start": start,
        "line_end": end,
        "context": context,
        "code": code,
    }


def test_format_hits_renders_header_context_and_code():
    out = sgrep_mcp._format_hits([_make_hit("def do_thing():\n    return 1")])
    assert "pkg/mod.py:10-20  (0.500)" in out
    assert "Function: do_thing" in out
    assert "def do_thing():" in out


def test_format_hits_empty_is_the_no_match_notice():
    assert sgrep_mcp._format_hits([]) == sgrep_mcp.NO_MATCH


def test_format_hits_single_line_ref_when_no_end():
    hit = _make_hit("x = 1")
    hit["line_end"] = None
    assert "pkg/mod.py:10  (0.500)" in sgrep_mcp._format_hits([hit])


def test_cap_code_truncates_long_body_with_remaining_count():
    body = "\n".join(f"    line_{i}" for i in range(sgrep_mcp.CODE_LINE_CAP + 5))
    capped = sgrep_mcp._cap_code(body)
    assert "line_0" in capped
    assert f"line_{sgrep_mcp.CODE_LINE_CAP}" not in capped  # beyond the cap
    assert "truncated, 5 more lines" in capped


def test_cap_code_leaves_short_body_untouched():
    body = "def f():\n    return 42"
    assert sgrep_mcp._cap_code(body) == body
    assert "truncated" not in sgrep_mcp._cap_code(body)


def test_cap_code_char_cap_marks_truncation():
    body = "x" * (sgrep_mcp.CODE_CHAR_CAP + 100)  # one very long line
    capped = sgrep_mcp._cap_code(body)
    assert len(capped) <= sgrep_mcp.CODE_CHAR_CAP + len("\n… (truncated)")
    assert "truncated" in capped


def test_search_code_clamps_top_k_and_min_score(monkeypatch):
    captured = {}

    def fake_search(query, top_k, min_score):
        captured["top_k"] = top_k
        captured["min_score"] = min_score
        return []

    monkeypatch.setattr(sgrep_mcp, "_search", fake_search)
    sgrep_mcp.search_code("q", top_k=999, min_score=5.0)
    assert captured["top_k"] == sgrep_mcp.MAX_TOP_K
    assert captured["min_score"] == 1.0


# --- End-to-end embedding test (needs the real runtime) ---------------------

_HAS_RUNTIME = importlib.util.find_spec("onnxruntime") is not None


@pytest.mark.skipif(not _HAS_RUNTIME, reason="onnxruntime not installed")
def test_search_over_this_repo_finds_the_search_function(monkeypatch):
    """_search against sgrep's own src/ should surface its ranking function."""
    monkeypatch.setattr(sgrep_mcp, "_ROOT", SRC)
    results = sgrep_mcp._search("rank code chunks by cosine similarity",
                                top_k=5, min_score=0.0)
    assert results, "expected at least one hit"
    paths = {r["display_path"] for r in results}
    assert any("sgrep.py" in p for p in paths)

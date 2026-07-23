"""MCP server exposing sgrep's semantic search to AI coding agents.

A thin wrapper over the sgrep library: one `search_code` tool that reuses
`sgrep.init_path` (build/incremental-update/cache) and `sgrep.search` (cosine
ranking) unchanged. The point is to give an agent something grep can't do —
find code by *intent* — while staying local, so it fits agents running where
code can't leave the machine.

Design notes:
- The server searches only the directory it was launched in (`_ROOT`), captured
  once at startup. That matches how an agent host starts one MCP server per
  project, and means there is no arbitrary-path argument to sandbox.
- Every search refreshes the index through `init_path`, so results reflect edits
  the agent makes mid-session; the incremental re-index keeps that cheap.
- stdout is the JSON-RPC channel for stdio transport, so this module must never
  print there. The sgrep library already sends all its diagnostics to stderr,
  and we call `search`/`init_path` (never the stdout-printing `search_codebase`).
- The heavy imports (onnxruntime via the model, and the mcp SDK) are deferred so
  that importing this module for unit tests stays cheap and dependency-light.
"""

import os
import threading
import logging

import sgrep

log = logging.getLogger("sgrep.mcp")

# The one directory this server answers for: its launch cwd. Absolute so a later
# chdir can't move it out from under a half-built index.
_ROOT = os.path.abspath(os.getcwd())

# init_path rebuilds/saves the on-disk index; serialize those so two concurrent
# tool calls can't race on the same cache file. search() itself only reads the
# returned matrix, so it runs outside the lock.
_INDEX_LOCK = threading.Lock()

# Ceilings so a hit — or a whole response — can't blow out the agent's context.
MAX_TOP_K = 10
CODE_LINE_CAP = 40
CODE_CHAR_CAP = 1500

NO_MATCH = "No semantic matches (try grep for exact identifiers)."


def _ensure_index() -> "sgrep.Index":
    """Load/build/refresh the index for _ROOT, serialized against concurrent calls."""
    with _INDEX_LOCK:
        return sgrep.init_path(_ROOT)


def _search(query: str, top_k: int, min_score: float) -> list:
    """Raw hit dicts for `query`, reusing the sgrep retrieval core verbatim."""
    index = _ensure_index()
    return sgrep.search(
        query, index, top_k=top_k, model=sgrep.get_model(), min_score=min_score
    )


def _cap_code(code: str) -> str:
    """Trim a function body to the line/char caps, marking that it was clipped.

    Agents rarely need the whole function to judge relevance, and an uncapped
    body on `top_k=10` can dominate the response, so we clip and let the agent
    open the file (it has the file:line) when it wants the rest.
    """
    lines = code.splitlines()
    total = len(lines)
    text = "\n".join(lines[:CODE_LINE_CAP])
    truncated = total > CODE_LINE_CAP
    if len(text) > CODE_CHAR_CAP:
        text = text[:CODE_CHAR_CAP].rstrip()
        truncated = True
    if truncated:
        omitted = total - min(CODE_LINE_CAP, total)
        note = (f"… (truncated, {omitted} more line{'s' if omitted != 1 else ''})"
                if omitted else "… (truncated)")
        text = f"{text}\n{note}"
    return text


def _format_hits(results: list) -> str:
    """Render hits as text: `path:lines (score)`, context header, capped source."""
    if not results:
        return NO_MATCH
    blocks = []
    for hit in results:
        line_ref = hit["line_start"]
        if hit.get("line_end"):
            line_ref = f"{hit['line_start']}-{hit['line_end']}"
        header = f"{hit['display_path']}:{line_ref}  ({hit['score']:.3f})"
        blocks.append(f"{header}\n{hit['context']}\n{_cap_code(hit['code'])}")
    return "\n\n".join(blocks)


def search_code(query: str, top_k: int = 5, min_score: float = 0.25) -> str:
    """Search this codebase by intent/behavior, not by keyword.

    Use this when you don't know the exact identifier or string to grep for —
    e.g. "retry a request with backoff", "where do we validate auth tokens",
    "code that parses the config file". It returns the most semantically similar
    Python functions with their file:line location and source.

    This complements grep/ripgrep rather than replacing it: reach for grep when
    you already know the token to match (a function name, an error message, a
    config key); reach for this to locate the right region by what the code
    *does* when you can only describe the behavior.

    Args:
        query: A natural-language description of the code you're looking for.
        top_k: Max number of matches to return (1-10, default 5).
        min_score: Cosine floor below which a hit is treated as noise
            (0.0-1.0, default 0.25). Lower it to see weaker matches; 0 disables.

    Returns:
        Formatted matches, or a note that nothing cleared the score floor.
    """
    top_k = max(1, min(int(top_k), MAX_TOP_K))
    min_score = min(max(float(min_score), 0.0), 1.0)
    return _format_hits(_search(query, top_k, min_score))


def _warm() -> None:
    """Load the model and build the index up front so the first search is fast.

    Runs in a background thread; failures are logged, not fatal — the server
    still starts and the first real search will surface the error properly.
    """
    try:
        sgrep.get_model()
        _ensure_index()
        log.info("warm: index ready for %s", _ROOT)
    except Exception:
        log.exception("warmup failed")


def _build_server():
    """Construct the FastMCP server. Imports the SDK lazily (see module docstring)."""
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("sgrep")
    # Registered by reference (not the @tool decorator) so search_code stays a
    # plain, directly-callable function for tests; FastMCP reads its signature
    # and docstring for the tool schema and description.
    server.add_tool(search_code)
    return server


def main() -> None:
    threading.Thread(target=_warm, daemon=True).start()
    _build_server().run()


if __name__ == "__main__":
    main()

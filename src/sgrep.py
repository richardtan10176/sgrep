#!/usr/bin/env python3

import os
import sys
import json
import argparse
import hashlib
import subprocess

import numpy as np

import embedder


CACHE_DIR = os.path.expanduser("~/.cache/sgrep")

# Stamped into every index. The revision and runtime are part of the identity:
# an index built by the old torch path, or against different weights, is
# rejected by load_index and rebuilt rather than silently mixed with fresh
# query vectors.
MODEL_NAME = (
    f"{embedder.MODEL_REPO}@{embedder.MODEL_REVISION[:8]}+onnx"
)

# Cosine floor below which a hit is treated as noise rather than a match.
# Calibrated against the requests corpus: on-topic queries put their 5th-best
# hit at 0.34-0.62, while off-topic ones ("recipe for chocolate cake") top out
# around 0.20. Without a floor, cosine always returns *something* and "no
# matches" — and therefore exit code 1 — could never happen.
DEFAULT_MIN_SCORE = 0.25

# Bump when the on-disk layout changes in a way older readers can't handle.
# load_index refuses anything it doesn't recognise, so a stale cache is
# rebuilt rather than silently misinterpreted.
INDEX_FORMAT_VERSION = 1

# Directories that never contain code worth indexing.
SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn", "__pycache__", "node_modules", "build", "dist",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox", ".eggs",
    ".idea", ".vscode", "site-packages",
})

# Loaded lazily and cached so the (slow) model init happens at most once per process.
_MODEL = None


class IndexFormatError(Exception):
    """Raised when an on-disk index can't be used as-is and must be rebuilt."""


def get_model():
    """Load the embedding model once and reuse it."""
    global _MODEL
    if _MODEL is None:
        _MODEL = embedder.OnnxEmbedder()
    return _MODEL


def get_index_path(target_dir: str) -> str:
    abs_path = os.path.abspath(target_dir)
    # Not a security boundary, just a filename; usedforsecurity=False keeps
    # this working on FIPS-enabled hosts where plain md5 is unavailable.
    path_hash = hashlib.md5(
        abs_path.encode('utf-8'), usedforsecurity=False
    ).hexdigest()
    return os.path.join(CACHE_DIR, f"index_{path_hash}.npz")


class Index:
    """Chunk metadata plus a pre-normalized embedding matrix.

    The matrix is kept contiguous and L2-normalized once, at construction, so
    a query is a single matmul instead of a per-search rebuild of the corpus.

    `files` maps absolute source path -> [mtime_ns, size] as of index time; it
    is what lets `update_index` re-embed only what actually changed.
    """

    def __init__(self, metadata, embeddings, model_name: str = MODEL_NAME,
                 files: dict = None):
        self.metadata = list(metadata)
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2:
            raise IndexFormatError(f"expected a 2-D embedding matrix, got {matrix.shape}")
        if len(matrix) != len(self.metadata):
            raise IndexFormatError(
                f"{len(matrix)} embeddings for {len(self.metadata)} chunks"
            )
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        self.embeddings = matrix / np.maximum(norms, 1e-12)
        self.model_name = model_name
        self.files = dict(files or {})

    def __len__(self):
        return len(self.metadata)


def save_index(index_path: str, index: Index):
    """Persist an Index as an embedding matrix + JSON metadata + JSON header.

    Kept separate from pickle on purpose: np.load(..., allow_pickle=False)
    cannot execute arbitrary code, so a tampered cache file can't own us.
    """
    header = {
        "format_version": INDEX_FORMAT_VERSION,
        "model_name": index.model_name,
        "files": index.files,
    }
    np.savez_compressed(
        index_path,
        embeddings=index.embeddings,
        metadata=np.array(json.dumps(index.metadata)),
        header=np.array(json.dumps(header)),
    )


def load_index(index_path: str, model_name: str = MODEL_NAME) -> Index:
    """Read an index, rejecting anything this build can't interpret.

    Raises IndexFormatError for a missing/older format or an index built with a
    different embedding model — ranking those against fresh query embeddings
    would produce confident nonsense rather than an obvious failure.
    """
    with np.load(index_path, allow_pickle=False) as data:
        if 'header' not in data:
            raise IndexFormatError("index predates format versioning")
        header = json.loads(data['header'].item())
        embeddings = data['embeddings']
        metadata = json.loads(data['metadata'].item())

    version = header.get("format_version")
    if version != INDEX_FORMAT_VERSION:
        raise IndexFormatError(
            f"index format v{version}, this build reads v{INDEX_FORMAT_VERSION}"
        )

    stored_model = header.get("model_name")
    if model_name and stored_model != model_name:
        raise IndexFormatError(
            f"index built with {stored_model!r}, current model is {model_name!r}"
        )

    return Index(
        metadata,
        embeddings,
        model_name=stored_model,
        files=header.get("files") or {},
    )


def _chunk_display_path(chunk: dict) -> str:
    """Path to show for a hit: prefer the portable display_path baked at index
    time, else fall back to a cwd-relative path from the absolute file_path."""
    if chunk.get('display_path'):
        return chunk['display_path']
    if chunk.get('file_path'):
        return os.path.relpath(chunk['file_path'], os.getcwd())
    return "<unknown>"


def _git_tracked_python_files(root: str):
    """Python files under `root` that git doesn't ignore, or None if not a repo.

    Delegating to `git ls-files` gets full .gitignore semantics (nested
    ignore files, negations, excludesfile) for free instead of reimplementing
    them badly.
    """
    try:
        proc = subprocess.run(
            ["git", "-C", root, "ls-files", "--cached", "--others",
             "--exclude-standard", "-z"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except (OSError, ValueError):
        return None
    if proc.returncode != 0:
        return None

    paths = []
    for rel in proc.stdout.decode('utf-8', 'replace').split('\0'):
        if not rel.endswith(".py"):
            continue
        parts = rel.split("/")
        if SKIP_DIRS.intersection(parts[:-1]):
            continue
        paths.append(os.path.join(root, rel))
    return paths


def _walk_python_files(root: str):
    """Fallback scan for non-git trees: walk, pruning junk and virtualenvs."""
    paths = []
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [
            d for d in dirs
            if d not in SKIP_DIRS
            and not os.path.exists(os.path.join(dirpath, d, "pyvenv.cfg"))
        ]
        for file in files:
            if file.endswith(".py"):
                paths.append(os.path.join(dirpath, file))
    return paths


def scan(root: str, use_gitignore: bool = True) -> dict:
    """Map absolute .py path -> [mtime_ns, size] for every indexable file.

    Honors .gitignore inside a git repo, and otherwise skips vendored/build
    directories and virtualenvs — without this, pointing sgrep at a project
    that has a .venv means embedding all of site-packages.
    """
    root = os.path.abspath(root)

    paths = _git_tracked_python_files(root) if use_gitignore else None
    if paths is None:
        paths = _walk_python_files(root)

    found = {}
    for path in paths:
        try:
            st = os.stat(path)
        except OSError:
            continue
        found[os.path.abspath(path)] = [st.st_mtime_ns, st.st_size]
    return found


def _parse_one(file_path: str, display_root: str):
    """Parse one file. Returns (chunks, ok); ok is False if it couldn't be read.

    A file that parses but defines no functions is a success with no chunks —
    distinct from a failure, which is what the skipped-file count reports.
    """
    from packages.sgrep_visitor import sgrepVisitor

    try:
        with open(file_path, "r", encoding='utf-8') as f:
            source = f.read()
        chunks = sgrepVisitor().parse(source)
    except (OSError, UnicodeDecodeError, SyntaxError, ValueError, RecursionError):
        # Unreadable, not UTF-8, or not valid Python for this interpreter.
        return [], False

    display_path = os.path.relpath(file_path, display_root).replace(os.sep, "/")
    for chunk in chunks:
        chunk['file_path'] = file_path
        # Forward slashes so paths render consistently in the web UI
        # regardless of the OS the index was built on.
        chunk['display_path'] = display_path
    return chunks, True


def parse_file(file_path: str, display_root: str) -> list:
    """Parse one file into function chunks. Returns [] if it can't be read."""
    return _parse_one(file_path, display_root)[0]


def _parse_files(paths, display_root: str):
    """Parse many files, reporting how many had to be skipped."""
    chunks = []
    skipped = 0
    for path in paths:
        file_chunks, ok = _parse_one(path, display_root)
        if not ok:
            skipped += 1
        chunks.extend(file_chunks)
    if skipped:
        print(f"Skipped {skipped} unreadable or unparsable file(s).", file=sys.stderr)
    return chunks


def embed_chunks(chunks: list, model=None, batch_size: int = None) -> np.ndarray:
    """Embed chunk text; returns an (n, dim) float32 matrix."""
    if not chunks:
        return np.zeros((0, 0), dtype=np.float32)

    if model is None:
        model = get_model()
    if batch_size is None:
        batch_size = int(os.environ.get("SGREP_BATCH_SIZE", "64"))

    print(f"Generating embeddings for {len(chunks)} code chunks...", file=sys.stderr)
    texts = [f"{c['context']}\n{c['code']}" for c in chunks]
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    return np.asarray(embeddings, dtype=np.float32)


def build_index_from_root(root: str, display_root: str = None, model=None,
                          use_gitignore: bool = True) -> Index:
    """Scan, parse and embed `root` from scratch.

    `display_root` is the base a portable `display_path` is computed against
    (defaults to `root`).
    """
    root = os.path.abspath(root)
    if display_root is None:
        display_root = root

    print("Parsing codebase...", file=sys.stderr)
    files = scan(root, use_gitignore=use_gitignore)
    chunks = _parse_files(sorted(files), display_root)

    if not chunks:
        print("No chunks found!", file=sys.stderr)
        return Index([], np.zeros((0, 0), dtype=np.float32), files=files)

    embeddings = embed_chunks(chunks, model=model)
    return Index(chunks, embeddings, files=files)


def update_index(index: Index, root: str, display_root: str = None, model=None,
                 use_gitignore: bool = True):
    """Bring `index` up to date with `root`. Returns (Index, changed).

    Only files whose (mtime, size) moved — plus files that appeared — are
    re-parsed and re-embedded; chunks from deleted files are dropped. This is
    what keeps results from going stale without paying for a full rebuild.
    """
    root = os.path.abspath(root)
    if display_root is None:
        display_root = root

    current = scan(root, use_gitignore=use_gitignore)
    previous = index.files

    stale = {p for p, key in current.items() if previous.get(p) != key}
    removed = set(previous) - set(current)

    if not stale and not removed:
        return index, False

    if removed:
        print(f"Dropping {len(removed)} deleted file(s) from the index.", file=sys.stderr)
    if stale:
        print(f"Re-indexing {len(stale)} new or changed file(s)...", file=sys.stderr)

    dropped = stale | removed
    keep = [i for i, c in enumerate(index.metadata)
            if c.get('file_path') not in dropped]
    metadata = [index.metadata[i] for i in keep]
    embeddings = (index.embeddings[keep] if len(index.embeddings)
                  else np.zeros((0, 0), dtype=np.float32))

    new_chunks = _parse_files(sorted(stale), display_root)
    if new_chunks:
        new_embeddings = embed_chunks(new_chunks, model=model)
        metadata = metadata + new_chunks
        embeddings = (np.vstack([embeddings, new_embeddings])
                      if len(embeddings) else new_embeddings)

    if not metadata:
        embeddings = np.zeros((0, 0), dtype=np.float32)

    return Index(metadata, embeddings, files=current), True


def build_index(repo_root: str, out_path: str) -> int:
    """Index `repo_root` and write a portable .npz to `out_path`.

    Used by the demo index builder. Display paths are relative to repo_root so
    results render the same regardless of where the index was built, and the
    build-machine absolute paths are stripped so they don't ship in the image.
    Returns the number of chunks indexed.
    """
    index = build_index_from_root(repo_root, display_root=repo_root)
    if not len(index):
        return 0

    for chunk in index.metadata:
        chunk.pop('file_path', None)
    index.files = {}

    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    save_index(out_path, index)
    return len(index)


def search(query: str, index: Index, top_k: int = 3, model=None,
           min_score: float = 0.0) -> list:
    """Return the top_k semantic matches for `query` as structured dicts.

    Ranking is cosine similarity against the index's pre-normalized matrix, so
    a query costs one encode plus one matmul. Hits scoring below `min_score`
    are dropped, which is what lets callers distinguish "nothing relevant"
    from "here are the least-bad rows in the corpus".
    """
    if index is None or not len(index):
        return []

    if model is None:
        model = get_model()

    query_embedding = np.asarray(model.encode(query), dtype=np.float32).reshape(-1)
    q = query_embedding / max(float(np.linalg.norm(query_embedding)), 1e-12)

    if q.shape[0] != index.embeddings.shape[1]:
        raise IndexFormatError(
            f"query dim {q.shape[0]} != index dim {index.embeddings.shape[1]}"
        )

    scores = index.embeddings @ q

    top_k = max(1, min(top_k, len(index)))
    top_idx = np.argpartition(-scores, top_k - 1)[:top_k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]

    results = []
    for i in top_idx:
        score = float(scores[i])
        if score < min_score:
            break          # top_idx is sorted, so everything after is worse
        chunk = index.metadata[int(i)]
        results.append({
            "score": score,
            "display_path": _chunk_display_path(chunk),
            "line_start": chunk['line_start'],
            "line_end": chunk.get('line_end'),
            "context": chunk['context'],
            "code": chunk['code'],
        })
    return results


def format_results(results: list, style: str = "full") -> str:
    """Render hits for the terminal.

    `path` emits `file:line:` lines that editors, xargs and $EDITOR +N can
    consume directly; `full` is the human-readable listing.
    """
    if style == "json":
        return json.dumps(results, indent=2)

    if style == "path":
        return "\n".join(
            f"{h['display_path']}:{h['line_start']}:{h['context']}"
            for h in results
        )

    lines = []
    for hit in results:
        line_ref = hit['line_start']
        if hit.get('line_end'):
            line_ref = f"{hit['line_start']}-{hit['line_end']}"
        first_line = hit['code'].split('\n')[0].strip()
        lines.append(f"{hit['display_path']}:{line_ref}  ({hit['score']:.3f})")
        lines.append(f"  {hit['context']}")
        lines.append(f"  {first_line}")
        lines.append("")
    return "\n".join(lines).rstrip()


def search_codebase(target_dir: str, query: str, index: Index, top_k: int = 3,
                    style: str = "full", min_score: float = DEFAULT_MIN_SCORE) -> int:
    """Run a search, print it, and return the number of hits."""
    print(f"Searching for '{query}' in {os.path.abspath(target_dir)}",
          file=sys.stderr)

    results = search(query, index, top_k=top_k, min_score=min_score)
    if not results:
        print("No matches found.", file=sys.stderr)
        if style == "json":
            print("[]")
        return 0

    print(format_results(results, style))
    return len(results)


def init_path(path: str, reindex: bool = False) -> Index:
    """Load (and freshen) the index for `path`, building it if needed."""
    index_path = get_index_path(path)

    index = None
    if not reindex and os.path.exists(index_path):
        try:
            index = load_index(index_path)
        except IndexFormatError as exc:
            print(f"Cache unusable ({exc}), rebuilding", file=sys.stderr)
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            print("Cache corrupted, rebuilding", file=sys.stderr)

    if index is not None:
        index, changed = update_index(index, path)
        if not changed:
            return index
    else:
        index = build_index_from_root(path)
        if not len(index):
            return index

    os.makedirs(CACHE_DIR, exist_ok=True)
    save_index(index_path, index)
    print(f"Index saved to {index_path}", file=sys.stderr)
    return index


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="sgrep",
        description="Semantic code search over a local Python codebase.",
    )
    parser.add_argument("query", help="Query to search for")
    parser.add_argument("--dir", default="./", help="Directory context to search")
    parser.add_argument("--reindex", action="store_true", help="Force rebuild index")
    parser.add_argument("--top-k", "--top_K", dest="top_k", type=int, default=3,
                        help="Number of matches to show (default: 3)")
    parser.add_argument("--format", dest="style", default="full",
                        choices=("full", "path", "json"),
                        help="Output style: full listing, file:line:, or JSON")
    parser.add_argument("--json", dest="style", action="store_const", const="json",
                        help="Shorthand for --format json")
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE,
                        metavar="F",
                        help=f"Drop hits below this cosine score "
                             f"(default: {DEFAULT_MIN_SCORE}; 0 disables)")
    args = parser.parse_args(argv)

    abs_dir = os.path.abspath(args.dir)
    if not os.path.isdir(abs_dir):
        print(f"sgrep: not a directory: {abs_dir}", file=sys.stderr)
        return 2

    try:
        index = init_path(abs_dir, reindex=args.reindex)
        found = search_codebase(abs_dir, args.query, index, top_k=args.top_k,
                                style=args.style, min_score=args.min_score)
    except KeyboardInterrupt:
        return 2
    except (OSError, RuntimeError, IndexFormatError) as exc:
        print(f"sgrep: {exc}", file=sys.stderr)
        return 2

    return 0 if found else 1


if __name__ == "__main__":
    sys.exit(main())

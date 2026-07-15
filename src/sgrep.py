#!/usr/bin/env python3

import os
import json
import argparse
import hashlib
import sys

import numpy as np


CACHE_DIR = os.path.expanduser("~/.cache/sgrep")
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

# Loaded lazily and cached so the (slow) model init happens at most once per process.
_MODEL = None


def get_model():
    """Load the sentence-transformers model once and reuse it."""
    global _MODEL
    if _MODEL is None:
        print("Loading AI model...")
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer(MODEL_NAME)
    return _MODEL


def get_index_path(target_dir: str) -> str:
    abs_path = os.path.abspath(target_dir)
    path_hash = hashlib.md5(abs_path.encode('utf-8')).hexdigest()
    return os.path.join(CACHE_DIR, f"index_{path_hash}.npz")


def save_index(index_path: str, chunks: list):
    """Persist chunks as a stacked embedding matrix + JSON metadata.

    Kept separate from pickle on purpose: np.load(..., allow_pickle=False)
    cannot execute arbitrary code, so a tampered cache file can't own us.
    """
    embeddings = np.stack([c['embedding'] for c in chunks]).astype(np.float32)
    metadata = [
        {k: v for k, v in c.items() if k != 'embedding'}
        for c in chunks
    ]
    np.savez_compressed(
        index_path,
        embeddings=embeddings,
        metadata=np.array(json.dumps(metadata)),
    )


def load_index(index_path: str) -> list:
    with np.load(index_path, allow_pickle=False) as data:
        embeddings = data['embeddings']
        metadata = json.loads(data['metadata'].item())

    for i, chunk in enumerate(metadata):
        chunk['embedding'] = embeddings[i]
    return metadata


def _chunk_display_path(chunk: dict) -> str:
    """Path to show for a hit: prefer the portable display_path baked at index
    time, else fall back to a cwd-relative path from the absolute file_path."""
    if chunk.get('display_path'):
        return chunk['display_path']
    return os.path.relpath(chunk['file_path'], os.getcwd())


def build_chunks(root: str, display_root: str = None) -> list:
    """Walk `root`, parse every .py file into function chunks, and embed them.

    `display_root` is the base a portable `display_path` is computed against
    (defaults to `root`). Returns chunks with an 'embedding' attached.
    """
    from packages.sgrep_visitor import sgrepVisitor

    if display_root is None:
        display_root = root

    parsed_chunks = []

    print("Parsing codebase...")
    for dirpath, dirs, files in os.walk(root):
        for file in files:
            if not file.endswith(".py"):
                continue

            file_path = os.path.abspath(os.path.join(dirpath, file))

            try:
                with open(file_path, "r", encoding='utf-8') as f:
                    sample_code = f.read()

                visitor = sgrepVisitor()
                file_chunks = visitor.parse(sample_code)

                for chunk in file_chunks:
                    chunk['file_path'] = file_path
                    # Forward slashes so paths render consistently in the web UI
                    # regardless of the OS the index was built on.
                    chunk['display_path'] = os.path.relpath(
                        file_path, display_root
                    ).replace(os.sep, "/")
                    parsed_chunks.append(chunk)
            except Exception:
                # AST failed (syntax error) or the file was unreadable; skip it.
                pass

    if not parsed_chunks:
        print("No chunks found!")
        return []

    print(f"Generating embeddings for {len(parsed_chunks)} code chunks...")
    model = get_model()

    texts_to_embed = [
        f"{chunk['context']}\n{chunk['code']}"
        for chunk in parsed_chunks
    ]
    # BATCH SIZE SHOULD BE CONFIGURABLE SINCE NOT ALL HARDWARE IS CREATED EQUAL
    embeddings = model.encode(texts_to_embed, batch_size=64, show_progress_bar=True)

    for i, chunk in enumerate(parsed_chunks):
        chunk['embedding'] = embeddings[i]

    return parsed_chunks


def build_index(repo_root: str, out_path: str) -> int:
    """Index `repo_root` and write a portable .npz to `out_path`.

    Used by the demo index builder. Display paths are relative to repo_root so
    results render the same regardless of where the index was built. Returns the
    number of chunks indexed.
    """
    chunks = build_chunks(repo_root, display_root=repo_root)
    if not chunks:
        return 0

    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    save_index(out_path, chunks)
    return len(chunks)


def search(query: str, chunks: list, top_k: int = 3, model=None) -> list:
    """Return the top_k semantic matches for `query` as structured dicts.

    Ranking is cosine similarity between the query embedding and each chunk
    embedding (same metric as sentence_transformers' semantic_search), done in
    numpy so this stays independent of the heavy ML libs at call time.
    """
    if not chunks:
        return []

    if model is None:
        model = get_model()

    query_embedding = np.asarray(model.encode(query), dtype=np.float32)
    chunk_embeddings = np.stack([c['embedding'] for c in chunks]).astype(np.float32)

    q = query_embedding / (np.linalg.norm(query_embedding) + 1e-12)
    norms = np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
    scores = (chunk_embeddings / (norms + 1e-12)) @ q

    top_k = max(1, min(top_k, len(chunks)))
    top_idx = np.argpartition(-scores, top_k - 1)[:top_k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]

    results = []
    for i in top_idx:
        chunk = chunks[int(i)]
        results.append({
            "score": float(scores[i]),
            "display_path": _chunk_display_path(chunk),
            "line_start": chunk['line_start'],
            "line_end": chunk.get('line_end'),
            "context": chunk['context'],
            "code": chunk['code'],
        })
    return results


def search_codebase(dir: str, query: str, chunks: list, top_k: int = 3):
    """CLI entry point: run a search and print the results."""
    print(f"\nSearching for: '{query}' in '{os.path.abspath(dir)}'")

    results = search(query, chunks, top_k=top_k)

    print(f"Top {top_k} Matches:")
    print("-" * 40)
    for hit in results:
        line_ref = hit['line_start']
        if hit.get('line_end'):
            line_ref = f"{hit['line_start']}-{hit['line_end']}"
        print(f"Score: {hit['score']:.4f} | {hit['display_path']}:{line_ref}")
        print(f"Context: {hit['context']}")
        first_line = hit['code'].split('\n')[0].strip()
        print(f"Code: {first_line}")
        print("-" * 40)


def init_path(path: str, reindex: bool = False):
    index_path = get_index_path(path)

    if not reindex and os.path.exists(index_path):
        try:
            return load_index(index_path)
        except Exception:
            print("Cache corrupted, rebuilding")

    parsed_chunks = build_chunks(path)
    if not parsed_chunks:
        return []

    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    save_index(index_path, parsed_chunks)
    print(f"Index saved to {index_path}")

    return parsed_chunks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="Query to search for")
    parser.add_argument("--dir", default="./", help="Directory context to search")
    parser.add_argument("--reindex", action="store_true", help="Force rebuild index")
    parser.add_argument("--top_K", action="store", type=int, default=3, help="Top K matches")
    args = parser.parse_args()

    abs_dir = os.path.abspath(args.dir)

    chunks = init_path(abs_dir, reindex=args.reindex)
    search_codebase(abs_dir, args.query, chunks, top_k=args.top_K)

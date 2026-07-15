#!/usr/bin/env python3
"""Build the prebuilt semantic-search indexes bundled into the demo image.

For each configured repo we shallow-clone it, index only its package source
(cleaner, more recognizable results than tests/docs), and write:
    <out>/<id>.npz          the sgrep index (embeddings + metadata)
    <out>/manifest.json     [{id, name, description, chunk_count, npz}, ...]

At runtime the web app needs only these files (each chunk carries its own code
snippet), so the cloned source trees are throwaway. Intended to run at image
build time; needs git + network + the sentence-transformers model.
"""

import os
import sys
import json
import shutil
import argparse
import tempfile
import subprocess

# Make the sgrep library importable whether run from repo root or deploy/.
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, SRC_DIR)

import sgrep  # noqa: E402


# `subdir` is the path within the clone that holds the package source we index.
# `local` repos are indexed in place (no clone) — used for sgrep itself.
REPOS = [
    {
        "id": "requests",
        "name": "requests",
        "url": "https://github.com/psf/requests",
        "subdir": "src/requests",
        "description": "psf/requests — the iconic Python HTTP library.",
    },
    {
        "id": "flask",
        "name": "Flask",
        "url": "https://github.com/pallets/flask",
        "subdir": "src/flask",
        "description": "pallets/flask — a lightweight WSGI web framework.",
    },
    {
        "id": "fastapi",
        "name": "FastAPI",
        "url": "https://github.com/fastapi/fastapi",
        "subdir": "fastapi",
        "description": "fastapi/fastapi — modern async, type-hinted web framework.",
    },
    {
        "id": "sgrep",
        "name": "sgrep",
        "local": SRC_DIR,
        "subdir": ".",
        "description": "sgrep itself — the tool powering this demo.",
    },
]


def clone(url: str, dest: str):
    subprocess.run(
        ["git", "clone", "--depth", "1", url, dest],
        check=True,
    )


def build_one(repo: dict, out_dir: str, work_dir: str) -> dict:
    """Index a single repo; return its manifest entry (or None if empty)."""
    if repo.get("local"):
        root = repo["local"]
    else:
        clone_dir = os.path.join(work_dir, repo["id"])
        clone(repo["url"], clone_dir)
        root = clone_dir

    index_root = os.path.join(root, repo.get("subdir", "."))
    if not os.path.isdir(index_root):
        raise SystemExit(f"[{repo['id']}] expected source dir missing: {index_root}")

    npz_name = f"{repo['id']}.npz"
    out_path = os.path.join(out_dir, npz_name)

    print(f"\n=== Indexing {repo['id']} ({index_root}) ===")
    count = sgrep.build_index(index_root, out_path)
    if count == 0:
        print(f"[{repo['id']}] no chunks found, skipping")
        return None

    print(f"[{repo['id']}] indexed {count} chunks -> {out_path}")
    return {
        "id": repo["id"],
        "name": repo["name"],
        "description": repo["description"],
        "chunk_count": count,
        "npz": npz_name,
    }


def main():
    parser = argparse.ArgumentParser(description="Build demo search indexes")
    parser.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(__file__), "indexes"),
        help="Output directory for .npz files + manifest.json",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        help="Only build these repo ids (default: all)",
    )
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    repos = REPOS
    if args.only:
        wanted = set(args.only)
        repos = [r for r in REPOS if r["id"] in wanted]
        if not repos:
            raise SystemExit(f"No matching repos in {args.only}")

    work_dir = tempfile.mkdtemp(prefix="sgrep_build_")
    manifest = []
    try:
        for repo in repos:
            entry = build_one(repo, out_dir, work_dir)
            if entry:
                manifest.append(entry)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    if not manifest:
        raise SystemExit("No indexes were built!")

    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nWrote {len(manifest)} indexes + manifest -> {manifest_path}")
    for e in manifest:
        print(f"  - {e['id']}: {e['chunk_count']} chunks")


if __name__ == "__main__":
    main()

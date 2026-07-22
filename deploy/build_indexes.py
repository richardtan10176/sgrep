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
#
# Each remote repo is pinned to a tag *and* the commit that tag pointed at when
# it was added here. Without the pin, two builds of the same Dockerfile index
# whatever `main` happened to be, and image contents drift silently; the sha
# check also means a moved tag fails the build instead of changing the demo.
#
# `sha` must be the COMMIT, which for an annotated tag is not what a plain
# ls-remote prints — ask for the peeled ref and fall back to the direct one:
#   git ls-remote <url> 'refs/tags/<tag>^{}' 'refs/tags/<tag>' | head -1
REPOS = [
    {
        "id": "requests",
        "name": "requests",
        "url": "https://github.com/psf/requests",
        "ref": "v2.34.2",
        "sha": "6e83187b8feb273ed4c6cdab5efd8d54901dfab3",
        "subdir": "src/requests",
        "description": "psf/requests — the iconic Python HTTP library.",
    },
    {
        "id": "flask",
        "name": "Flask",
        "url": "https://github.com/pallets/flask",
        "ref": "3.1.3",
        "sha": "22d924701a6ae2e4cd01e9a15bbaf3946094af65",
        "subdir": "src/flask",
        "description": "pallets/flask — a lightweight WSGI web framework.",
    },
    {
        "id": "fastapi",
        "name": "FastAPI",
        "url": "https://github.com/fastapi/fastapi",
        "ref": "0.139.2",
        "sha": "866b7a3d0ce1025a3811f23aea4846d01a2b16a8",
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


def clone(url: str, ref: str, sha: str, dest: str):
    """Shallow-clone `ref` and verify it still resolves to the pinned commit."""
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", ref, url, dest],
        check=True,
    )
    head = subprocess.run(
        ["git", "-C", dest, "rev-parse", "HEAD"],
        check=True, stdout=subprocess.PIPE, text=True,
    ).stdout.strip()
    if head != sha:
        raise SystemExit(
            f"{url} {ref} resolves to commit {head}, expected pinned {sha}.\n"
            f"Either the tag moved, or the pin holds an annotated tag's object "
            f"hash instead of its commit — see the note above REPOS."
        )


def build_one(repo: dict, out_dir: str, work_dir: str) -> dict:
    """Index a single repo; return its manifest entry (or None if empty)."""
    if repo.get("local"):
        root = repo["local"]
    else:
        clone_dir = os.path.join(work_dir, repo["id"])
        clone(repo["url"], repo["ref"], repo["sha"], clone_dir)
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

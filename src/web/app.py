"""FastAPI web layer for the sgrep demo.

Loads the embedding model and the prebuilt repo indexes once at startup, then
serves a tiny UI plus a JSON search API. Visitors pick a prebuilt corpus and
query it; there is no user-supplied indexing, so the request path only embeds
the short query.
"""

import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# Make the sgrep library importable regardless of the working directory.
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import sgrep  # noqa: E402

REPO_ROOT = os.path.dirname(SRC_DIR)
INDEX_DIR = os.environ.get(
    "SGREP_INDEX_DIR", os.path.join(REPO_ROOT, "deploy", "indexes")
)
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

MAX_QUERY_LEN = 500
MAX_TOP_K = 10

# Populated at startup.
_MANIFEST = []          # public repo metadata (no embeddings)
_INDEXES = {}           # repo id -> list of chunks (with embeddings)
_READY = False


def _load_indexes():
    """Load manifest.json + every referenced .npz into memory."""
    import json

    manifest_path = os.path.join(INDEX_DIR, "manifest.json")
    if not os.path.exists(manifest_path):
        raise RuntimeError(
            f"No manifest at {manifest_path}; run deploy/build_indexes.py first."
        )

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for entry in manifest:
        npz_path = os.path.join(INDEX_DIR, entry["npz"])
        _INDEXES[entry["id"]] = sgrep.load_index(npz_path)
        _MANIFEST.append({
            "id": entry["id"],
            "name": entry["name"],
            "description": entry["description"],
            "chunk_count": entry["chunk_count"],
        })


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _READY
    sgrep.get_model()      # warm the model once (slow); blocks readiness until done
    _load_indexes()
    _READY = True
    yield


app = FastAPI(title="sgrep demo", lifespan=lifespan)


class SearchRequest(BaseModel):
    repo: str
    query: str = Field(..., min_length=1)
    top_k: int = 3


@app.get("/healthz")
def healthz():
    if not _READY:
        raise HTTPException(status_code=503, detail="warming up")
    return {"status": "ok", "repos": len(_INDEXES)}


@app.get("/api/repos")
def repos():
    return _MANIFEST


@app.post("/api/search")
def api_search(req: SearchRequest):
    if not _READY:
        raise HTTPException(status_code=503, detail="warming up")

    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is empty")
    if len(query) > MAX_QUERY_LEN:
        raise HTTPException(
            status_code=400, detail=f"query too long (max {MAX_QUERY_LEN} chars)"
        )

    chunks = _INDEXES.get(req.repo)
    if chunks is None:
        raise HTTPException(status_code=404, detail=f"unknown repo: {req.repo}")

    top_k = max(1, min(req.top_k, MAX_TOP_K))
    results = sgrep.search(query, chunks, top_k=top_k, model=sgrep.get_model())
    return {"repo": req.repo, "query": query, "results": results}


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

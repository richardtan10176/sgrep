"""FastAPI web layer for the sgrep demo.

Loads the embedding model and the prebuilt repo indexes once at startup, then
serves a tiny UI plus a JSON search API. Visitors pick a prebuilt corpus and
query it; there is no user-supplied indexing, so the request path only embeds
the short query.

Warmup runs in the background rather than blocking startup, so the server can
answer /healthz with an honest "503 warming up" instead of looking dead to a
readiness probe for the minute or so the model takes to load.
"""

import os
import sys
import json
import asyncio
import logging
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

# Inference is CPU-bound and the pod runs a single worker, so cap how many
# encodes can be in flight; excess callers get a fast 503 instead of piling
# onto a saturated box.
MAX_CONCURRENT_SEARCHES = int(os.environ.get("SGREP_MAX_CONCURRENCY", "4"))

log = logging.getLogger("sgrep.web")

# Populated by the background warmup task.
_MANIFEST = []          # public repo metadata (no embeddings)
_INDEXES = {}           # repo id -> sgrep.Index
_READY = False
_WARMUP_ERROR = None
_SEARCH_SLOTS = None    # asyncio.Semaphore, created inside the running loop


def _load_indexes():
    """Load manifest.json + every referenced .npz into memory."""
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


def _warm():
    """Blocking warmup; run in a worker thread so the event loop stays free."""
    sgrep.get_model()
    _load_indexes()


async def _warmup():
    global _READY, _WARMUP_ERROR
    try:
        await asyncio.to_thread(_warm)
    except Exception as exc:  # surfaced via /healthz; the pod stays unready
        _WARMUP_ERROR = str(exc)
        log.exception("warmup failed")
        return
    _READY = True
    log.info("warm: %d repos loaded", len(_INDEXES))


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _SEARCH_SLOTS
    _SEARCH_SLOTS = asyncio.Semaphore(MAX_CONCURRENT_SEARCHES)
    task = asyncio.create_task(_warmup())
    try:
        yield
    finally:
        task.cancel()


app = FastAPI(title="sgrep demo", lifespan=lifespan)


class SearchRequest(BaseModel):
    repo: str
    query: str = Field(..., min_length=1, max_length=MAX_QUERY_LEN)
    top_k: int = 3


def _require_ready():
    if _WARMUP_ERROR:
        raise HTTPException(status_code=500, detail="index load failed")
    if not _READY:
        raise HTTPException(status_code=503, detail="warming up")


@app.get("/healthz")
def healthz():
    if _WARMUP_ERROR:
        raise HTTPException(status_code=500, detail=f"warmup failed: {_WARMUP_ERROR}")
    if not _READY:
        raise HTTPException(status_code=503, detail="warming up")
    return {"status": "ok", "repos": len(_INDEXES)}


@app.get("/api/repos")
def repos():
    _require_ready()
    return _MANIFEST


@app.post("/api/search")
async def api_search(req: SearchRequest):
    _require_ready()

    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is empty")

    index = _INDEXES.get(req.repo)
    if index is None:
        raise HTTPException(status_code=404, detail=f"unknown repo: {req.repo}")

    top_k = max(1, min(req.top_k, MAX_TOP_K))

    if _SEARCH_SLOTS.locked():
        raise HTTPException(status_code=503, detail="busy, try again shortly")
    async with _SEARCH_SLOTS:
        results = await asyncio.to_thread(
            sgrep.search, query, index, top_k, sgrep.get_model()
        )
    return {"repo": req.repo, "query": query, "results": results}


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

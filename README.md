# sgrep: Local Semantic Code Search

sgrep is an open-source semantic search and basic Q&A tool for developers. It is a 100% local CLI based application utilizing a RAG architecture.

## Demo

Try it out for yourself at [https://sgrep.r-tan.dev/](https://sgrep.r-tan.dev/)

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

The first run downloads the embedding model (`all-MiniLM-L6-v2`, ~90 MB) into
the sentence-transformers cache.

## Usage

```bash
sgrep "retry a request with backoff"          # search the current directory
sgrep "parse the config file" --dir ~/code/myproject
sgrep "where do we validate tokens" --top-k 10
sgrep "..." --reindex                         # force a full rebuild
```

Indexes live in `~/.cache/sgrep/`, one per directory. Re-running only
re-embeds files whose mtime or size changed, so day-to-day searches stay fast
and never serve stale code. An index also records the model it was built with,
and is rebuilt automatically if that changes.

Only `.py` files are indexed. Inside a git repo sgrep honors `.gitignore`;
outside one it skips vendored/build directories and virtualenvs.

## Web demo

```bash
pip install -e '.[web]'
python deploy/build_indexes.py --out deploy/indexes   # clones + indexes the demo corpora
uvicorn web.app:app --app-dir src --port 8000
```

`GET /healthz` returns 503 while the model is still loading, then 200 —
warmup happens in the background, so the server is answerable throughout.
`deploy/Dockerfile` builds the same thing with the model and indexes baked in.

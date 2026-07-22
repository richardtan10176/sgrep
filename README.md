# sgrep: Local Semantic Code Search

sgrep finds code by intent instead of by keyword. Ask for "retry a request with
backoff" and it returns the function that does that, even when it contains none
of those words. Everything runs locally: your code is never sent anywhere.

## Demo

Try it at [sgrep.r-tan.dev](https://sgrep.r-tan.dev/) — a real index of
requests, Flask and FastAPI, searched from a terminal in the browser.

## Install

Requires Python 3.11+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

The first run downloads the embedding model (~90 MB) to `~/.cache/sgrep/models`.

## Usage

```bash
sgrep "retry a request with backoff"           # search the current directory
sgrep "parse the config file" --dir ~/code/myproject
sgrep "where do we validate tokens" --top-k 10
sgrep "..." --reindex                          # force a full rebuild
```

It composes like grep:

```bash
sgrep "database migrations" --format path      # file:line:context
sgrep "auth middleware" --json | jq '.[0].code'
sgrep "..." --format path | cut -d: -f1 | sort -u | xargs $EDITOR
```

**Exit codes:** `0` found something, `1` found nothing, `2` failed. Results go
to stdout, diagnostics to stderr, so pipes stay clean.

`--min-score` (default `0.25`) sets the cosine floor below which a hit counts as
noise. Without it cosine similarity always returns *something* — asking an HTTP
library about chocolate cake would confidently return its five least-irrelevant
functions. Pass `--min-score 0` to see everything.

## How it works

Files are parsed with `ast` into one chunk per function, each embedded with
all-MiniLM-L6-v2 and stored as a normalized matrix, so a query is one encode
plus one matmul.

Embedding runs the model's prebuilt **ONNX graph on onnxruntime** rather than
torch. That is not a micro-optimisation: importing torch dominated a run, and
removing it is what makes a cold invocation ~0.9 s instead of several seconds —
with no daemon, no background process, and nothing to restart. Vectors are
identical to the torch pipeline (cosine 1.000000, max elementwise difference
2.2e-07).

Indexes live in `~/.cache/sgrep/`, one per directory, and record the model
revision and each file's mtime/size. Re-running re-embeds only what changed and
drops deleted files; an index built by a different model is rejected and
rebuilt rather than silently mixed with fresh query vectors.

Only `.py` files are indexed. Inside a git repo sgrep honors `.gitignore`;
outside one it skips vendored/build directories and virtualenvs.

## Web demo

```bash
pip install -e '.[web]'
python deploy/build_indexes.py --out deploy/indexes   # clone + index the corpora
uvicorn web.app:app --app-dir src --port 8000
```

`GET /healthz` returns 503 while the model loads, then 200 — warmup runs in the
background so the server answers throughout. `deploy/Dockerfile` builds the same
thing with the model, indexes and corpus sources baked in.

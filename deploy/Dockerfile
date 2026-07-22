# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Builder: install deps, download the model, and prebuild the repo indexes.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

# git is needed to clone the demo corpora at build time.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    SGREP_MODEL_DIR=/opt/models

WORKDIR /app

# No torch: embedding runs the prebuilt ONNX graph on onnxruntime, which is
# ~20 MB of wheel instead of a multi-GB CUDA/CPU torch install.
COPY requirements.txt .
RUN pip install -r requirements.txt

# App source + index builder.
COPY src ./src
COPY deploy ./deploy

# Fetch the pinned model into /opt/models so the runtime image never needs
# network, then build /opt/indexes/*.npz + manifest + the source snapshot the
# demo's editor pane serves. The clones this makes are discarded by the builder.
RUN python -c "import sys; sys.path.insert(0, 'src'); import embedder; embedder.ensure_model()"
RUN python deploy/build_indexes.py --out /opt/indexes

# ---------------------------------------------------------------------------
# Runtime: copy installed packages + model + indexes. No pip, no network.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV SGREP_MODEL_DIR=/opt/models \
    SGREP_INDEX_DIR=/opt/indexes \
    SGREP_OFFLINE=1 \
    PYTHONUNBUFFERED=1

# Installed Python packages + console scripts from the builder.
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Baked model + prebuilt indexes and sources.
COPY --from=builder /opt/models /opt/models
COPY --from=builder /opt/indexes /opt/indexes

# App source (only src/ is needed at runtime).
WORKDIR /app
COPY src ./src

# Drop privileges.
RUN useradd -m -u 10001 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
WORKDIR /app/src
CMD ["uvicorn", "web.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

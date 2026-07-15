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
    HF_HOME=/opt/models \
    SENTENCE_TRANSFORMERS_HOME=/opt/models

WORKDIR /app

# CPU-only torch first (avoids the multi-GB CUDA wheel), then the rest.
COPY requirements.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

# App source + index builder.
COPY src ./src
COPY deploy ./deploy

# Downloads the model into /opt/models AND builds /opt/indexes/*.npz + manifest.
# Runtime needs only the .npz files (each chunk carries its own snippet), so the
# clones this makes are discarded by the builder itself.
RUN python deploy/build_indexes.py --out /opt/indexes

# ---------------------------------------------------------------------------
# Runtime: copy installed packages + model cache + indexes. No pip, no network.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

# libgomp1 is required by torch's CPU kernels.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV HF_HOME=/opt/models \
    SENTENCE_TRANSFORMERS_HOME=/opt/models \
    SGREP_INDEX_DIR=/opt/indexes \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONUNBUFFERED=1

# Installed Python packages + console scripts from the builder.
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Baked model cache + prebuilt indexes.
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

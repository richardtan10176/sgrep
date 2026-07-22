"""Query/chunk embedding via ONNX Runtime.

sgrep used to embed through sentence-transformers, which meant importing torch
and loading a 90 MB checkpoint on every CLI invocation — seconds of startup for
a tool whose actual search takes tens of milliseconds. The model ships prebuilt
ONNX graphs upstream, so we run the same network on onnxruntime instead and
drop torch entirely.

The pipeline reproduced here is exactly what the model card declares:
    Transformer -> Pooling(mean over tokens, attention-masked) -> L2 Normalize
"""

import os
import sys
import urllib.error
import urllib.request

import numpy as np


MODEL_REPO = "sentence-transformers/all-MiniLM-L6-v2"
# Pinned so the weights can't change under an existing index. Bumping this
# changes MODEL_NAME in sgrep.py, which invalidates and rebuilds old indexes.
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

# remote path -> local filename
MODEL_FILES = {
    "onnx/model.onnx": "model.onnx",
    "tokenizer.json": "tokenizer.json",
}

MAX_SEQ_LENGTH = 256
EMBEDDING_DIM = 384

DEFAULT_CACHE = os.path.expanduser("~/.cache/sgrep/models")


def model_dir() -> str:
    """Where the pinned model files live."""
    override = os.environ.get("SGREP_MODEL_DIR")
    if override:
        return override
    return os.path.join(DEFAULT_CACHE, MODEL_REVISION[:12])


def _offline() -> bool:
    return any(
        os.environ.get(k, "").lower() in ("1", "true", "yes")
        for k in ("SGREP_OFFLINE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    )


def _download(url: str, dest: str):
    """Fetch to a temp name then rename, so an interrupted download can't leave
    a truncated model that later fails in a confusing way."""
    tmp = dest + ".part"
    with urllib.request.urlopen(url) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        with open(tmp, "wb") as f:
            while True:
                block = resp.read(1 << 20)
                if not block:
                    break
                f.write(block)
                done += len(block)
                if total and sys.stderr.isatty():
                    pct = 100 * done / total
                    print(f"\r  {os.path.basename(dest)}: {pct:5.1f}%",
                          end="", file=sys.stderr)
    if total and sys.stderr.isatty():
        print(file=sys.stderr)
    os.replace(tmp, dest)


def ensure_model() -> str:
    """Make sure the pinned model is on disk; return its directory."""
    target = model_dir()
    missing = [
        (remote, os.path.join(target, local))
        for remote, local in MODEL_FILES.items()
        if not os.path.exists(os.path.join(target, local))
    ]
    if not missing:
        return target

    if _offline():
        raise RuntimeError(
            f"model files missing from {target} and offline mode is set"
        )

    os.makedirs(target, exist_ok=True)
    print(f"Fetching {MODEL_REPO} ({MODEL_REVISION[:8]}) -> {target}",
          file=sys.stderr)
    for remote, dest in missing:
        url = f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/{remote}"
        try:
            _download(url, dest)
        except urllib.error.URLError as exc:
            raise RuntimeError(f"could not download {url}: {exc}") from exc
    return target


class OnnxEmbedder:

    def __init__(self, threads: int = None):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        directory = ensure_model()

        opts = ort.SessionOptions()

        opts.intra_op_num_threads = threads or min(4, os.cpu_count() or 1)
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            os.path.join(directory, "model.onnx"),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self._input_names = {i.name for i in self.session.get_inputs()}

        self.tokenizer = Tokenizer.from_file(os.path.join(directory, "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=MAX_SEQ_LENGTH)
        pad_id = self.tokenizer.token_to_id("[PAD]")
        self.tokenizer.enable_padding(
            pad_id=pad_id if pad_id is not None else 0, pad_token="[PAD]"
        )

    def _forward(self, texts: list) -> np.ndarray:
        encodings = self.tokenizer.encode_batch(texts)

        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)

        feeds = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = np.array(
                [e.type_ids for e in encodings], dtype=np.int64
            )
        feeds = {k: v for k, v in feeds.items() if k in self._input_names}

        hidden = self.session.run(None, feeds)[0]
        if hidden.ndim != 3:
            raise RuntimeError(f"unexpected model output shape {hidden.shape}")

        mask = attention_mask[:, :, None].astype(np.float32)
        summed = (hidden * mask).sum(axis=1)
        counts = np.maximum(mask.sum(axis=1), 1e-9)
        pooled = summed / counts

        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        return (pooled / np.maximum(norms, 1e-12)).astype(np.float32)

    def encode(self, texts, batch_size: int = 64, show_progress_bar: bool = False):
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        if not items:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

        out = []
        for start in range(0, len(items), batch_size):
            out.append(self._forward(items[start:start + batch_size]))
            if show_progress_bar and sys.stderr.isatty():
                done = min(start + batch_size, len(items))
                print(f"\r  embedding {done}/{len(items)}", end="", file=sys.stderr)
        if show_progress_bar and sys.stderr.isatty():
            print(file=sys.stderr)

        result = np.vstack(out)
        return result[0] if single else result

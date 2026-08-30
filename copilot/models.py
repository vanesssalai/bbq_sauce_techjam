from __future__ import annotations

import math
import os
from pathlib import Path


def _pick_device() -> str:
    forced = os.environ.get("COPILOT_DEVICE", "").strip().lower()
    if forced in ("cpu", "cuda", "mps"):
        return forced
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            try:
                torch.ones(1, device="mps")
                return "mps"
            except Exception:
                return "cpu"
    except Exception:
        pass
    return "cpu"


def _construct(factory, source: str, device: str, **kwargs):
    try:
        return factory(source, device=device, **kwargs), device
    except Exception:
        if device == "cpu":
            raise
        return factory(source, device="cpu", **kwargs), "cpu"


_GPU_BATCH = 128

BI_ENCODER_NAME = "BAAI/bge-small-en-v1.5"
BI_ENCODER_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
BI_ENCODER_DIRNAME = "bge-small-en-v1.5"

BI_ENCODER_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CROSS_ENCODER_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
CROSS_ENCODER_DIRNAME = "ms-marco-MiniLM-L-6-v2"

NLI_ENCODER_NAME = "cross-encoder/nli-deberta-v3-xsmall"
NLI_ENCODER_REVISION: str | None = None
NLI_ENCODER_DIRNAME = "nli-deberta-v3-xsmall"

_NLI_ENTAIL_IDX = 1

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def _resolve(name: str, dirname: str) -> tuple[str, bool]:
    local = MODELS_DIR / dirname
    if (local / "config.json").is_file():
        return str(local), True
    return name, False


def _softmax(row) -> list[float]:
    vals = [float(x) for x in row]
    m = max(vals)
    exps = [math.exp(x - m) for x in vals]
    s = sum(exps) or 1.0
    return [e / s for e in exps]


class Encoder:
    def __init__(self, bi_encoder: "BiEncoder | None" = None) -> None:
        self._bi = bi_encoder if bi_encoder is not None else BiEncoder()

    def encode(self, texts, *, is_query: bool = False) -> list[list[float]]:
        vecs = self._bi.encode(list(texts), is_query=is_query)
        if hasattr(vecs, "tolist"):
            return vecs.tolist()
        return [list(v) for v in vecs]

    @property
    def dim(self) -> int:
        return int(self._bi.dim)

    @property
    def name(self) -> str:
        return self._bi.name


class BiEncoder:
    def __init__(self, name: str = BI_ENCODER_NAME, device: str | None = None) -> None:
        self.name = name
        self.device = device or _pick_device()
        self._model = None
        if name == BI_ENCODER_NAME:
            self.source, self.is_local = _resolve(name, BI_ENCODER_DIRNAME)
        else:
            self.source, self.is_local = name, Path(name).exists()

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            kwargs: dict = {}
            if not self.is_local:
                kwargs["revision"] = BI_ENCODER_REVISION
            self._model, self.device = _construct(
                SentenceTransformer, self.source, self.device, **kwargs
            )
        return self._model

    def encode(
        self,
        texts,
        *,
        is_query: bool = False,
        batch_size: int | None = None,
        normalize: bool = True,
    ):
        import numpy as np

        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        if not items:
            return np.zeros((0, self.dim), dtype="float32")
        if is_query:
            items = [BI_ENCODER_QUERY_PREFIX + t for t in items]
        model = self._load()
        bs = batch_size or (_GPU_BATCH if self.device in ("cuda", "mps") else 64)
        vecs = model.encode(
            items,
            batch_size=bs,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )
        vecs = np.asarray(vecs, dtype="float32")
        return vecs[0] if single else vecs

    @property
    def dim(self) -> int:
        return int(self._load().get_sentence_embedding_dimension())


class CrossEncoder:
    def __init__(
        self,
        name: str = CROSS_ENCODER_NAME,
        device: str | None = None,
        max_length: int = 512,
    ) -> None:
        self.name = name
        self.device = device or _pick_device()
        self.max_length = max_length
        self._model = None
        if name == CROSS_ENCODER_NAME:
            self.source, self.is_local = _resolve(name, CROSS_ENCODER_DIRNAME)
        else:
            self.source, self.is_local = name, Path(name).exists()

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder as _STCrossEncoder

            kwargs: dict = {"max_length": self.max_length}
            if not self.is_local:
                kwargs["revision"] = CROSS_ENCODER_REVISION
            self._model, self.device = _construct(
                _STCrossEncoder, self.source, self.device, **kwargs
            )
        return self._model

    def score(self, query: str, docs, batch_size: int | None = None):
        import numpy as np

        docs = list(docs)
        if not docs:
            return np.zeros(0, dtype="float32")
        model = self._load()
        bs = batch_size or (_GPU_BATCH if self.device in ("cuda", "mps") else 32)
        scores = model.predict(
            [(query, doc) for doc in docs],
            batch_size=bs,
            show_progress_bar=False,
        )
        return np.asarray(scores, dtype="float32")


class NliCrossEncoder:

    def __init__(self, name: str = NLI_ENCODER_NAME, device: str | None = None) -> None:
        self.name = name
        self.device = device or _pick_device()
        self._model = None
        if name == NLI_ENCODER_NAME:
            self.source, self.is_local = _resolve(name, NLI_ENCODER_DIRNAME)
        else:
            self.source, self.is_local = name, Path(name).exists()

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder as _STCrossEncoder

            kwargs: dict = {}
            if not self.is_local and NLI_ENCODER_REVISION:
                kwargs["revision"] = NLI_ENCODER_REVISION
            self._model, self.device = _construct(
                _STCrossEncoder, self.source, self.device, **kwargs
            )
        return self._model

    def entails_batch(self, pairs, batch_size: int | None = None) -> list[float]:
        pairs = list(pairs)
        if not pairs:
            return []
        model = self._load()
        bs = batch_size or (_GPU_BATCH if self.device in ("cuda", "mps") else 32)
        raw = model.predict(list(pairs), batch_size=bs, show_progress_bar=False)
        rows = raw.tolist() if hasattr(raw, "tolist") else list(raw)
        out: list[float] = []
        for row in rows:
            if isinstance(row, (int, float)):  # single-logit head -> sigmoid
                out.append(1.0 / (1.0 + math.exp(-float(row))))
                continue
            probs = _softmax(row)
            out.append(probs[_NLI_ENTAIL_IDX] if len(probs) > _NLI_ENTAIL_IDX else probs[-1])
        return out
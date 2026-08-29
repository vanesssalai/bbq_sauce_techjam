from __future__ import annotations

from pathlib import Path

BI_ENCODER_NAME = "BAAI/bge-small-en-v1.5"
BI_ENCODER_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
BI_ENCODER_DIRNAME = "bge-small-en-v1.5"

BI_ENCODER_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CROSS_ENCODER_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
CROSS_ENCODER_DIRNAME = "ms-marco-MiniLM-L-6-v2"

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def _resolve(name: str, dirname: str) -> tuple[str, bool]:
    local = MODELS_DIR / dirname
    if (local / "config.json").is_file():
        return str(local), True
    return name, False


class BiEncoder:
    def __init__(self, name: str = BI_ENCODER_NAME, device: str | None = None) -> None:
        self.name = name
        self.device = device
        self._model = None
        if name == BI_ENCODER_NAME:
            self.source, self.is_local = _resolve(name, BI_ENCODER_DIRNAME)
        else:
            self.source, self.is_local = name, Path(name).exists()

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            kwargs: dict = {"device": self.device}
            if not self.is_local:
                kwargs["revision"] = BI_ENCODER_REVISION
            self._model = SentenceTransformer(self.source, **kwargs)
        return self._model

    def encode(
        self,
        texts,
        *,
        is_query: bool = False,
        batch_size: int = 64,
        normalize: bool = True,
    ):
        import numpy as np

        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        if not items:
            return np.zeros((0, self.dim), dtype="float32")
        if is_query:
            items = [BI_ENCODER_QUERY_PREFIX + t for t in items]
        vecs = self._load().encode(
            items,
            batch_size=batch_size,
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
        self.device = device
        self.max_length = max_length
        self._model = None
        if name == CROSS_ENCODER_NAME:
            self.source, self.is_local = _resolve(name, CROSS_ENCODER_DIRNAME)
        else:
            self.source, self.is_local = name, Path(name).exists()

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder as _STCrossEncoder

            kwargs: dict = {"device": self.device, "max_length": self.max_length}
            if not self.is_local:
                kwargs["revision"] = CROSS_ENCODER_REVISION
            self._model = _STCrossEncoder(self.source, **kwargs)
        return self._model

    def score(self, query: str, docs, batch_size: int = 32):
        import numpy as np

        docs = list(docs)
        if not docs:
            return np.zeros(0, dtype="float32")
        scores = self._load().predict(
            [(query, doc) for doc in docs],
            batch_size=batch_size,
            show_progress_bar=False,
        )
        return np.asarray(scores, dtype="float32")
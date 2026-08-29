"""Model name constants and lazy sentence-transformers wrappers.

Wrappers load on first use, so importing this module needs no torch.
"""

from __future__ import annotations

BI_ENCODER_NAME = "BAAI/bge-small-en-v1.5"
CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class BiEncoder:
    """Dense embedding model. ``encode`` returns L2-normalized float32 vectors."""

    def __init__(self, name: str = BI_ENCODER_NAME, device: str | None = None) -> None:
        self.name = name
        self.device = device
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.name, device=self.device)
        return self._model

    def encode(self, texts, batch_size: int = 64, normalize: bool = True):
        import numpy as np

        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        if not items:
            return np.zeros((0, self.dim), dtype="float32")
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
    """Pairwise reranker. ``score`` returns one float32 relevance score per doc."""

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

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder as _STCrossEncoder

            self._model = _STCrossEncoder(
                self.name, device=self.device, max_length=self.max_length
            )
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

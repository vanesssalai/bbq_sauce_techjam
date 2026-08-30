"""Precompute one dense embedding per catalog product, offline, reproducibly.

    python scripts/build_artifacts.py
    python scripts/build_artifacts.py --verify   # check the artifact matches the current catalog

Writes:
  data/dense_embeddings.npy   float16, (N, dim), rows L2-normalized, row i <-> catalog line i
  data/embedding_meta.json    {model, revision, dim, dtype, normalized, parent_asins,
                               catalog_sha256, embeddings_sha256, built_at}
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from copilot.catalog import dense_text
from copilot.models import BI_ENCODER_NAME, BI_ENCODER_REVISION, BiEncoder

CATALOG_PATH = REPO_ROOT / "data" / "catalog.jsonl"
NPY_PATH = REPO_ROOT / "data" / "dense_embeddings.npy"
META_PATH = REPO_ROOT / "data" / "embedding_meta.json"
BATCH_SIZE = 256


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_rows() -> list[dict]:
    with CATALOG_PATH.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build() -> int:
    import numpy as np

    rows = _read_rows()
    texts = [dense_text(r) for r in rows]
    print(f"encoding {len(texts)} products with {BI_ENCODER_NAME} ...")
    started = time.time()

    encoder = BiEncoder()
    vecs = encoder.encode(texts, batch_size=BATCH_SIZE, normalize=True)
    vecs = np.asarray(vecs, dtype="float16")
    np.save(NPY_PATH, vecs)

    META_PATH.write_text(
        json.dumps(
            {
                "model": BI_ENCODER_NAME,
                "revision": BI_ENCODER_REVISION,
                "dim": int(vecs.shape[1]),
                "dtype": "float16",
                "normalized": True,
                "parent_asins": [str(r["parent_asin"]) for r in rows],
                "catalog_sha256": _sha256(CATALOG_PATH),
                "embeddings_sha256": _sha256(NPY_PATH),
                "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {NPY_PATH.relative_to(REPO_ROOT)} {tuple(vecs.shape)} and "
        f"{META_PATH.relative_to(REPO_ROOT)} in {time.time() - started:.1f}s\n"
        f"catalog_sha256={_sha256(CATALOG_PATH)}\nembeddings_sha256={_sha256(NPY_PATH)}"
    )
    return 0


def verify() -> int:
    if not (NPY_PATH.is_file() and META_PATH.is_file()):
        print("artifact missing - run without --verify first", file=sys.stderr)
        return 1
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    if meta.get("catalog_sha256") != _sha256(CATALOG_PATH):
        print("STALE: embedding_meta.catalog_sha256 != current catalog", file=sys.stderr)
        return 1
    if meta.get("embeddings_sha256") != _sha256(NPY_PATH):
        print("CORRUPT: embeddings_sha256 mismatch", file=sys.stderr)
        return 1
    print(f"OK  {meta['dim']}-d, {len(meta['parent_asins'])} rows, model {meta['model']}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    raise SystemExit(verify() if args.verify else build())


if __name__ == "__main__":
    main()

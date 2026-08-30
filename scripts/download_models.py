"""Vendor the retrieval models into ``models/`` so official scoring can run offline.

    python scripts/download_models.py            # fetch both models + write models/SHA256SUMS
    python scripts/download_models.py --verify   # re-hash local files against models/SHA256SUMS

Both models are pinned to exact commit revisions (see ``copilot/models.py``).
Only the weights sentence-transformers actually needs are fetched: the
safetensors checkpoint, the tokenizer, and the ST config files -- no ONNX,
OpenVINO, Flax, or duplicate ``.bin`` copies.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from copilot.models import (
    BI_ENCODER_DIRNAME,
    BI_ENCODER_NAME,
    BI_ENCODER_REVISION,
    CROSS_ENCODER_DIRNAME,
    CROSS_ENCODER_NAME,
    CROSS_ENCODER_REVISION,
    MODELS_DIR,
)

MODELS = [
    (BI_ENCODER_NAME, BI_ENCODER_REVISION, BI_ENCODER_DIRNAME),
    (CROSS_ENCODER_NAME, CROSS_ENCODER_REVISION, CROSS_ENCODER_DIRNAME),
]
ALLOW_PATTERNS = ["*.json", "*.txt", "model.safetensors", "1_Pooling/*"]
SUMS_PATH = MODELS_DIR / "SHA256SUMS"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tracked_files():
    """Every vendored model file, sorted, excluding SHA256SUMS and HF cache metadata."""
    for path in sorted(MODELS_DIR.rglob("*")):
        if path.is_file() and path != SUMS_PATH and ".cache" not in path.parts:
            yield path


def download() -> int:
    from huggingface_hub import snapshot_download

    for repo, revision, dirname in MODELS:
        dest = MODELS_DIR / dirname
        print(f"{repo}@{revision[:12]} -> {dest.relative_to(REPO_ROOT)}")
        snapshot_download(
            repo_id=repo,
            revision=revision,
            local_dir=str(dest),
            allow_patterns=ALLOW_PATTERNS,
        )

    lines = [
        f"{_sha256(path)}  {path.relative_to(MODELS_DIR).as_posix()}"
        for path in _tracked_files()
    ]
    SUMS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {SUMS_PATH.relative_to(REPO_ROOT)} ({len(lines)} files)")
    return 0


def verify() -> int:
    if not SUMS_PATH.is_file():
        print("models/SHA256SUMS missing -- run without --verify first", file=sys.stderr)
        return 1

    expected: dict[str, str] = {}
    for line in SUMS_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, rel = line.split(maxsplit=1)
            expected[rel.strip()] = digest

    ok = True
    seen: set[str] = set()
    for path in _tracked_files():
        rel = path.relative_to(MODELS_DIR).as_posix()
        seen.add(rel)
        if rel not in expected:
            print(f"UNTRACKED  {rel}")
            ok = False
        elif _sha256(path) != expected[rel]:
            print(f"MISMATCH   {rel}")
            ok = False

    for rel in sorted(expected.keys() - seen):
        print(f"MISSING    {rel}")
        ok = False

    print("OK" if ok else "FAILED")
    return 0 if ok else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="re-hash local files against models/SHA256SUMS instead of downloading",
    )
    args = parser.parse_args()
    raise SystemExit(verify() if args.verify else download())


if __name__ == "__main__":
    main()

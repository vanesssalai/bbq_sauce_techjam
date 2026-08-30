"""One-time model vendoring for offline scoring.

Run WITH network access:  python -m scripts.fetch_models
Pulls the exact repos/revisions pinned in copilot/models.py into ./models/,
after which copilot.models._resolve() finds them and never touches the network.

Pass --check to only verify an existing ./models/ tree loads (no download).
"""
from __future__ import annotations

import argparse
import sys

from huggingface_hub import snapshot_download

from copilot import models as M

# (repo, revision, target dirname, required)
# the NLI model is only loaded when COPILOT_NLI is enabled (see copilot/dialog/nli.py)
JOBS = [
    (M.BI_ENCODER_NAME, M.BI_ENCODER_REVISION, M.BI_ENCODER_DIRNAME, True),
    (M.CROSS_ENCODER_NAME, M.CROSS_ENCODER_REVISION, M.CROSS_ENCODER_DIRNAME, True),
    (M.NLI_ENCODER_NAME, M.NLI_ENCODER_REVISION, M.NLI_ENCODER_DIRNAME, False),
]

# weight formats copilot never loads - skipped to keep the vendored tree small and
# deterministic across re-runs. all three target models ship model.safetensors, so the
# duplicate pytorch_model.bin is redundant; drop it here rather than pruning after.
IGNORE = [
    "pytorch_model.bin",
    "*.onnx",
    "onnx/*",
    "openvino/*",
    "*.h5",
    "*.msgpack",
    "*.ot",
    "coreml/*",
    "*.mlmodel",
    "*.tflite",
]


def fetch(include_optional: bool) -> None:
    M.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name, revision, dirname, required in JOBS:
        if not required and not include_optional:
            print(f"skip {name}  (optional; pass --with-nli to vendor it)")
            continue
        dest = M.MODELS_DIR / dirname
        tag = "required" if required else "optional (COPILOT_NLI)"
        print(f"-> {name}@{revision or 'main'}  [{tag}]\n   {dest}")
        snapshot_download(
            repo_id=name,
            revision=revision,
            local_dir=str(dest),
            ignore_patterns=IGNORE,
        )


def check() -> int:
    ok = True

    bi = M.BiEncoder()
    vec = bi.encode(["probe"], is_query=True)
    print(f"bi-encoder : {bi.source}  local={bi.is_local}  dim={len(vec[0])}")
    ok &= bi.is_local

    cross = M.CrossEncoder()
    scores = list(cross.score("red running shoes", ["nike pegasus trail", "ceramic coffee mug"]))
    print(f"cross-enc  : {cross.source}  local={cross.is_local}  score={scores}")
    ok &= cross.is_local

    if not ok:
        print("\nWARNING: at least one model is not resolving locally - "
              "the ./models/ tree is missing or incomplete.")
    return 0 if ok else 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="verify ./models/ without downloading")
    ap.add_argument("--with-nli", action="store_true", help="also vendor the optional NLI model")
    args = ap.parse_args()
    if not args.check:
        fetch(include_optional=args.with_nli)
    raise SystemExit(check())


if __name__ == "__main__":
    main()

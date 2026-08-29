import sys
from pathlib import Path

# Put the repo root on sys.path so `copilot` / `evaluator` import under pytest.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

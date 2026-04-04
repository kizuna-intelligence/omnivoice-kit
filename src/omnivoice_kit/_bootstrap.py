from __future__ import annotations

import sys
from pathlib import Path


def ensure_omnivoice_on_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    submodule_root = repo_root / "third_party" / "OmniVoice"
    if str(submodule_root) not in sys.path:
        sys.path.insert(0, str(submodule_root))
    return submodule_root


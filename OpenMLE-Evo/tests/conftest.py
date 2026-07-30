from __future__ import annotations

import sys
from pathlib import Path

OPENMLE_EVO_ROOT = Path(__file__).resolve().parents[1]
AIRA_EVO_SRC = OPENMLE_EVO_ROOT / "third_party" / "aira-evo" / "src"

for path in (OPENMLE_EVO_ROOT, AIRA_EVO_SRC):
    path_value = str(path)
    if path_value not in sys.path:
        sys.path.insert(0, path_value)

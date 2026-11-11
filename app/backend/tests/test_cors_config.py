"""P1.8 — CORS origins are explicit, not wildcard."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main as api


def test_cors_not_wildcard():
    assert "*" not in api.CORS_ALLOW_ORIGINS
    assert any("5173" in o for o in api.CORS_ALLOW_ORIGINS)

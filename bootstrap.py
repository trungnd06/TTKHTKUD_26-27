"""Use project-local optional dependencies without changing the global environment."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# A real virtual environment should use its own installed dependencies.
if sys.prefix == sys.base_prefix and (ROOT / '.vendor').exists():
    sys.path.insert(0, str(ROOT / '.vendor'))

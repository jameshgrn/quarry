"""Pressure test config — add all packages to sys.path."""

import sys
from pathlib import Path

# Add all quarry packages to path for testing without install
packages_dir = Path(__file__).parent.parent.parent / "packages"
for pkg in ["quarry-core", "quarry-connectors", "quarry-operators"]:
    src = packages_dir / pkg / "src"
    if src.exists() and str(src) not in sys.path:
        sys.path.insert(0, str(src))

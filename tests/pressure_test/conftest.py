"""Pressure test config — add all packages to sys.path."""

import sys
from pathlib import Path

tests_dir = Path(__file__).parent
if str(tests_dir) not in sys.path:
    sys.path.insert(0, str(tests_dir))

# Add all quarry packages to path for testing without install
packages_dir = Path(__file__).parent.parent.parent / "packages"
for pkg in [
    "quarry-core",
    "quarry-connectors",
    "quarry-operators",
    "quarry-registry",
    "quarry-cli",
]:
    src = packages_dir / pkg / "src"
    if src.exists() and str(src) not in sys.path:
        sys.path.insert(0, str(src))

"""Hydrology-domain checks — implements Check protocol with geospatial deps.

Lane: check
These live in quarry-operators (not quarry-core) because they need
rasterio/numpy to read raster data from backing stores.

Standalone checks can be run independently on any artifact, unlike
operator-internal checks which run during execute().
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from quarry_core.artifact import (
    Artifact,
    ArtifactType,
    BackingStoreKind,
    CheckResult,
    ValidationState,
)

from quarry_operators.d8_flow_direction import D8_DC, D8_DR, NODATA


class InternalOutletCount:
    """Check that no valid cells flow into nodata regions.

    Reads a D8 flow direction raster and counts cells whose flow direction
    points into a nodata/invalid neighbor (excluding domain boundary exits).
    Non-zero indicates nodata mask inconsistency or fill failure near
    nodata boundaries.

    Expects: single-band int16 raster with D8 encoding
    (0-7=directions, 8=OUTLET, 9=PIT, -1=NODATA).
    """

    @property
    def name(self) -> str:
        return "no_internal_outlets"

    @property
    def description(self) -> str:
        return "No valid cells flow into nodata regions (no flow leaks)"

    def run(self, artifact: Artifact) -> CheckResult:
        if artifact.type != ArtifactType.RASTER:
            return CheckResult(
                check_name=self.name,
                state=ValidationState.INVALID,
                message=f"Expected raster artifact, got {artifact.type.value}",
            )

        if artifact.backing is None or artifact.backing.kind != BackingStoreKind.LOCAL_FILE:
            return CheckResult(
                check_name=self.name,
                state=ValidationState.INVALID,
                message="Artifact must have a local file backing store",
            )

        path = Path(artifact.backing.uri)
        if not path.exists():
            return CheckResult(
                check_name=self.name,
                state=ValidationState.INVALID,
                message=f"File not found: {path}",
            )

        import rasterio

        with rasterio.open(path) as src:
            flow = src.read(1)
            nodata_val = src.nodata

        # Build validity mask
        valid = np.ones(flow.shape, dtype=bool)
        if nodata_val is not None:
            valid = flow != int(nodata_val)
        # Also exclude cells coded as NODATA in D8 encoding
        valid = valid & (flow != NODATA)

        count = self._count(flow, valid)

        if count == 0:
            return CheckResult(
                check_name=self.name,
                state=ValidationState.VALID,
                message="No flow leaks into nodata regions",
            )
        return CheckResult(
            check_name=self.name,
            state=ValidationState.WARN,
            message=f"{count} cells flow into nodata (potential mask inconsistency)",
        )

    @staticmethod
    def _count(flow: np.ndarray, valid: np.ndarray) -> int:
        nrows, ncols = flow.shape
        count = 0
        for r in range(nrows):
            for c in range(ncols):
                if not valid[r, c]:
                    continue
                d = int(flow[r, c])
                if d < 0 or d > 7:
                    continue
                nr = r + D8_DR[d]
                nc = c + D8_DC[d]
                if nr < 0 or nr >= nrows or nc < 0 or nc >= ncols:
                    continue
                if not valid[nr, nc]:
                    count += 1
        return count

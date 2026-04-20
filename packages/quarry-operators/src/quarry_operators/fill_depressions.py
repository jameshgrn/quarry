"""FillDepressionsOperator — fills depressions in a DEM using Priority-Flood.

First hydrology domain operator. Canonical preprocessing step before
D8 flow direction computation.

Algorithm: Priority-Flood (Wang & Liu 2006)
- O(n log n) single-pass with min-heap priority queue
- Boundary cells seed the queue as known outlets
- Interior cells lower than their discovery elevation get raised
- FIFO queue propagates fill through flat regions

Accepts: one raster artifact (single-band DEM)
Produces: one raster artifact (depression-filled DEM)
Checks: no_interior_pits, elevation_only_raised, backing_accessible
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from quarry_core.artifact import (
    Artifact,
    ArtifactType,
    BackingStore,
    BackingStoreKind,
    CheckResult,
    Lineage,
    SpatialDescriptor,
    ValidationState,
    content_hash,
)
from quarry_core.operator import (
    OperatorError,
    OperatorParams,
    OperatorResult,
    OperatorSpec,
    ResourceScale,
)


@dataclass(frozen=True)
class FillDepressionsParams(OperatorParams):
    """Parameters for depression filling."""

    output_path: str = ""
    # Nodata value override (None = read from source)
    nodata: float | None = None
    # Apply micro-gradient to flat regions for D8 resolvability
    apply_gradient: bool = True
    # Gradient epsilon (elevation increment per cell in flat regions)
    epsilon: float = 1e-5


class FillDepressionsOperator:
    """Fills depressions in a DEM using Priority-Flood algorithm.

    Ensures every cell can drain to the grid boundary, which is
    required for valid D8 flow direction computation.
    """

    @property
    def name(self) -> str:
        return "fill_depressions"

    @property
    def spec(self) -> OperatorSpec:
        return OperatorSpec(
            input_types=(ArtifactType.RASTER,),
            output_type=ArtifactType.RASTER,
            min_inputs=1,
            max_inputs=1,
            resource_scale=ResourceScale.MEDIUM,
        )

    def validate_inputs(self, inputs: list[Artifact], params: OperatorParams) -> list[str]:
        errors = []

        if not inputs:
            errors.append("Exactly one input raster required")
            return errors

        if len(inputs) > 1:
            errors.append(f"Expected 1 input, got {len(inputs)}")

        artifact = inputs[0]

        if artifact.type != ArtifactType.RASTER:
            errors.append(f"Input must be raster, got {artifact.type.value}")

        if not artifact.is_materialized:
            errors.append("Input raster is not materialized (lazy handle)")

        if not isinstance(params, FillDepressionsParams):
            errors.append("Params must be FillDepressionsParams")
            return errors

        if not params.output_path:
            errors.append("output_path is required")

        if params.epsilon <= 0:
            errors.append("epsilon must be positive")

        return errors

    def execute(self, inputs: list[Artifact], params: OperatorParams) -> OperatorResult:
        if not isinstance(params, FillDepressionsParams):
            raise OperatorError(self.name, "Params must be FillDepressionsParams")

        import rasterio

        artifact = inputs[0]
        input_path = artifact.backing.uri
        output_path = Path(params.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with rasterio.open(input_path) as src:
                if src.count != 1:
                    raise OperatorError(
                        self.name,
                        f"DEM must be single-band, got {src.count} bands",
                        inputs=[artifact.id],
                    )

                dem = src.read(1).astype(np.float64)
                nodata = params.nodata if params.nodata is not None else src.nodata
                meta = src.meta.copy()

            # Build validity mask (True = valid cell)
            valid = np.ones(dem.shape, dtype=bool)
            if nodata is not None:
                valid = ~np.isnan(dem) & (dem != nodata)
            else:
                valid = ~np.isnan(dem)

            # Run Priority-Flood
            filled = _priority_flood(dem, valid)

            # Apply micro-gradient to flat regions if requested
            if params.apply_gradient:
                filled = _apply_flat_gradient(filled, valid, params.epsilon)

            # Restore nodata
            if nodata is not None:
                filled[~valid] = nodata

            # Write output
            meta.update({"dtype": "float64"})
            with rasterio.open(output_path, "w", **meta) as dst:
                dst.write(filled, 1)

        except OperatorError:
            raise
        except Exception as e:
            raise OperatorError(
                self.name,
                f"Depression filling failed: {e}",
                inputs=[artifact.id],
            ) from e

        # Build output artifact with fresh metadata
        with rasterio.open(output_path) as out_src:
            out_bounds = out_src.bounds
            output_artifact = Artifact(
                type=ArtifactType.RASTER,
                name=output_path.stem,
                backing=BackingStore(
                    kind=BackingStoreKind.LOCAL_FILE,
                    uri=str(output_path),
                    size_bytes=output_path.stat().st_size,
                    content_hash=content_hash(output_path),
                ),
                spatial=SpatialDescriptor(
                    crs=str(out_src.crs) if out_src.crs else None,
                    extent=(
                        out_bounds.left,
                        out_bounds.bottom,
                        out_bounds.right,
                        out_bounds.top,
                    ),
                    resolution=(out_src.res[0], out_src.res[1]),
                    band_count=out_src.count,
                ),
                lineage=Lineage(
                    operation=self.name,
                    inputs=(artifact.id,),
                    params={
                        "nodata": nodata,
                        "apply_gradient": params.apply_gradient,
                        "epsilon": params.epsilon,
                    },
                ),
                metadata={
                    "driver": out_src.driver,
                    "dtype": str(out_src.dtypes[0]),
                    "algorithm": "priority_flood_wang_liu_2006",
                },
            )

        checks = self._run_checks(output_artifact, filled, dem, valid)
        return OperatorResult(artifact=output_artifact, checks=checks)

    def declared_checks(self) -> list[str]:
        return ["no_interior_pits", "elevation_only_raised", "backing_accessible"]

    def _run_checks(
        self,
        output: Artifact,
        filled: np.ndarray,
        original: np.ndarray,
        valid: np.ndarray,
    ) -> list[CheckResult]:
        results = []

        # No interior pits: every valid interior cell must have at least one
        # neighbor at equal or lower elevation
        pit_count = _count_interior_pits(filled, valid)
        if pit_count == 0:
            results.append(
                CheckResult(
                    check_name="no_interior_pits",
                    state=ValidationState.VALID,
                    message="No interior pits remain after filling",
                )
            )
        else:
            results.append(
                CheckResult(
                    check_name="no_interior_pits",
                    state=ValidationState.INVALID,
                    message=f"{pit_count} interior pits remain after filling",
                )
            )

        # Elevation only raised: filled >= original everywhere valid
        lowered = np.any(filled[valid] < original[valid] - 1e-10)
        if not lowered:
            results.append(
                CheckResult(
                    check_name="elevation_only_raised",
                    state=ValidationState.VALID,
                    message="All filled elevations >= original",
                )
            )
        else:
            results.append(
                CheckResult(
                    check_name="elevation_only_raised",
                    state=ValidationState.INVALID,
                    message="Some cells were lowered during filling",
                )
            )

        # Backing accessible
        if output.backing and Path(output.backing.uri).exists():
            results.append(
                CheckResult(
                    check_name="backing_accessible",
                    state=ValidationState.VALID,
                    message=f"File exists: {output.backing.uri}",
                )
            )
        else:
            results.append(
                CheckResult(
                    check_name="backing_accessible",
                    state=ValidationState.INVALID,
                    message="Output file not found",
                )
            )

        return results


# ---------------------------------------------------------------------------
# Algorithm: Priority-Flood (Wang & Liu 2006)
# ---------------------------------------------------------------------------

# D8 neighbor offsets (row, col)
_D8_DR = np.array([-1, -1, 0, 1, 1, 1, 0, -1], dtype=np.intp)
_D8_DC = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.intp)


def _priority_flood(dem: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Fill all depressions so every cell drains to the boundary.

    Uses a min-heap seeded with boundary cells. Interior cells that are
    lower than their discovery elevation get raised. A FIFO queue
    propagates fill through flat regions efficiently.
    """
    nrows, ncols = dem.shape
    filled = dem.copy()
    closed = ~valid  # nodata cells are pre-closed

    # Min-heap: (elevation, insertion_order, row, col)
    heap: list[tuple[float, int, int, int]] = []
    # FIFO queue for flat propagation: list of (elevation, row, col)
    pit_queue: list[tuple[float, int, int]] = []
    pit_pos = 0
    counter = 0

    # Seed heap with valid boundary cells
    for r in range(nrows):
        for c in range(ncols):
            if not valid[r, c]:
                continue
            if r == 0 or r == nrows - 1 or c == 0 or c == ncols - 1:
                heapq.heappush(heap, (filled[r, c], counter, r, c))
                counter += 1
                closed[r, c] = True

    # Process
    while heap or pit_pos < len(pit_queue):
        # Prefer FIFO pit queue (flat propagation) over heap
        if pit_pos < len(pit_queue):
            elv, r, c = pit_queue[pit_pos]
            pit_pos += 1
        else:
            elv, _, r, c = heapq.heappop(heap)

        for d in range(8):
            nr = r + _D8_DR[d]
            nc = c + _D8_DC[d]

            if nr < 0 or nr >= nrows or nc < 0 or nc >= ncols:
                continue
            if closed[nr, nc]:
                continue

            closed[nr, nc] = True

            if filled[nr, nc] <= elv:
                # Depression cell: raise to current elevation
                filled[nr, nc] = elv
                pit_queue.append((elv, nr, nc))
            else:
                # Higher cell: add to heap at its own elevation
                heapq.heappush(heap, (filled[nr, nc], counter, nr, nc))
                counter += 1

        # Memory: clear consumed pit queue entries periodically
        if pit_pos == len(pit_queue) and pit_pos > 1024:
            pit_queue.clear()
            pit_pos = 0

    return filled


# ---------------------------------------------------------------------------
# Flat gradient resolution
# ---------------------------------------------------------------------------


def _apply_flat_gradient(filled: np.ndarray, valid: np.ndarray, epsilon: float) -> np.ndarray:
    """Apply micro-gradient to flat regions so D8 can resolve flow direction.

    Strategy: BFS from higher terrain into flat regions, incrementing
    elevation by epsilon per step. This creates a gentle slope toward
    the outlet of each flat.
    """
    nrows, ncols = filled.shape
    gradient = filled.copy()

    # Identify flat cells: valid cells that were raised (have no lower valid neighbor)
    flat = np.zeros((nrows, ncols), dtype=bool)
    for r in range(1, nrows - 1):
        for c in range(1, ncols - 1):
            if not valid[r, c]:
                continue
            has_lower = False
            for d in range(8):
                nr = r + _D8_DR[d]
                nc = c + _D8_DC[d]
                if 0 <= nr < nrows and 0 <= nc < ncols and valid[nr, nc]:
                    if filled[nr, nc] < filled[r, c]:
                        has_lower = True
                        break
            if not has_lower:
                # Check if it's on the boundary (always has outlet)
                if r == 0 or r == nrows - 1 or c == 0 or c == ncols - 1:
                    continue
                flat[r, c] = True

    if not np.any(flat):
        return gradient

    # BFS from non-flat edges into flat regions
    # "edges" = flat cells adjacent to non-flat lower/equal terrain
    from collections import deque

    dist = np.full((nrows, ncols), -1, dtype=np.int32)
    queue: deque[tuple[int, int]] = deque()

    # Find flat cells that neighbor a non-flat cell with lower elevation
    # These are the outlets of flat regions
    for r in range(nrows):
        for c in range(ncols):
            if not flat[r, c]:
                continue
            for d in range(8):
                nr = r + _D8_DR[d]
                nc = c + _D8_DC[d]
                if 0 <= nr < nrows and 0 <= nc < ncols:
                    if valid[nr, nc] and not flat[nr, nc]:
                        # This flat cell touches a non-flat cell
                        dist[r, c] = 0
                        queue.append((r, c))
                        break

    # BFS to assign distance from outlet edge
    while queue:
        r, c = queue.popleft()
        for d in range(8):
            nr = r + _D8_DR[d]
            nc = c + _D8_DC[d]
            if 0 <= nr < nrows and 0 <= nc < ncols:
                if flat[nr, nc] and dist[nr, nc] == -1:
                    dist[nr, nc] = dist[r, c] + 1
                    queue.append((nr, nc))

    # Apply gradient: cells farther from outlet get higher elevation
    mask = dist >= 0
    gradient[mask] += dist[mask].astype(np.float64) * epsilon

    return gradient


# ---------------------------------------------------------------------------
# Check helpers
# ---------------------------------------------------------------------------


def _count_interior_pits(dem: np.ndarray, valid: np.ndarray) -> int:
    """Count interior cells that are strict local minima (lower than all valid neighbors)."""
    nrows, ncols = dem.shape
    count = 0
    for r in range(1, nrows - 1):
        for c in range(1, ncols - 1):
            if not valid[r, c]:
                continue
            is_pit = True
            for d in range(8):
                nr = r + _D8_DR[d]
                nc = c + _D8_DC[d]
                if 0 <= nr < nrows and 0 <= nc < ncols and valid[nr, nc]:
                    if dem[nr, nc] <= dem[r, c]:
                        is_pit = False
                        break
            if is_pit:
                count += 1
    return count

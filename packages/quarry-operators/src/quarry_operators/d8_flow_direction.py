"""D8FlowDirectionOperator — computes steepest-descent flow direction.

Second hydrology operator. Requires a depression-filled DEM as input
(output of FillDepressionsOperator).

Algorithm: D8 (O'Callaghan & Mark 1984)
- Each cell flows to the neighbor with steepest downhill slope
- Slope accounts for diagonal distance (sqrt(2) vs 1 for cardinal)
- Boundary cells with no lower neighbor are OUTLET
- Interior cells with no lower neighbor are PIT (should be zero after fill)

Direction encoding (row-major, clockwise from East):
    0=E, 1=SE, 2=S, 3=SW, 4=W, 5=NW, 6=N, 7=NE
    8=OUTLET, 9=PIT, -1=NODATA

Accepts: one raster artifact (single-band depression-filled DEM)
Produces: one raster artifact (int8 flow direction grid)
Checks: valid_code_set, no_pits, all_valid_assigned, backing_accessible
"""

from __future__ import annotations

import math
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# D8 direction offsets: (row_offset, col_offset)
# Index = direction code: 0=E, 1=SE, 2=S, 3=SW, 4=W, 5=NW, 6=N, 7=NE
D8_DR = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.intp)
D8_DC = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.intp)

# Distance weights: 1.0 for cardinal, sqrt(2) for diagonal
_SQRT2 = math.sqrt(2.0)
D8_DIST = np.array([1.0, _SQRT2, 1.0, _SQRT2, 1.0, _SQRT2, 1.0, _SQRT2])

# Special codes
OUTLET = 8
PIT = 9
NODATA = -1


# ---------------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class D8FlowDirectionParams(OperatorParams):
    """Parameters for D8 flow direction computation."""

    output_path: str | None = None
    nodata: float | None = None


class D8FlowDirectionOperator:
    """Computes D8 flow direction from a depression-filled DEM.

    Every valid cell is assigned the direction of steepest descent
    to one of its 8 neighbors. Boundary cells with no lower neighbor
    become OUTLET. Interior cells with no lower neighbor become PIT
    (should be zero if input is properly filled).
    """

    @property
    def name(self) -> str:
        return "d8_flow_direction"

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

        if not isinstance(params, D8FlowDirectionParams):
            errors.append("Params must be D8FlowDirectionParams")
            return errors

        if params.output_path is None:
            errors.append("output_path is required")

        return errors

    def execute(self, inputs: list[Artifact], params: OperatorParams) -> OperatorResult:
        if not isinstance(params, D8FlowDirectionParams):
            raise OperatorError(self.name, "Params must be D8FlowDirectionParams")

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

            # Build validity mask
            valid = np.ones(dem.shape, dtype=bool)
            if nodata is not None:
                valid = ~np.isnan(dem) & (dem != nodata)
            else:
                valid = ~np.isnan(dem)

            # Compute D8
            flow = _compute_d8(dem, valid)

            # Write output as int16 (supports -1 to 9)
            meta.update({"dtype": "int16", "nodata": NODATA})
            with rasterio.open(output_path, "w", **meta) as dst:
                dst.write(flow.astype(np.int16), 1)

        except OperatorError:
            raise
        except Exception as e:
            raise OperatorError(
                self.name,
                f"D8 flow direction failed: {e}",
                inputs=[artifact.id],
            ) from e

        # Build output artifact
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
                    params={"nodata": nodata},
                ),
                metadata={
                    "driver": out_src.driver,
                    "dtype": str(out_src.dtypes[0]),
                    "algorithm": "d8_steepest_descent",
                    "direction_encoding": (
                        "0=E,1=SE,2=S,3=SW,4=W,5=NW,6=N,7=NE,8=OUTLET,9=PIT,-1=NODATA"
                    ),
                },
            )

        checks = self._run_checks(output_artifact, flow, valid)
        return OperatorResult(artifact=output_artifact, checks=checks)

    def declared_checks(self) -> list[str]:
        return [
            "valid_code_set",
            "no_pits",
            "no_internal_outlets",
            "all_valid_assigned",
            "backing_accessible",
        ]

    def _run_checks(
        self,
        output: Artifact,
        flow: np.ndarray,
        valid: np.ndarray,
    ) -> list[CheckResult]:
        results = []

        valid_flow = flow[valid]

        # Valid code set: all values in {-1, 0..9}
        unique_codes = set(np.unique(flow).tolist())
        allowed = {NODATA, 0, 1, 2, 3, 4, 5, 6, 7, OUTLET, PIT}
        invalid_codes = unique_codes - allowed
        if not invalid_codes:
            results.append(
                CheckResult(
                    check_name="valid_code_set",
                    state=ValidationState.VALID,
                    message=f"All codes valid: {sorted(unique_codes)}",
                )
            )
        else:
            results.append(
                CheckResult(
                    check_name="valid_code_set",
                    state=ValidationState.INVALID,
                    message=f"Invalid direction codes found: {invalid_codes}",
                )
            )

        # No pits: count of PIT code should be 0 (input should be filled)
        pit_count = int(np.sum(valid_flow == PIT))
        if pit_count == 0:
            results.append(
                CheckResult(
                    check_name="no_pits",
                    state=ValidationState.VALID,
                    message="No interior pits (input properly filled)",
                )
            )
        else:
            results.append(
                CheckResult(
                    check_name="no_pits",
                    state=ValidationState.WARN,
                    message=f"{pit_count} interior pits detected (input may not be fully filled)",
                )
            )

        # No internal outlets: valid cells flowing into nodata (not at domain boundary)
        internal_outlets = _count_internal_outlets(flow, valid)
        if internal_outlets == 0:
            results.append(
                CheckResult(
                    check_name="no_internal_outlets",
                    state=ValidationState.VALID,
                    message="No flow leaks into nodata regions",
                )
            )
        else:
            results.append(
                CheckResult(
                    check_name="no_internal_outlets",
                    state=ValidationState.WARN,
                    message=(
                        f"{internal_outlets} cells flow into nodata (potential mask inconsistency)"
                    ),
                )
            )

        # All valid cells assigned: no valid cell should have NODATA code
        unassigned = int(np.sum(valid_flow == NODATA))
        if unassigned == 0:
            results.append(
                CheckResult(
                    check_name="all_valid_assigned",
                    state=ValidationState.VALID,
                    message="All valid cells have a flow direction",
                )
            )
        else:
            results.append(
                CheckResult(
                    check_name="all_valid_assigned",
                    state=ValidationState.INVALID,
                    message=f"{unassigned} valid cells have no flow direction",
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
# Algorithm: D8 steepest descent
# ---------------------------------------------------------------------------


def _compute_d8(dem: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Compute D8 flow direction for every cell.

    Two-pass algorithm:
    1. Steepest descent to strictly lower neighbors
    2. Flat resolution: PIT cells flow to equal-elevation neighbors that can drain

    Returns int array with codes: 0-7 (directions), 8 (OUTLET), 9 (PIT), -1 (NODATA).
    """
    nrows, ncols = dem.shape
    flow = np.full((nrows, ncols), NODATA, dtype=np.int8)

    # Pass 1: strict steepest descent
    for r in range(nrows):
        for c in range(ncols):
            if not valid[r, c]:
                continue

            is_boundary = r == 0 or r == nrows - 1 or c == 0 or c == ncols - 1

            max_slope = 0.0
            best_dir = -1

            for d in range(8):
                nr = r + D8_DR[d]
                nc = c + D8_DC[d]

                if nr < 0 or nr >= nrows or nc < 0 or nc >= ncols:
                    continue
                if not valid[nr, nc]:
                    continue

                drop = dem[r, c] - dem[nr, nc]
                if drop > 0:
                    slope = drop / D8_DIST[d]
                    if slope > max_slope:
                        max_slope = slope
                        best_dir = d

            if best_dir >= 0:
                flow[r, c] = best_dir
            elif is_boundary:
                flow[r, c] = OUTLET
            else:
                flow[r, c] = PIT

    # Pass 2: resolve flat-region PITs via iterative drainage propagation.
    # A PIT on a flat surface can flow to an equal-elevation neighbor that
    # already has a valid direction (0-7 or OUTLET). Iterate until stable.
    changed = True
    while changed:
        changed = False
        for r in range(nrows):
            for c in range(ncols):
                if flow[r, c] != PIT:
                    continue

                for d in range(8):
                    nr = r + D8_DR[d]
                    nc = c + D8_DC[d]

                    if nr < 0 or nr >= nrows or nc < 0 or nc >= ncols:
                        continue
                    if not valid[nr, nc]:
                        continue

                    # Equal elevation and neighbor can drain
                    if dem[nr, nc] <= dem[r, c] and 0 <= flow[nr, nc] <= OUTLET:
                        flow[r, c] = d
                        changed = True
                        break

    return flow


def _count_internal_outlets(flow: np.ndarray, valid: np.ndarray) -> int:
    """Count valid cells whose D8 direction points into a nodata cell.

    These are "internal outlets" — flow leaks into holes in the valid mask
    rather than reaching the domain boundary. On a clean filled DEM this
    should be zero; non-zero indicates nodata mask inconsistency or a
    fill failure near nodata boundaries.

    Excludes cells that flow off-grid (those are legitimate boundary outlets)
    and cells with special codes (OUTLET, PIT, NODATA).
    """
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

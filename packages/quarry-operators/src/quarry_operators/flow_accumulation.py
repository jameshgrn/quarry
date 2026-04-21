"""FlowAccumulationOperator — computes upstream contributing area from D8 flow directions.

Third hydrology operator. Completes the minimal hydrology spine:
fill_depressions → d8_flow_direction → flow_accumulation

Algorithm: Topological sort (Kahn's algorithm)
- Build in-degree count from D8 flow graph
- Process cells with zero in-degree first (headwater cells)
- Each cell passes its accumulated weight to its downstream neighbor
- O(n) time, single pass over all cells

Input: D8 flow direction raster (codes 0-7, 8=OUTLET, 9=PIT, -1=NODATA)
Output: float64 accumulation raster (each cell = count of upstream cells including self)
Checks: no_cycles, nonnegative, conservation, backing_accessible
"""

from __future__ import annotations

from collections import deque
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

# D8 direction offsets matching d8_flow_direction encoding
D8_DR = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.intp)
D8_DC = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.intp)

OUTLET = 8
PIT = 9
NODATA_CODE = -1


@dataclass(frozen=True)
class FlowAccumulationParams(OperatorParams):
    """Parameters for flow accumulation."""

    output_path: str = ""
    # Weight per cell (default 1.0 = count upstream cells)
    weight: float = 1.0


class FlowAccumulationOperator:
    """Computes upstream contributing area from a D8 flow direction grid.

    Uses topological sort to accumulate flow weights from headwaters
    downstream. Each cell's accumulation = sum of upstream accumulations + self weight.
    """

    @property
    def name(self) -> str:
        return "flow_accumulation"

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

        if not isinstance(params, FlowAccumulationParams):
            errors.append("Params must be FlowAccumulationParams")
            return errors

        if not params.output_path:
            errors.append("output_path is required")

        if params.weight <= 0:
            errors.append("weight must be positive")

        return errors

    def execute(self, inputs: list[Artifact], params: OperatorParams) -> OperatorResult:
        if not isinstance(params, FlowAccumulationParams):
            raise OperatorError(self.name, "Params must be FlowAccumulationParams")

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
                        f"Flow direction must be single-band, got {src.count} bands",
                        inputs=[artifact.id],
                    )

                flow = src.read(1).astype(np.int8)
                meta = src.meta.copy()

            # Determine valid mask from flow codes (0-7=direction, 8=OUTLET, 9=PIT)
            valid = (flow >= 0) & (flow <= PIT)

            # Check for cycles before accumulating
            has_cycle = _detect_cycle(flow, valid)
            if has_cycle:
                raise OperatorError(
                    self.name,
                    "Cycle detected in D8 flow network — input is invalid",
                    inputs=[artifact.id],
                )

            # Compute accumulation via topological sort
            acc = _accumulate_toposort(flow, valid, params.weight)

            # Write output
            meta.update({"dtype": "float64", "nodata": -1.0})
            with rasterio.open(output_path, "w", **meta) as dst:
                out = np.where(valid, acc, -1.0)
                dst.write(out, 1)

        except OperatorError:
            raise
        except Exception as e:
            raise OperatorError(
                self.name,
                f"Flow accumulation failed: {e}",
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
                    params={"weight": params.weight},
                ),
                metadata={
                    "driver": out_src.driver,
                    "dtype": str(out_src.dtypes[0]),
                    "algorithm": "topological_sort_kahn",
                },
            )

        checks = self._run_checks(output_artifact, acc, flow, valid, params.weight)
        return OperatorResult(artifact=output_artifact, checks=checks)

    def declared_checks(self) -> list[str]:
        return ["no_cycles", "nonnegative", "conservation", "backing_accessible"]

    def _run_checks(
        self,
        output: Artifact,
        acc: np.ndarray,
        flow: np.ndarray,
        valid: np.ndarray,
        weight: float,
    ) -> list[CheckResult]:
        results = []

        # No cycles (already checked during execute, but confirm in result)
        results.append(
            CheckResult(
                check_name="no_cycles",
                state=ValidationState.VALID,
                message="No cycles in D8 flow network",
            )
        )

        # Nonnegative: all valid accumulation values >= weight
        valid_acc = acc[valid]
        if np.all(valid_acc >= weight - 1e-10):
            results.append(
                CheckResult(
                    check_name="nonnegative",
                    state=ValidationState.VALID,
                    message=f"All accumulation values >= {weight} (min={valid_acc.min():.2f})",
                )
            )
        else:
            neg_count = int(np.sum(valid_acc < weight - 1e-10))
            results.append(
                CheckResult(
                    check_name="nonnegative",
                    state=ValidationState.INVALID,
                    message=f"{neg_count} cells have accumulation below {weight}",
                )
            )

        # Conservation: total accumulation at outlets should equal total input weight.
        # Sum of acc at outlet cells = sum over all valid cells of weight
        # (each cell contributes exactly 'weight' to exactly one outlet path)
        total_weight = float(np.sum(valid)) * weight
        outlet_mask = valid & ((flow == OUTLET) | (flow == PIT))
        outlet_acc_sum = float(np.sum(acc[outlet_mask]))
        residual = abs(outlet_acc_sum - total_weight)
        if residual < 1e-6:
            results.append(
                CheckResult(
                    check_name="conservation",
                    state=ValidationState.VALID,
                    message=(
                        f"Flow conserved: outlet sum={outlet_acc_sum:.1f}, "
                        f"total weight={total_weight:.1f}"
                    ),
                )
            )
        else:
            results.append(
                CheckResult(
                    check_name="conservation",
                    state=ValidationState.INVALID,
                    message=(
                        f"Flow NOT conserved: outlet sum={outlet_acc_sum:.1f}, "
                        f"total weight={total_weight:.1f}, residual={residual:.6f}"
                    ),
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
# Algorithm: topological sort accumulation (Kahn's algorithm)
# ---------------------------------------------------------------------------


def _accumulate_toposort(flow: np.ndarray, valid: np.ndarray, weight: float) -> np.ndarray:
    """Accumulate flow via topological sort.

    Each cell starts with `weight`. Cells are processed in topological order
    (upstream before downstream). Each cell passes its total accumulation to
    its downstream neighbor.
    """
    nrows, ncols = flow.shape
    acc = np.zeros((nrows, ncols), dtype=np.float64)

    # Initialize: every valid cell gets self-weight
    acc[valid] = weight

    # Build in-degree: count how many valid cells flow INTO each cell
    indegree = np.zeros((nrows, ncols), dtype=np.int32)

    for r in range(nrows):
        for c in range(ncols):
            if not valid[r, c]:
                continue
            d = flow[r, c]
            if d < 0 or d > 7:
                continue  # OUTLET, PIT, NODATA don't flow to a neighbor
            nr = r + D8_DR[d]
            nc = c + D8_DC[d]
            if 0 <= nr < nrows and 0 <= nc < ncols and valid[nr, nc]:
                indegree[nr, nc] += 1

    # Seed queue with cells that have zero in-degree (headwaters)
    queue: deque[tuple[int, int]] = deque()
    for r in range(nrows):
        for c in range(ncols):
            if valid[r, c] and indegree[r, c] == 0:
                queue.append((r, c))

    # Process in topological order
    while queue:
        r, c = queue.popleft()
        d = flow[r, c]
        if d < 0 or d > 7:
            continue  # OUTLET/PIT — accumulation stays, nothing to pass

        nr = r + D8_DR[d]
        nc = c + D8_DC[d]

        if 0 <= nr < nrows and 0 <= nc < ncols and valid[nr, nc]:
            acc[nr, nc] += acc[r, c]
            indegree[nr, nc] -= 1
            if indegree[nr, nc] == 0:
                queue.append((nr, nc))

    return acc


# ---------------------------------------------------------------------------
# Cycle detection
# ---------------------------------------------------------------------------


def _detect_cycle(flow: np.ndarray, valid: np.ndarray) -> bool:
    """Detect if the D8 flow graph contains a cycle.

    Uses Kahn's algorithm: if topological sort doesn't process all valid cells,
    a cycle exists.
    """
    nrows, ncols = flow.shape

    # Build in-degree
    indegree = np.zeros((nrows, ncols), dtype=np.int32)
    valid_count = 0

    for r in range(nrows):
        for c in range(ncols):
            if not valid[r, c]:
                continue
            valid_count += 1
            d = flow[r, c]
            if d < 0 or d > 7:
                continue
            nr = r + D8_DR[d]
            nc = c + D8_DC[d]
            if 0 <= nr < nrows and 0 <= nc < ncols and valid[nr, nc]:
                indegree[nr, nc] += 1

    # BFS from zero-indegree cells
    queue: deque[tuple[int, int]] = deque()
    for r in range(nrows):
        for c in range(ncols):
            if valid[r, c] and indegree[r, c] == 0:
                queue.append((r, c))

    processed = 0
    while queue:
        r, c = queue.popleft()
        processed += 1
        d = flow[r, c]
        if d < 0 or d > 7:
            continue
        nr = r + D8_DR[d]
        nc = c + D8_DC[d]
        if 0 <= nr < nrows and 0 <= nc < ncols and valid[nr, nc]:
            indegree[nr, nc] -= 1
            if indegree[nr, nc] == 0:
                queue.append((nr, nc))

    return processed < valid_count

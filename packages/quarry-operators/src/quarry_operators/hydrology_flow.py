"""HydrologyFlow — canonical composition of fill → D8 → accumulation.

Lane: flow
Composes three hydrology operators into a single end-to-end chain:
    1. FillDepressions  — remove interior pits from DEM
    2. D8FlowDirection  — compute steepest-descent flow directions
    3. FlowAccumulation — compute upstream contributing area

Each step executes through a supplied Executor. Each step's output
artifact feeds the next step's input. All artifacts, runs, checks,
and lineage edges are persisted to the Registry.

Usage:
    flow = HydrologyFlow(executor=LocalExecutor(), registry=registry)
    result = flow.run(dem_artifact, workspace=Path("/tmp/hydro"))

    # Discriminated union: result is either success or failure
    if result.success:
        # Type narrowing: result is HydrologyFlowSuccess
        print(result.filled_dem.id)
        print(result.flow_direction.id)
        print(result.flow_accumulation.id)
    else:
        # Type narrowing: result is HydrologyFlowFailure
        print(f"Failed at {result.failed_step}: {result.error}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from quarry_core.artifact import Artifact, CheckResult, ValidationState
from quarry_core.executor import RunRecord, RunStatus
from quarry_core.operator import Operator, OperatorParams

from quarry_operators.d8_flow_direction import D8FlowDirectionOperator, D8FlowDirectionParams
from quarry_operators.fill_depressions import FillDepressionsOperator, FillDepressionsParams
from quarry_operators.flow_accumulation import FlowAccumulationOperator, FlowAccumulationParams

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class HydrologyFlowParams:
    """Parameters for the full hydrology flow."""

    workspace: Path
    # FillDepressions
    nodata: float | None = None
    apply_gradient: bool = True
    epsilon: float = 1e-5
    # FlowAccumulation
    weight: float = 1.0


@dataclass
class HydrologyFlowSuccess:
    """Result of a successful hydrology flow — all 3 artifacts present.

    Invariants:
        - filled_dem, flow_direction, flow_accumulation are all non-None
        - failed_step is None
        - error is None
        - success property returns True
    """

    # All three artifacts completed
    filled_dem: Artifact
    flow_direction: Artifact
    flow_accumulation: Artifact

    # RunRecords from each stage
    runs: list[RunRecord] = field(default_factory=list)

    # Aggregated checks across all stages
    all_checks: list[CheckResult] = field(default_factory=list)

    # Success invariants
    failed_step: None = field(default=None, init=False)
    error: None = field(default=None, init=False)

    @property
    def success(self) -> bool:
        """Always True for success results."""
        return True

    @property
    def has_invalid_checks(self) -> bool:
        """True if any check is INVALID."""
        return any(c.state == ValidationState.INVALID for c in self.all_checks)

    @property
    def artifacts(self) -> list[Artifact]:
        """All three artifacts as a list."""
        return [self.filled_dem, self.flow_direction, self.flow_accumulation]


@dataclass
class HydrologyFlowFailure:
    """Result of a failed hydrology flow — partial artifacts may exist.

    Invariants:
        - failed_step is a non-empty string indicating which step failed
        - error is a non-empty string describing the failure
        - 0-2 artifacts may be present (depending on when failure occurred)
        - success property returns False
    """

    # Which step failed (required for failures)
    failed_step: str
    # Error message (required for failures)
    error: str

    # Partial artifacts — 0 to 2 of these may be set depending on failure timing
    filled_dem: Artifact | None = None
    flow_direction: Artifact | None = None
    flow_accumulation: Artifact | None = None

    # RunRecords from completed stages (may be partial)
    runs: list[RunRecord] = field(default_factory=list)

    # Aggregated checks from completed stages
    all_checks: list[CheckResult] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """Always False for failure results."""
        return False

    @property
    def has_invalid_checks(self) -> bool:
        """True if any check is INVALID."""
        return any(c.state == ValidationState.INVALID for c in self.all_checks)

    @property
    def artifacts(self) -> list[Artifact]:
        """Artifacts that were completed before failure."""
        return [a for a in [self.filled_dem, self.flow_direction, self.flow_accumulation] if a]


# Discriminated union type
HydrologyFlowResult = HydrologyFlowSuccess | HydrologyFlowFailure


class HydrologyFlow:
    """Composes fill → D8 → accumulation through executor + registry.

    Each step:
    1. Build operator + params
    2. Submit to executor → RunRecord
    3. Persist RunRecord to registry (cascades artifact + checks + lineage)
    4. Feed output artifact to next step
    """

    def __init__(self, executor, registry=None):
        """Initialize flow.

        Args:
            executor: Any Executor protocol implementation.
            registry: Optional Registry for persistence. If None, runs execute
                      but nothing is persisted.
        """
        self._executor = executor
        self._registry = registry

    def run(
        self,
        dem_artifact: Artifact,
        params: HydrologyFlowParams,
    ) -> HydrologyFlowResult:
        """Execute the full hydrology chain.

        Args:
            dem_artifact: Input DEM raster artifact (must be materialized).
            params: Flow parameters.

        Returns:
            HydrologyFlowSuccess if all steps complete, HydrologyFlowFailure otherwise.
        """
        workspace = Path(params.workspace)
        workspace.mkdir(parents=True, exist_ok=True)

        # Accumulators for incremental state
        runs: list[RunRecord] = []
        all_checks: list[CheckResult] = []
        filled_dem: Artifact | None = None
        flow_direction: Artifact | None = None
        flow_accumulation: Artifact | None = None

        # If registry provided, persist the input artifact first
        if self._registry is not None:
            self._registry.save_artifact(dem_artifact)

        # --- Step 1: Fill depressions ---
        fill_params = FillDepressionsParams(
            output_path=str(workspace / "filled_dem.tif"),
            nodata=params.nodata,
            apply_gradient=params.apply_gradient,
            epsilon=params.epsilon,
        )
        run_record = self._execute_step(
            operator=FillDepressionsOperator(),
            inputs=[dem_artifact],
            params=fill_params,
            step_name="fill_depressions",
        )
        runs.append(run_record)
        all_checks.extend(run_record.checks)
        if run_record.status != RunStatus.COMPLETED:
            # Step failed
            return HydrologyFlowFailure(
                failed_step="fill_depressions",
                error=run_record.error or "fill_depressions did not complete",
                runs=runs,
                all_checks=all_checks,
            )
        filled_dem = run_record.output.artifact

        # --- Step 2: D8 flow direction ---
        d8_params = D8FlowDirectionParams(
            output_path=str(workspace / "flow_direction.tif"),
            nodata=params.nodata,
        )
        run_record = self._execute_step(
            operator=D8FlowDirectionOperator(),
            inputs=[filled_dem],
            params=d8_params,
            step_name="d8_flow_direction",
        )
        runs.append(run_record)
        all_checks.extend(run_record.checks)
        if run_record.status != RunStatus.COMPLETED:
            return HydrologyFlowFailure(
                failed_step="d8_flow_direction",
                error=run_record.error or "d8_flow_direction did not complete",
                filled_dem=filled_dem,
                runs=runs,
                all_checks=all_checks,
            )
        flow_direction = run_record.output.artifact

        # --- Step 3: Flow accumulation ---
        acc_params = FlowAccumulationParams(
            output_path=str(workspace / "flow_accumulation.tif"),
            weight=params.weight,
        )
        run_record = self._execute_step(
            operator=FlowAccumulationOperator(),
            inputs=[flow_direction],
            params=acc_params,
            step_name="flow_accumulation",
        )
        runs.append(run_record)
        all_checks.extend(run_record.checks)
        if run_record.status != RunStatus.COMPLETED:
            return HydrologyFlowFailure(
                failed_step="flow_accumulation",
                error=run_record.error or "flow_accumulation did not complete",
                filled_dem=filled_dem,
                flow_direction=flow_direction,
                runs=runs,
                all_checks=all_checks,
            )
        flow_accumulation = run_record.output.artifact

        # Success — all three artifacts present
        return HydrologyFlowSuccess(
            filled_dem=filled_dem,
            flow_direction=flow_direction,
            flow_accumulation=flow_accumulation,
            runs=runs,
            all_checks=all_checks,
        )

    def _execute_step(
        self,
        operator: Operator,
        inputs: Sequence[Artifact],
        params: OperatorParams,
        step_name: str,
    ) -> RunRecord:
        """Execute one step and persist to registry.

        Args:
            operator: The operator to execute
            inputs: Input artifacts
            params: Operator parameters
            step_name: Name of the step for error reporting

        Returns:
            RunRecord from this step. The caller can access checks via
            run_record.checks. Failure is represented as status=FAILED.
        """
        run_record = self._executor.submit(operator, list(inputs), params)

        # Persist every attempted run. Failed runs are still real runs.
        if self._registry is not None:
            self._registry.save_run(run_record)

        return run_record

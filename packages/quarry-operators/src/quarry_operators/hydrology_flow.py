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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from quarry_core.artifact import Artifact, CheckResult, ValidationState
from quarry_core.executor import RunRecord, RunStatus
from quarry_core.operator import Operator, OperatorParams

from quarry_operators.d8_flow_direction import D8FlowDirectionOperator, D8FlowDirectionParams
from quarry_operators.fill_depressions import FillDepressionsOperator, FillDepressionsParams
from quarry_operators.flow_accumulation import FlowAccumulationOperator, FlowAccumulationParams


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
class HydrologyFlowResult:
    """Result of the full hydrology flow."""

    # Artifacts at each stage
    filled_dem: Artifact | None = None
    flow_direction: Artifact | None = None
    flow_accumulation: Artifact | None = None

    # RunRecords at each stage
    runs: list[RunRecord] = field(default_factory=list)

    # Aggregated checks across all stages
    all_checks: list[CheckResult] = field(default_factory=list)

    # Which step failed (None = success)
    failed_step: str | None = None
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.failed_step is None

    @property
    def has_invalid_checks(self) -> bool:
        return any(c.state == ValidationState.INVALID for c in self.all_checks)

    @property
    def artifacts(self) -> list[Artifact]:
        return [a for a in [self.filled_dem, self.flow_direction, self.flow_accumulation] if a]


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
            HydrologyFlowResult with artifacts, runs, and checks from each stage.
        """
        result = HydrologyFlowResult()
        workspace = Path(params.workspace)
        workspace.mkdir(parents=True, exist_ok=True)

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
            result=result,
            step_name="fill_depressions",
        )
        if not result.success:
            return result
        result.filled_dem = run_record.output.artifact

        # --- Step 2: D8 flow direction ---
        d8_params = D8FlowDirectionParams(
            output_path=str(workspace / "flow_direction.tif"),
            nodata=params.nodata,
        )
        run_record = self._execute_step(
            operator=D8FlowDirectionOperator(),
            inputs=[result.filled_dem],
            params=d8_params,
            result=result,
            step_name="d8_flow_direction",
        )
        if not result.success:
            return result
        result.flow_direction = run_record.output.artifact

        # --- Step 3: Flow accumulation ---
        acc_params = FlowAccumulationParams(
            output_path=str(workspace / "flow_accumulation.tif"),
            weight=params.weight,
        )
        run_record = self._execute_step(
            operator=FlowAccumulationOperator(),
            inputs=[result.flow_direction],
            params=acc_params,
            result=result,
            step_name="flow_accumulation",
        )
        if not result.success:
            return result
        result.flow_accumulation = run_record.output.artifact

        return result

    def _execute_step(
        self,
        operator: Operator,
        inputs: list[Artifact],
        params: OperatorParams,
        result: HydrologyFlowResult,
        step_name: str,
    ) -> RunRecord | None:
        """Execute one step, persist to registry, update result."""
        try:
            run_record = self._executor.submit(operator, inputs, params)
        except Exception as e:
            result.failed_step = step_name
            result.error = str(e)
            return None

        result.runs.append(run_record)

        if run_record.status != RunStatus.COMPLETED:
            result.failed_step = step_name
            result.error = run_record.error or f"{step_name} did not complete"
            return run_record

        # Collect checks
        result.all_checks.extend(run_record.checks)

        # Persist to registry (cascades artifact + checks + lineage)
        if self._registry is not None:
            self._registry.save_run(run_record)

        return run_record

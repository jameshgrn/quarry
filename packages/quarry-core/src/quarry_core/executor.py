"""Executor — where operations run.

Execution is orthogonal to logic. A flow should not care whether it runs
local, subprocess, SSH, SLURM, or cloud worker.

The executor protocol is deliberately minimal:
- Submit an operator with inputs and params
- Get back a run record
- Query status

First backend: LocalExecutor (subprocess / concurrent.futures).
Future backends: Dask, SLURM, SSH, cloud.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from quarry_core.artifact import Artifact, CheckResult
from quarry_core.operator import Operator, OperatorParams, OperatorResult

# ---------------------------------------------------------------------------
# Supporting types
# ---------------------------------------------------------------------------


class RunStatus(Enum):
    """Lifecycle of a run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class RunRecord:
    """The canonical record of an operator execution.

    This is what the registry stores. It captures:
    - What ran (operator + params)
    - What went in (input artifact IDs)
    - What came out (output artifact, or error)
    - Validation results
    - Timing and executor metadata
    """

    id: str
    operator_name: str
    status: RunStatus = RunStatus.PENDING

    # Inputs
    input_ids: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)

    # Output (populated on completion)
    output: OperatorResult | None = None

    # Timing
    submitted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: datetime | None = None
    completed_at: datetime | None = None

    # Executor metadata
    executor_name: str = ""
    executor_meta: dict[str, Any] = field(default_factory=dict)

    # Error info (populated on failure)
    error: str | None = None

    @property
    def checks(self) -> tuple[CheckResult, ...]:
        """Validation truth derives from the output operator result."""
        if self.output is None:
            return ()
        return tuple(self.output.checks)

    @property
    def duration_seconds(self) -> float | None:
        """Wall-clock duration if completed."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


# ---------------------------------------------------------------------------
# The Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Executor(Protocol):
    """Dispatches operator execution to a compute backend.

    The executor is responsible for:
    - Running the operator's execute method
    - Capturing the result or error
    - Producing a RunRecord
    - Not caring about what the operator does (that's the operator's job)
    """

    @property
    def name(self) -> str:
        """Executor backend name (e.g. 'local', 'dask', 'slurm')."""
        ...

    def submit(
        self,
        operator: Operator,
        inputs: list[Artifact],
        params: OperatorParams,
    ) -> RunRecord:
        """Submit an operator for execution.

        This may be synchronous (LocalExecutor) or async (Dask/SLURM).
        Either way, it returns a RunRecord immediately. Validation or execution
        failure is represented as a FAILED RunRecord, not an exception side channel.

        Args:
            operator: The operator to run.
            inputs: Input artifacts (must be materialized unless operator handles lazy).
            params: Typed parameters for the operator.

        Returns:
            RunRecord with lifecycle status populated. Backends may return a
            completed/failed record immediately or a pending/running record for async work.
        """
        ...

    def status(self, run_id: str) -> RunRecord:
        """Get current status of a run.

        Args:
            run_id: The run record ID.

        Returns:
            Updated RunRecord.

        Raises:
            RunNotFoundError: If run_id is unknown.
        """
        ...

    def wait(self, run_id: str, timeout_seconds: float | None = None) -> RunRecord:
        """Block until a run completes or times out.

        Args:
            run_id: The run record ID.
            timeout_seconds: Max time to wait. None = wait forever.

        Returns:
            Completed RunRecord.

        Raises:
            TimeoutError: If timeout exceeded.
            RunNotFoundError: If run_id is unknown.
        """
        ...


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ExecutorError(Exception):
    """Base error for executor failures."""


class RunNotFoundError(ExecutorError):
    """Unknown run ID."""

    def __init__(self, run_id: str):
        self.run_id = run_id
        super().__init__(f"Run not found: {run_id}")


class SubmitError(ExecutorError):
    """Failed to submit operator for execution."""

    def __init__(self, operator_name: str, reason: str):
        self.operator_name = operator_name
        self.reason = reason
        super().__init__(f"Failed to submit '{operator_name}': {reason}")

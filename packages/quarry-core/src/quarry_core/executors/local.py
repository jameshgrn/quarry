"""LocalExecutor — runs operators in-process, synchronously.

The simplest possible executor. No scheduling, no distribution.
Just: validate, execute, capture result, produce RunRecord.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from time import perf_counter

from quarry_core.artifact import Artifact
from quarry_core.executor import RunNotFoundError, RunRecord, RunStatus
from quarry_core.operator import Operator, OperatorParams


def _serialize_params(params: OperatorParams) -> dict:
    """Convert typed params to a plain dict for RunRecord persistence."""
    if is_dataclass(params):
        return asdict(params)
    if hasattr(params, "__dict__"):
        return dict(vars(params))
    return {}


class LocalExecutor:
    """Synchronous in-process executor."""

    def __init__(self) -> None:
        self._runs: dict[str, RunRecord] = {}

    @property
    def name(self) -> str:
        return "local"

    def submit(
        self,
        operator: Operator,
        inputs: list[Artifact],
        params: OperatorParams,
    ) -> RunRecord:
        """Execute operator synchronously and return completed RunRecord."""
        run_id = str(uuid.uuid4())
        record = RunRecord(
            id=run_id,
            operator_name=operator.name,
            status=RunStatus.PENDING,
            input_ids=[a.id for a in inputs],
            params=_serialize_params(params),
            executor_name=self.name,
        )

        # Validate inputs
        errors = operator.validate_inputs(inputs, params)
        if errors:
            record.status = RunStatus.FAILED
            record.error = f"Validation failed: {'; '.join(errors)}"
            self._runs[run_id] = record
            record.completed_at = datetime.now(tz=timezone.utc)
            return record

        # Execute
        record.status = RunStatus.RUNNING
        record.started_at = datetime.now(tz=timezone.utc)

        try:
            t0 = perf_counter()
            result = operator.execute(inputs, params)
            elapsed = perf_counter() - t0

            record.status = RunStatus.COMPLETED
            record.completed_at = datetime.now(tz=timezone.utc)
            record.output = result
            result.timing_seconds = elapsed

        except Exception as e:
            record.status = RunStatus.FAILED
            record.completed_at = datetime.now(tz=timezone.utc)
            record.error = str(e)

        self._runs[run_id] = record
        return record

    def status(self, run_id: str) -> RunRecord:
        """Get run record by ID."""
        if run_id not in self._runs:
            raise RunNotFoundError(run_id)
        return self._runs[run_id]

    def wait(self, run_id: str, timeout_seconds: float | None = None) -> RunRecord:
        """For LocalExecutor, submit is synchronous so wait just returns status."""
        return self.status(run_id)

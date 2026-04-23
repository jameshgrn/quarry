from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from quarry_core.artifact import (
    Artifact,
    ArtifactType,
    BackingStore,
    BackingStoreKind,
    CheckResult,
    ValidationState,
    content_hash,
)
from quarry_core.executor import RunRecord, RunStatus
from quarry_core.operator import OperatorResult


def make_materialized_artifact(path: Path, artifact_type: ArtifactType) -> Artifact:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".csv":
        path.write_text("id\n1\n")
    else:
        path.write_bytes(b"placeholder")

    return Artifact(
        type=artifact_type,
        name=path.stem,
        backing=BackingStore(
            kind=BackingStoreKind.LOCAL_FILE,
            uri=str(path),
            size_bytes=path.stat().st_size,
            content_hash=content_hash(path),
        ),
    )


def make_invalid_completed_run(
    workspace: Path,
    *,
    operator_name: str,
    artifact_type: ArtifactType,
    output_name: str,
    check_name: str = "semantic_truth",
) -> RunRecord:
    artifact = make_materialized_artifact(workspace / output_name, artifact_type)
    check = CheckResult(
        check_name=check_name,
        state=ValidationState.INVALID,
        message="pressure-test invalid output",
    )
    now = datetime.now(timezone.utc)
    return RunRecord(
        id=f"{operator_name}-invalid-run",
        operator_name=operator_name,
        status=RunStatus.COMPLETED,
        output=OperatorResult(artifact=artifact, checks=[check]),
        submitted_at=now,
        started_at=now,
        completed_at=now,
        executor_name="test",
    )

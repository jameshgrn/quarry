"""Registry — persistent memory for artifacts, runs, checks, and lineage.

Lane: registry

This is what turns transient contract objects into a real harness.
DuckDB-backed, single-file, append-oriented.

Design rules:
- Artifacts, runs, checks, and lineage each get their own table
- Check truth lives in the checks table, not embedded in artifacts
- Lineage is stored as edges (parent→child) for graph queries
- Serialization must be lossless: what goes in must come back identical
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

import duckdb
from quarry_core.artifact import (
    Artifact,
    ArtifactType,
    BackingStore,
    BackingStoreKind,
    CheckResult,
    Lineage,
    SpatialDescriptor,
    ValidationState,
)
from quarry_core.executor import RunRecord, RunStatus
from quarry_core.operator import OperatorResult


def _to_jsonable(value):
    """Convert immutable core payloads into plain JSON-compatible structures."""
    if isinstance(value, Mapping):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_to_jsonable(v) for v in value]
    return value


class Registry:
    """DuckDB-backed persistent registry for Quarry."""

    def __init__(self, workspace: Path | str):
        self.workspace = Path(workspace)
        self.db_path = self.workspace / ".quarry" / "registry.duckdb"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(str(self.db_path))

    def _init_schema(self) -> None:
        conn = self._connect()
        try:
            conn.execute("CREATE SEQUENCE IF NOT EXISTS checks_seq START 1")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS artifacts (
                    id VARCHAR PRIMARY KEY,
                    type VARCHAR NOT NULL,
                    name VARCHAR NOT NULL,
                    backing_kind VARCHAR,
                    backing_uri VARCHAR,
                    backing_size_bytes BIGINT,
                    backing_content_hash VARCHAR,
                    crs VARCHAR,
                    extent_xmin DOUBLE,
                    extent_ymin DOUBLE,
                    extent_xmax DOUBLE,
                    extent_ymax DOUBLE,
                    resolution_x DOUBLE,
                    resolution_y DOUBLE,
                    feature_count INTEGER,
                    band_count INTEGER,
                    lineage_operation VARCHAR,
                    lineage_inputs_json VARCHAR,
                    lineage_params_json VARCHAR,
                    lineage_timestamp TIMESTAMP WITH TIME ZONE,
                    lineage_executor_id VARCHAR,
                    metadata_json VARCHAR,
                    created_at TIMESTAMP WITH TIME ZONE NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    id VARCHAR PRIMARY KEY,
                    operator_name VARCHAR NOT NULL,
                    status VARCHAR NOT NULL,
                    input_ids_json VARCHAR,
                    output_artifact_id VARCHAR,
                    params_json VARCHAR,
                    executor_name VARCHAR,
                    executor_meta_json VARCHAR,
                    submitted_at TIMESTAMP WITH TIME ZONE NOT NULL,
                    started_at TIMESTAMP WITH TIME ZONE,
                    completed_at TIMESTAMP WITH TIME ZONE,
                    output_timing_seconds DOUBLE,
                    output_warnings_json VARCHAR,
                    output_metadata_json VARCHAR,
                    error VARCHAR
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS checks (
                    id INTEGER PRIMARY KEY DEFAULT(nextval('checks_seq')),
                    artifact_id VARCHAR NOT NULL,
                    run_id VARCHAR,
                    check_name VARCHAR NOT NULL,
                    state VARCHAR NOT NULL,
                    message VARCHAR,
                    checked_at TIMESTAMP WITH TIME ZONE NOT NULL,
                    FOREIGN KEY (artifact_id) REFERENCES artifacts(id),
                    FOREIGN KEY (run_id) REFERENCES runs(id)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS lineage (
                    parent_id VARCHAR NOT NULL,
                    child_id VARCHAR NOT NULL,
                    operation VARCHAR NOT NULL,
                    run_id VARCHAR,
                    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
                    PRIMARY KEY (parent_id, child_id, operation),
                    FOREIGN KEY (parent_id) REFERENCES artifacts(id),
                    FOREIGN KEY (child_id) REFERENCES artifacts(id),
                    FOREIGN KEY (run_id) REFERENCES runs(id)
                )
            """)
            self._ensure_column(conn, "artifacts", "lineage_operation", "VARCHAR")
            self._ensure_column(conn, "artifacts", "lineage_inputs_json", "VARCHAR")
            self._ensure_column(conn, "artifacts", "lineage_params_json", "VARCHAR")
            self._ensure_column(conn, "artifacts", "lineage_timestamp", "TIMESTAMP WITH TIME ZONE")
            self._ensure_column(conn, "artifacts", "lineage_executor_id", "VARCHAR")
            self._ensure_column(conn, "runs", "output_timing_seconds", "DOUBLE")
            self._ensure_column(conn, "runs", "output_warnings_json", "VARCHAR")
            self._ensure_column(conn, "runs", "output_metadata_json", "VARCHAR")
        finally:
            conn.close()

    def _ensure_column(
        self,
        conn: duckdb.DuckDBPyConnection,
        table: str,
        column: str,
        column_type: str,
    ) -> None:
        """Apply additive schema migration for older local registries."""
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info('{table}')").fetchall()}
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")

    # -----------------------------------------------------------------------
    # Artifact operations
    # -----------------------------------------------------------------------

    def save_artifact(self, artifact: Artifact) -> None:
        """Persist an artifact to the registry."""
        conn = self._connect()
        try:
            self._save_artifact_conn(conn, artifact)
        finally:
            conn.close()

    def get_artifact(self, artifact_id: str) -> Artifact | None:
        """Load an artifact by ID, including its checks from the checks table."""
        conn = self._connect()
        try:
            row = conn.execute("SELECT * FROM artifacts WHERE id = ?", [artifact_id]).fetchone()
            if row is None:
                return None

            columns = [desc[0] for desc in conn.description]
            data = dict(zip(columns, row))

            # Load checks separately
            check_rows = conn.execute(
                "SELECT check_name, state, message, checked_at FROM checks "
                "WHERE artifact_id = ? ORDER BY checked_at",
                [artifact_id],
            ).fetchall()
        finally:
            conn.close()

        return self._row_to_artifact(data, check_rows)

    def list_artifacts(
        self,
        artifact_type: ArtifactType | None = None,
        limit: int = 100,
    ) -> list[Artifact]:
        """List artifacts, optionally filtered by type."""
        conn = self._connect()
        try:
            if artifact_type:
                rows = conn.execute(
                    "SELECT * FROM artifacts WHERE type = ? ORDER BY created_at DESC LIMIT ?",
                    [artifact_type.value, limit],
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM artifacts ORDER BY created_at DESC LIMIT ?",
                    [limit],
                ).fetchall()

            if not rows:
                return []

            columns = [desc[0] for desc in conn.description]
            artifacts = []
            for row in rows:
                data = dict(zip(columns, row))
                # Load checks for each artifact
                check_rows = conn.execute(
                    "SELECT check_name, state, message, checked_at FROM checks "
                    "WHERE artifact_id = ? ORDER BY checked_at",
                    [data["id"]],
                ).fetchall()
                artifacts.append(self._row_to_artifact(data, check_rows))
        finally:
            conn.close()

        return artifacts

    # -----------------------------------------------------------------------
    # Run operations
    # -----------------------------------------------------------------------

    def save_run(self, record: RunRecord) -> None:
        """Persist a run record. Also saves output artifact and checks if present."""
        conn = self._connect()
        try:
            conn.execute("BEGIN TRANSACTION")
            self._save_run_conn(conn, record)
            self._delete_run_checks_conn(conn, record.id)

            if record.output:
                self._save_artifact_conn(conn, record.output.artifact)

                for input_id in record.input_ids:
                    self._save_lineage_conn(
                        conn,
                        parent_id=input_id,
                        child_id=record.output.artifact.id,
                        operation=record.operator_name,
                        run_id=record.id,
                    )

                for check in record.checks:
                    self._save_check_conn(
                        conn,
                        artifact_id=record.output.artifact.id,
                        check=check,
                        run_id=record.id,
                    )

            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    def get_run(self, run_id: str) -> RunRecord | None:
        """Load a run record by ID."""
        conn = self._connect()
        try:
            row = conn.execute("SELECT * FROM runs WHERE id = ?", [run_id]).fetchone()
            if row is None:
                return None
            columns = [desc[0] for desc in conn.description]
            data = dict(zip(columns, row))

            # Load checks for this run
            check_rows = conn.execute(
                "SELECT check_name, state, message, checked_at FROM checks "
                "WHERE run_id = ? ORDER BY checked_at",
                [run_id],
            ).fetchall()
        finally:
            conn.close()

        return self._row_to_run(data, check_rows)

    def list_runs(
        self,
        status: RunStatus | None = None,
        limit: int = 100,
    ) -> list[RunRecord]:
        """List runs, optionally filtered by status."""
        conn = self._connect()
        try:
            if status:
                rows = conn.execute(
                    "SELECT * FROM runs WHERE status = ? ORDER BY submitted_at DESC LIMIT ?",
                    [status.value, limit],
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM runs ORDER BY submitted_at DESC LIMIT ?",
                    [limit],
                ).fetchall()

            if not rows:
                return []

            columns = [desc[0] for desc in conn.description]
            records = []
            for row in rows:
                data = dict(zip(columns, row))
                check_rows = conn.execute(
                    "SELECT check_name, state, message, checked_at FROM checks "
                    "WHERE run_id = ? ORDER BY checked_at",
                    [data["id"]],
                ).fetchall()
                records.append(self._row_to_run(data, check_rows))
        finally:
            conn.close()

        return records

    # -----------------------------------------------------------------------
    # Check operations
    # -----------------------------------------------------------------------

    def save_check(
        self,
        artifact_id: str,
        check: CheckResult,
        run_id: str | None = None,
    ) -> None:
        """Persist a check result. Links to artifact and optionally to a run."""
        conn = self._connect()
        try:
            self._save_check_conn(conn, artifact_id, check, run_id)
        finally:
            conn.close()

    def get_checks(
        self,
        artifact_id: str | None = None,
        run_id: str | None = None,
    ) -> list[CheckResult]:
        """Get checks filtered by artifact and/or run."""
        conditions = []
        params = []
        if artifact_id:
            conditions.append("artifact_id = ?")
            params.append(artifact_id)
        if run_id:
            conditions.append("run_id = ?")
            params.append(run_id)

        where = " AND ".join(conditions) if conditions else "1=1"

        conn = self._connect()
        try:
            rows = conn.execute(
                f"SELECT check_name, state, message, checked_at FROM checks "
                f"WHERE {where} ORDER BY checked_at",
                params,
            ).fetchall()
        finally:
            conn.close()

        return [
            CheckResult(
                check_name=r[0],
                state=ValidationState(r[1]),
                message=r[2] or "",
                timestamp=r[3],
            )
            for r in rows
        ]

    # -----------------------------------------------------------------------
    # Lineage operations
    # -----------------------------------------------------------------------

    def save_lineage(
        self,
        parent_id: str,
        child_id: str,
        operation: str,
        run_id: str | None = None,
    ) -> None:
        """Record a lineage edge (parent→child via operation)."""
        conn = self._connect()
        try:
            self._save_lineage_conn(conn, parent_id, child_id, operation, run_id)
        finally:
            conn.close()

    def get_parents(self, artifact_id: str) -> list[dict]:
        """Get immediate parent artifacts of an artifact."""
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT l.parent_id, l.operation, l.run_id, a.name, a.type
                FROM lineage l
                JOIN artifacts a ON a.id = l.parent_id
                WHERE l.child_id = ?
                ORDER BY l.created_at
                """,
                [artifact_id],
            ).fetchall()
        finally:
            conn.close()

        return [
            {
                "artifact_id": r[0],
                "operation": r[1],
                "run_id": r[2],
                "name": r[3],
                "type": r[4],
            }
            for r in rows
        ]

    def get_children(self, artifact_id: str) -> list[dict]:
        """Get immediate child artifacts derived from an artifact."""
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT l.child_id, l.operation, l.run_id, a.name, a.type
                FROM lineage l
                JOIN artifacts a ON a.id = l.child_id
                WHERE l.parent_id = ?
                ORDER BY l.created_at
                """,
                [artifact_id],
            ).fetchall()
        finally:
            conn.close()

        return [
            {
                "artifact_id": r[0],
                "operation": r[1],
                "run_id": r[2],
                "name": r[3],
                "type": r[4],
            }
            for r in rows
        ]

    def get_full_lineage(self, artifact_id: str) -> list[dict]:
        """Walk the full ancestor chain of an artifact (recursive)."""
        visited = set()
        chain = []
        self._walk_ancestors(artifact_id, visited, chain)
        return chain

    def _walk_ancestors(self, artifact_id: str, visited: set, chain: list) -> None:
        if artifact_id in visited:
            return
        visited.add(artifact_id)
        parents = self.get_parents(artifact_id)
        for p in parents:
            chain.append(p)
            self._walk_ancestors(p["artifact_id"], visited, chain)

    # -----------------------------------------------------------------------
    # Summary / stats
    # -----------------------------------------------------------------------

    def stats(self) -> dict:
        """Get registry statistics."""
        conn = self._connect()
        try:
            artifact_row = conn.execute("SELECT COUNT(*) FROM artifacts").fetchone()
            run_row = conn.execute("SELECT COUNT(*) FROM runs").fetchone()
            check_row = conn.execute("SELECT COUNT(*) FROM checks").fetchone()
            lineage_row = conn.execute("SELECT COUNT(*) FROM lineage").fetchone()
            if artifact_row is None or run_row is None or check_row is None or lineage_row is None:
                raise RuntimeError("Registry stats query returned no rows")

            artifact_count = artifact_row[0]
            run_count = run_row[0]
            check_count = check_row[0]
            lineage_count = lineage_row[0]

            type_counts = conn.execute(
                "SELECT type, COUNT(*) FROM artifacts GROUP BY type"
            ).fetchall()
            status_counts = conn.execute(
                "SELECT status, COUNT(*) FROM runs GROUP BY status"
            ).fetchall()
        finally:
            conn.close()

        return {
            "artifacts": artifact_count,
            "runs": run_count,
            "checks": check_count,
            "lineage_edges": lineage_count,
            "artifact_types": dict(type_counts),
            "run_statuses": dict(status_counts),
        }

    # -----------------------------------------------------------------------
    # Deserialization helpers
    # -----------------------------------------------------------------------

    def _row_to_artifact(self, data: dict, check_rows: list) -> Artifact:
        """Reconstruct an Artifact from a DB row + check rows."""
        backing = None
        if data.get("backing_kind"):
            backing = BackingStore(
                kind=BackingStoreKind(data["backing_kind"]),
                uri=data["backing_uri"] or "",
                size_bytes=data.get("backing_size_bytes"),
                content_hash=data.get("backing_content_hash"),
            )

        extent = None
        if data.get("extent_xmin") is not None:
            extent = (
                data["extent_xmin"],
                data["extent_ymin"],
                data["extent_xmax"],
                data["extent_ymax"],
            )

        resolution = None
        if data.get("resolution_x") is not None:
            resolution = (data["resolution_x"], data["resolution_y"])

        checks = [
            CheckResult(
                check_name=cr[0],
                state=ValidationState(cr[1]),
                message=cr[2] or "",
                timestamp=cr[3],
            )
            for cr in check_rows
        ]

        metadata = {}
        if data.get("metadata_json"):
            metadata = json.loads(data["metadata_json"])

        lineage = None
        if data.get("lineage_operation"):
            lineage = Lineage(
                operation=data["lineage_operation"],
                inputs=tuple(json.loads(data["lineage_inputs_json"]))
                if data.get("lineage_inputs_json")
                else (),
                params=json.loads(data["lineage_params_json"])
                if data.get("lineage_params_json")
                else {},
                timestamp=data.get("lineage_timestamp") or data["created_at"],
                executor_id=data.get("lineage_executor_id"),
            )

        return Artifact(
            id=data["id"],
            type=ArtifactType(data["type"]),
            name=data["name"],
            backing=backing,
            spatial=SpatialDescriptor(
                crs=data.get("crs"),
                extent=extent,
                resolution=resolution,
                feature_count=data.get("feature_count"),
                band_count=data.get("band_count"),
            ),
            lineage=lineage,
            checks=tuple(checks),
            metadata=metadata,
            created_at=data["created_at"],
        )

    def _row_to_run(self, data: dict, check_rows: list) -> RunRecord:
        """Reconstruct a RunRecord from a DB row + check rows."""
        checks = [
            CheckResult(
                check_name=cr[0],
                state=ValidationState(cr[1]),
                message=cr[2] or "",
                timestamp=cr[3],
            )
            for cr in check_rows
        ]

        # Reconstruct output OperatorResult from stored artifact ID
        output = None
        output_artifact_id = data.get("output_artifact_id")
        if output_artifact_id:
            artifact = self.get_artifact(output_artifact_id)
            if artifact:
                output = OperatorResult(
                    artifact=artifact,
                    checks=checks,
                    warnings=json.loads(data["output_warnings_json"])
                    if data.get("output_warnings_json")
                    else [],
                    timing_seconds=data.get("output_timing_seconds"),
                    metadata=json.loads(data["output_metadata_json"])
                    if data.get("output_metadata_json")
                    else {},
                )

        return RunRecord(
            id=data["id"],
            operator_name=data["operator_name"],
            status=RunStatus(data["status"]),
            input_ids=json.loads(data["input_ids_json"]) if data.get("input_ids_json") else [],
            params=json.loads(data["params_json"]) if data.get("params_json") else {},
            output=output,
            executor_name=data.get("executor_name") or "",
            executor_meta=json.loads(data["executor_meta_json"])
            if data.get("executor_meta_json")
            else {},
            submitted_at=data["submitted_at"],
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            error=data.get("error"),
        )

    def _save_artifact_conn(self, conn: duckdb.DuckDBPyConnection, artifact: Artifact) -> None:
        """Persist an artifact using an existing connection."""
        extent = artifact.spatial.extent
        resolution = artifact.spatial.resolution
        lineage = artifact.lineage
        conn.execute(
            """
            INSERT OR REPLACE INTO artifacts (
                id, type, name,
                backing_kind, backing_uri, backing_size_bytes, backing_content_hash,
                crs, extent_xmin, extent_ymin, extent_xmax, extent_ymax,
                resolution_x, resolution_y, feature_count, band_count,
                lineage_operation, lineage_inputs_json, lineage_params_json,
                lineage_timestamp, lineage_executor_id,
                metadata_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                artifact.id,
                artifact.type.value,
                artifact.name,
                artifact.backing.kind.value if artifact.backing else None,
                artifact.backing.uri if artifact.backing else None,
                artifact.backing.size_bytes if artifact.backing else None,
                artifact.backing.content_hash if artifact.backing else None,
                artifact.spatial.crs,
                extent[0] if extent else None,
                extent[1] if extent else None,
                extent[2] if extent else None,
                extent[3] if extent else None,
                resolution[0] if resolution else None,
                resolution[1] if resolution else None,
                artifact.spatial.feature_count,
                artifact.spatial.band_count,
                lineage.operation if lineage else None,
                json.dumps(list(lineage.inputs)) if lineage else None,
                json.dumps(_to_jsonable(lineage.params)) if lineage else None,
                lineage.timestamp if lineage else None,
                lineage.executor_id if lineage else None,
                json.dumps(_to_jsonable(artifact.metadata)) if artifact.metadata else None,
                artifact.created_at,
            ],
        )

    def _save_run_conn(self, conn: duckdb.DuckDBPyConnection, record: RunRecord) -> None:
        """Persist a run using an existing connection."""
        conn.execute(
            """
            INSERT OR REPLACE INTO runs (
                id, operator_name, status, input_ids_json,
                output_artifact_id, params_json,
                executor_name, executor_meta_json,
                submitted_at, started_at, completed_at,
                output_timing_seconds, output_warnings_json, output_metadata_json,
                error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                record.id,
                record.operator_name,
                record.status.value,
                json.dumps(record.input_ids),
                record.output.artifact.id if record.output else None,
                json.dumps(_to_jsonable(record.params)) if record.params else None,
                record.executor_name,
                json.dumps(_to_jsonable(record.executor_meta)) if record.executor_meta else None,
                record.submitted_at,
                record.started_at,
                record.completed_at,
                record.output.timing_seconds if record.output else None,
                json.dumps(_to_jsonable(record.output.warnings))
                if record.output and record.output.warnings
                else None,
                json.dumps(_to_jsonable(record.output.metadata))
                if record.output and record.output.metadata
                else None,
                record.error,
            ],
        )

    def _save_check_conn(
        self,
        conn: duckdb.DuckDBPyConnection,
        artifact_id: str,
        check: CheckResult,
        run_id: str | None = None,
    ) -> None:
        """Persist a check using an existing connection."""
        conn.execute(
            """
            INSERT INTO checks (artifact_id, run_id, check_name, state, message, checked_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                artifact_id,
                run_id,
                check.check_name,
                check.state.value,
                check.message,
                check.timestamp,
            ],
        )

    def _delete_run_checks_conn(
        self,
        conn: duckdb.DuckDBPyConnection,
        run_id: str,
    ) -> None:
        """Replace semantics for run-linked checks during save_run()."""
        conn.execute("DELETE FROM checks WHERE run_id = ?", [run_id])

    def _save_lineage_conn(
        self,
        conn: duckdb.DuckDBPyConnection,
        parent_id: str,
        child_id: str,
        operation: str,
        run_id: str | None = None,
    ) -> None:
        """Persist a lineage edge using an existing connection."""
        conn.execute(
            """
            INSERT OR IGNORE INTO lineage (parent_id, child_id, operation, run_id, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            [parent_id, child_id, operation, run_id, datetime.now(tz=timezone.utc)],
        )

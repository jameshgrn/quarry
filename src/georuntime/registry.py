"""Artifact registry backed by DuckDB."""

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import duckdb

VALID_ARTIFACT_TYPES = {"vector", "raster", "table", "preview", "summary"}


class Registry:
    """Append-only artifact registry."""

    def __init__(self, workspace: str | None = None):
        if workspace is None:
            workspace = os.getcwd()
        self.workspace = Path(workspace)
        self.db_path = self.workspace / ".georuntime" / "registry.duckdb"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Create the artifacts table if it doesn't exist."""
        conn = duckdb.connect(str(self.db_path))
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS artifacts (
                    id VARCHAR PRIMARY KEY,
                    name VARCHAR,
                    artifact_type VARCHAR,
                    path VARCHAR,
                    crs VARCHAR,
                    extent_json VARCHAR,
                    band_count INTEGER,
                    feature_count INTEGER,
                    driver VARCHAR,
                    created_at TIMESTAMP,
                    source_operation VARCHAR,
                    source_inputs_json VARCHAR,
                    metadata_json VARCHAR
                )
            """)
        finally:
            conn.close()

    def register(self, artifact: dict) -> str:
        """Register an artifact in the registry.

        Args:
            artifact: Dict with keys matching the artifacts table columns.
                     Required: name, artifact_type, path
                     Optional: crs, extent, band_count, feature_count, driver,
                              source_operation, source_inputs, metadata

        Returns:
            The generated artifact ID (UUID string).

        Raises:
            ValueError: If artifact_type is not valid.
        """
        artifact_type = artifact.get("artifact_type")
        if artifact_type not in VALID_ARTIFACT_TYPES:
            raise ValueError(
                f"Invalid artifact_type: {artifact_type}. Must be one of: {VALID_ARTIFACT_TYPES}"
            )

        artifact_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc)

        # Convert dict/list fields to JSON strings
        extent_json = json.dumps(artifact.get("extent")) if artifact.get("extent") else None
        source_inputs_json = (
            json.dumps(artifact.get("source_inputs")) if artifact.get("source_inputs") else None
        )
        metadata_json = json.dumps(artifact.get("metadata")) if artifact.get("metadata") else None

        # Store absolute path
        path = Path(artifact["path"]).resolve()

        conn = duckdb.connect(str(self.db_path))
        try:
            conn.execute(
                """
                INSERT INTO artifacts (
                    id, name, artifact_type, path, crs, extent_json,
                    band_count, feature_count, driver, created_at,
                    source_operation, source_inputs_json, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    artifact_id,
                    artifact["name"],
                    artifact_type,
                    str(path),
                    artifact.get("crs"),
                    extent_json,
                    artifact.get("band_count"),
                    artifact.get("feature_count"),
                    artifact.get("driver"),
                    created_at,
                    artifact.get("source_operation"),
                    source_inputs_json,
                    metadata_json,
                ],
            )
        finally:
            conn.close()

        return artifact_id

    def get(self, artifact_id: str) -> dict | None:
        """Get an artifact by ID."""
        conn = duckdb.connect(str(self.db_path))
        try:
            row = conn.execute("SELECT * FROM artifacts WHERE id = ?", [artifact_id]).fetchone()
        finally:
            conn.close()

        if row is None:
            return None

        return self._row_to_dict(row)

    def list(self, artifact_type: str | None = None) -> list[dict]:
        """List artifacts, optionally filtered by type."""
        conn = duckdb.connect(str(self.db_path))
        try:
            if artifact_type:
                rows = conn.execute(
                    "SELECT * FROM artifacts WHERE artifact_type = ? ORDER BY created_at DESC",
                    [artifact_type],
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM artifacts ORDER BY created_at DESC").fetchall()
        finally:
            conn.close()

        return [self._row_to_dict(row) for row in rows]

    def exists(self, path: str) -> bool:
        """Check if an artifact exists at the given path."""
        abs_path = str(Path(path).resolve())
        conn = duckdb.connect(str(self.db_path))
        try:
            result = conn.execute(
                "SELECT COUNT(*) FROM artifacts WHERE path = ?", [abs_path]
            ).fetchone()
        finally:
            conn.close()

        return result[0] > 0

    def _row_to_dict(self, row) -> dict:
        """Convert a DuckDB row to a dictionary."""
        columns = [
            "id",
            "name",
            "artifact_type",
            "path",
            "crs",
            "extent_json",
            "band_count",
            "feature_count",
            "driver",
            "created_at",
            "source_operation",
            "source_inputs_json",
            "metadata_json",
        ]

        result = dict(zip(columns, row))

        # Parse JSON fields back to Python objects
        for key in ["extent_json", "source_inputs_json", "metadata_json"]:
            if result.get(key):
                try:
                    result[key.replace("_json", "")] = json.loads(result[key])
                except json.JSONDecodeError:
                    pass  # Leave as string if parsing fails

        return result

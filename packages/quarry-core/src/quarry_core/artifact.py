"""Artifact — the internal unit of truth.

An artifact is NOT a file. It is a typed, tracked, validated unit of geospatial data
with identity independent of its storage location.

Identity = id + type + lineage.
Storage = backing store descriptor (could be path, URI, lazy handle, or nothing yet).
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


class ArtifactType(Enum):
    """Canonical artifact types."""

    RASTER = "raster"
    VECTOR = "vector"
    TABLE = "table"
    TEMPORAL_STACK = "temporal_stack"
    TILE_SET = "tile_set"
    MODEL = "model"
    POINT_CLOUD = "point_cloud"


class BackingStoreKind(Enum):
    """How an artifact is physically stored."""

    LOCAL_FILE = "local_file"
    REMOTE_URI = "remote_uri"
    LAZY_HANDLE = "lazy_handle"  # metadata known, data not yet fetched
    MEMORY = "memory"  # in-process only, not persisted
    DUCKDB = "duckdb"  # table stored in DuckDB
    POSTGIS = "postgis"  # table/layer in PostGIS


class ValidationState(Enum):
    """Artifact validation status."""

    UNCHECKED = "unchecked"
    VALID = "valid"
    INVALID = "invalid"
    WARN = "warn"


@dataclass(frozen=True)
class BackingStore:
    """Describes where an artifact's data physically lives.

    This is NOT the artifact's identity. It's a descriptor of one possible
    materialization of the artifact's data.
    """

    kind: BackingStoreKind
    uri: str  # path, URL, connection string, or handle reference
    size_bytes: int | None = None
    content_hash: str | None = None  # SHA-256 of content if known


@dataclass(frozen=True)
class SpatialDescriptor:
    """Spatial properties of an artifact."""

    crs: str | None = None  # e.g. "EPSG:4326"
    extent: tuple[float, float, float, float] | None = None  # xmin, ymin, xmax, ymax
    resolution: tuple[float, float] | None = None  # x_res, y_res (rasters)
    feature_count: int | None = None  # vectors/tables
    band_count: int | None = None  # rasters


@dataclass(frozen=True)
class Lineage:
    """How this artifact came into existence."""

    operation: str  # e.g. "clip", "reproject", "materialize"
    inputs: tuple[str, ...] = ()  # artifact IDs that fed this operation
    params: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    executor_id: str | None = None  # which executor ran this


@dataclass(frozen=True)
class CheckResult:
    """Result of a validation check on an artifact."""

    check_name: str  # e.g. "crs_valid", "extent_sane", "no_nodata_explosion"
    state: ValidationState
    message: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class Artifact:
    """The canonical internal unit of truth in Quarry.

    Identity is the `id` field. Not the path. Not the name.
    An artifact can exist with a LAZY backing store (metadata known, data not fetched).
    An artifact carries its own validation history.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: ArtifactType = ArtifactType.RASTER
    name: str = ""

    # Where the data lives (or will live)
    backing: BackingStore | None = None

    # Spatial properties
    spatial: SpatialDescriptor = field(default_factory=SpatialDescriptor)

    # How it was created
    lineage: Lineage | None = None

    # Validation state
    checks: list[CheckResult] = field(default_factory=list)

    # Extensible metadata bag — for driver info, tags, domain-specific fields
    metadata: dict[str, Any] = field(default_factory=dict)

    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def validation_state(self) -> ValidationState:
        """Overall validation state derived from check results."""
        if not self.checks:
            return ValidationState.UNCHECKED
        states = [c.state for c in self.checks]
        if ValidationState.INVALID in states:
            return ValidationState.INVALID
        if ValidationState.WARN in states:
            return ValidationState.WARN
        return ValidationState.VALID

    @property
    def is_materialized(self) -> bool:
        """Whether the artifact's data is locally accessible."""
        if self.backing is None:
            return False
        return self.backing.kind in (
            BackingStoreKind.LOCAL_FILE,
            BackingStoreKind.MEMORY,
            BackingStoreKind.DUCKDB,
        )

    def with_check(self, result: CheckResult) -> Artifact:
        """Return a new artifact with an additional check result."""
        new_checks = [*self.checks, result]
        # Dataclass isn't frozen so we can mutate, but prefer explicit copy pattern
        return Artifact(
            id=self.id,
            type=self.type,
            name=self.name,
            backing=self.backing,
            spatial=self.spatial,
            lineage=self.lineage,
            checks=new_checks,
            metadata=self.metadata,
            created_at=self.created_at,
        )


def content_hash(path: Path) -> str:
    """Compute SHA-256 hash of a file's contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

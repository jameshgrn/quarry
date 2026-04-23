"""Connector — the canonical gateway from outside world to artifact.

No geospatial object enters the system except through a connector.
This is the deepest rule.

A connector's ONLY hard requirement is `materialize`: given a source reference,
produce a canonical Artifact. Everything else is an optional capability.

Materialize does NOT always mean "download." It can mean:
- Copy local data
- Stage a remote asset
- Wrap an already-local path
- Partially fetch metadata + lazy handle
- Normalize weird formats into internal backing form
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Flag, auto
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from quarry_core.artifact import Artifact

if TYPE_CHECKING:
    from quarry_core.source_ref import SourceRef

# ---------------------------------------------------------------------------
# Capability flags
# ---------------------------------------------------------------------------


class ConnectorCapability(Flag):
    """Explicit capabilities a connector may declare."""

    MATERIALIZE = auto()  # Always required (enforced by protocol)
    DISCOVER = auto()  # Can browse/search available data
    AUTHENTICATE = auto()  # Needs/supports auth setup
    STREAM = auto()  # Can stream data without full materialization
    MATERIALIZE_LAZY = auto()  # Can produce lazy-handle artifacts
    METADATA_ONLY = auto()  # Can emit metadata without fetching data


# ---------------------------------------------------------------------------
# Supporting types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CatalogEntry:
    """A discoverable item from a connector's catalog.

    This is what `discover` returns — a reference that can be passed to `materialize`.
    """

    source_ref: SourceRef | str  # The reference materialize needs (path, URL, asset ID, etc.)
    name: str = ""
    description: str = ""
    spatial_hint: dict[str, Any] = field(default_factory=dict)  # CRS, extent if known
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MaterializeResult:
    """What materialize returns: the artifact plus provenance about how it was materialized."""

    artifact: Artifact
    strategy: str  # "copied", "wrapped_local", "fetched_remote", "lazy_handle", "normalized"
    source_ref: SourceRef | str  # Original source reference
    notes: str = ""  # Any additional context


# ---------------------------------------------------------------------------
# The Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Connector(Protocol):
    """The canonical gateway protocol.

    ONLY `materialize` is required. All other methods are optional capabilities.
    Connectors declare their capabilities via the `capabilities` property.
    """

    @property
    def name(self) -> str:
        """Human-readable connector name."""
        ...

    @property
    def capabilities(self) -> ConnectorCapability:
        """Declare what this connector can do beyond materialize."""
        ...

    def materialize(
        self,
        source_ref: SourceRef | str,
        workspace: Path,
        *,
        lazy: bool = False,
    ) -> MaterializeResult:
        """The sacred gateway. Source reference in, canonical artifact out.

        Args:
            source_ref: What to materialize (path, URL, asset ID, query, etc.)
            workspace: Where to put materialized data if needed.
            lazy: If True and connector supports it, produce a lazy-handle artifact
                  (metadata known, data not yet fetched).

        Returns:
            MaterializeResult with the artifact and provenance.

        Raises:
            MaterializeError: If materialization fails.
        """
        ...


# ---------------------------------------------------------------------------
# Optional capability protocols (mix in as needed)
# ---------------------------------------------------------------------------


@runtime_checkable
class Discoverable(Protocol):
    """A connector that can browse/search for available data."""

    def discover(self, query: str | dict[str, Any] | None = None) -> list[CatalogEntry]:
        """Search or list available data.

        Args:
            query: Search string, filter dict, or None for "list all."

        Returns:
            List of catalog entries that can be passed to materialize.
        """
        ...


@runtime_checkable
class Authenticatable(Protocol):
    """A connector that requires authentication."""

    def authenticate(self, credentials: dict[str, Any]) -> None:
        """Set up authentication context.

        Args:
            credentials: Auth info (API key, token, username/password, etc.)

        Raises:
            AuthError: If authentication fails.
        """
        ...


@runtime_checkable
class MetadataEmitter(Protocol):
    """A connector that can emit metadata without fetching data."""

    def metadata(self, source_ref: SourceRef | str) -> dict[str, Any]:
        """Get metadata about a source without materializing it.

        Args:
            source_ref: What to inspect.

        Returns:
            Metadata dict (CRS, extent, schema, etc.)
        """
        ...


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ConnectorError(Exception):
    """Base error for connector operations."""


class MaterializeError(ConnectorError):
    """Materialization failed."""

    def __init__(self, source_ref: SourceRef | str, reason: str):
        self.source_ref = source_ref
        self.reason = reason
        super().__init__(f"Failed to materialize '{source_ref}': {reason}")


class AuthError(ConnectorError):
    """Authentication failed."""

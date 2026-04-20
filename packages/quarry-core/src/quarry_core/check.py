"""Check — validation layer for artifacts and runs.

Every operator declares checks. Every artifact carries validation state.
Every run emits pass/warn/fail signals.

Checks are not afterthoughts. They are how the system knows it produced
sane spatial truth, not just "ran successfully."
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from quarry_core.artifact import Artifact, CheckResult, ValidationState

# ---------------------------------------------------------------------------
# The Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Check(Protocol):
    """A validation rule applied to an artifact."""

    @property
    def name(self) -> str:
        """Check name (e.g. 'crs_valid', 'extent_sane', 'no_nodata_explosion')."""
        ...

    @property
    def description(self) -> str:
        """Human-readable description of what this check validates."""
        ...

    def run(self, artifact: Artifact) -> CheckResult:
        """Execute the check against an artifact.

        Args:
            artifact: The artifact to validate.

        Returns:
            CheckResult with state (valid/invalid/warn) and message.
        """
        ...


# ---------------------------------------------------------------------------
# Common checks (concrete implementations)
# ---------------------------------------------------------------------------


class CRSValid:
    """Check that artifact has a non-null, parseable CRS."""

    @property
    def name(self) -> str:
        return "crs_valid"

    @property
    def description(self) -> str:
        return "Artifact has a defined coordinate reference system"

    def run(self, artifact: Artifact) -> CheckResult:
        if artifact.spatial.crs is None:
            return CheckResult(
                check_name=self.name,
                state=ValidationState.INVALID,
                message="No CRS defined",
            )
        if artifact.spatial.crs.strip() == "":
            return CheckResult(
                check_name=self.name,
                state=ValidationState.INVALID,
                message="CRS is empty string",
            )
        return CheckResult(
            check_name=self.name,
            state=ValidationState.VALID,
            message=f"CRS: {artifact.spatial.crs}",
        )


class ExtentSane:
    """Check that artifact extent is non-degenerate and within plausible bounds."""

    @property
    def name(self) -> str:
        return "extent_sane"

    @property
    def description(self) -> str:
        return "Artifact extent is non-zero and within plausible geographic bounds"

    def run(self, artifact: Artifact) -> CheckResult:
        ext = artifact.spatial.extent
        if ext is None:
            return CheckResult(
                check_name=self.name,
                state=ValidationState.WARN,
                message="No extent defined",
            )

        xmin, ymin, xmax, ymax = ext

        # Degenerate check
        if xmin >= xmax or ymin >= ymax:
            return CheckResult(
                check_name=self.name,
                state=ValidationState.INVALID,
                message=f"Degenerate extent: ({xmin}, {ymin}, {xmax}, {ymax})",
            )

        # Plausibility check (WGS84 bounds — warn if outside, might be projected)
        if abs(xmin) > 180 or abs(xmax) > 180 or abs(ymin) > 90 or abs(ymax) > 90:
            # Could be valid projected coordinates — warn, don't fail
            if artifact.spatial.crs and "4326" not in artifact.spatial.crs:
                return CheckResult(
                    check_name=self.name,
                    state=ValidationState.VALID,
                    message="Extent exceeds WGS84 bounds (projected CRS, acceptable)",
                )
            return CheckResult(
                check_name=self.name,
                state=ValidationState.WARN,
                message=f"Extent exceeds WGS84 bounds: ({xmin}, {ymin}, {xmax}, {ymax})",
            )

        return CheckResult(
            check_name=self.name,
            state=ValidationState.VALID,
            message=f"Extent OK: ({xmin:.4f}, {ymin:.4f}, {xmax:.4f}, {ymax:.4f})",
        )


class BackingStoreAccessible:
    """Check that the artifact's backing store is actually reachable."""

    @property
    def name(self) -> str:
        return "backing_accessible"

    @property
    def description(self) -> str:
        return "Artifact's backing store exists and is accessible"

    def run(self, artifact: Artifact) -> CheckResult:
        from pathlib import Path

        if artifact.backing is None:
            return CheckResult(
                check_name=self.name,
                state=ValidationState.INVALID,
                message="No backing store defined",
            )

        from quarry_core.artifact import BackingStoreKind

        if artifact.backing.kind == BackingStoreKind.LOCAL_FILE:
            path = Path(artifact.backing.uri)
            if not path.exists():
                return CheckResult(
                    check_name=self.name,
                    state=ValidationState.INVALID,
                    message=f"File not found: {artifact.backing.uri}",
                )
            if path.stat().st_size == 0:
                return CheckResult(
                    check_name=self.name,
                    state=ValidationState.WARN,
                    message=f"File is empty: {artifact.backing.uri}",
                )
            return CheckResult(
                check_name=self.name,
                state=ValidationState.VALID,
                message=f"Accessible: {artifact.backing.uri}",
            )

        if artifact.backing.kind == BackingStoreKind.LAZY_HANDLE:
            return CheckResult(
                check_name=self.name,
                state=ValidationState.WARN,
                message="Lazy handle — not yet materialized",
            )

        return CheckResult(
            check_name=self.name,
            state=ValidationState.VALID,
            message=f"Backing store kind: {artifact.backing.kind.value}",
        )

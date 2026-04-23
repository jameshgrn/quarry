"""Operator — a typed transformation on canonical artifacts.

Lane: operator

Operators only consume and emit canonical artifacts.
No operator operates on raw chaos unless it is explicitly a normalization operator.

Every operator:
- Declares accepted input artifact types
- Declares output artifact type
- Has typed params
- Declares checks it will run on output
- Can report resource profile (for executor scheduling)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from quarry_core.artifact import Artifact, ArtifactType, CheckResult

# ---------------------------------------------------------------------------
# Supporting types
# ---------------------------------------------------------------------------


class ResourceScale(Enum):
    """Rough resource profile for executor scheduling."""

    TRIVIAL = "trivial"  # < 1s, negligible memory
    LIGHT = "light"  # seconds, < 1GB memory
    MEDIUM = "medium"  # minutes, 1-8GB memory
    HEAVY = "heavy"  # 10+ minutes, 8GB+ memory
    MASSIVE = "massive"  # hours, cluster-scale


@dataclass(frozen=True)
class OperatorSpec:
    """Static declaration of what an operator accepts and produces."""

    input_types: tuple[ArtifactType, ...]  # Accepted input artifact types
    output_type: ArtifactType  # What it produces
    min_inputs: int = 1
    max_inputs: int = 1  # -1 for unbounded
    resource_scale: ResourceScale = ResourceScale.LIGHT


@dataclass(frozen=True)
class OperatorParams:
    """Base for operator parameters. Subclass per operator."""


@dataclass
class OperatorResult:
    """What an operator returns after execution."""

    artifact: Artifact
    checks: list[CheckResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    timing_seconds: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# The Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Operator(Protocol):
    """A typed transformation: artifacts in, artifact out."""

    @property
    def name(self) -> str:
        """Operator name (e.g. 'clip_raster', 'reproject_vector')."""
        ...

    @property
    def spec(self) -> OperatorSpec:
        """Static declaration of inputs/outputs/resources."""
        ...

    def validate_inputs(self, inputs: list[Artifact], params: OperatorParams) -> list[str]:
        """Check whether inputs are valid for this operator.

        Returns:
            List of error messages. Empty list = valid.
        """
        ...

    def execute(self, inputs: list[Artifact], params: OperatorParams) -> OperatorResult:
        """Run the transformation.

        Args:
            inputs: Canonical artifacts to operate on.
            params: Typed parameters for this operation.

        Returns:
            OperatorResult with the output artifact and check results.

        Raises:
            OperatorError: If execution fails.
        """
        ...

    def declared_checks(self) -> list[str]:
        """Names of checks this operator will run on its output.

        Returns:
            List of check names (e.g. ['crs_valid', 'extent_within_input']).
        """
        ...


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class OperatorError(Exception):
    """Base error for operator failures."""

    def __init__(self, operator_name: str, reason: str, inputs: list[str] | None = None):
        self.operator_name = operator_name
        self.reason = reason
        self.inputs = inputs or []
        super().__init__(f"Operator '{operator_name}' failed: {reason}")


class ValidationError(OperatorError):
    """Input validation failed."""

    def __init__(self, operator_name: str, errors: list[str]):
        self.errors = errors
        super().__init__(operator_name, f"Validation failed: {'; '.join(errors)}")

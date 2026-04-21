"""ConnectorRouter — selection layer for matching sources to connectors.

Lane: registry

Given a SourceRef (or raw string), returns ranked eligible connectors.
No execution — only selection. Connectors are registered with explicit
kind-affinity declarations so the router never guesses.

Design:
- Connectors declare what SourceRefKinds they handle via registration
- Router ranks candidates by specificity (exact kind match > fallback)
- Ambiguity is surfaced, not hidden — multiple matches are valid
- Raw strings are auto-inferred via SourceRef.infer()
- Zero external deps (quarry-core only)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from quarry_core.connector import Connector
from quarry_core.source_ref import SourceRef, SourceRefKind


class MatchReason(Enum):
    """Why a connector was selected."""

    KIND_MATCH = "kind_match"  # SourceRefKind in connector's declared kinds
    FALLBACK = "fallback"  # Connector registered as fallback for UNKNOWN


@dataclass(frozen=True)
class ConnectorMatch:
    """A ranked match of a connector to a source reference."""

    connector: Connector
    reason: MatchReason
    rank: int  # Lower = higher priority (0 is best)

    def __lt__(self, other: ConnectorMatch) -> bool:
        return self.rank < other.rank


class NoConnectorError(Exception):
    """No registered connector can handle this source."""

    def __init__(self, ref: SourceRef):
        self.ref = ref
        super().__init__(f"No connector registered for {ref.kind.value}: {ref.raw!r}")


@dataclass(frozen=True)
class _Registration:
    """Internal: a connector with its routing metadata."""

    connector: Connector
    kinds: frozenset[SourceRefKind]
    priority: int  # Lower = higher priority within same kind
    fallback: bool  # Accept UNKNOWN refs as last resort


class ConnectorRouter:
    """Select eligible connectors for a given source reference.

    Usage:
        router = ConnectorRouter()
        router.register(local_conn, kinds={SourceRefKind.LOCAL_PATH}, priority=10)
        kinds = {SourceRefKind.LOCAL_PATH, SourceRefKind.REMOTE_URI}
        router.register(cog_conn, kinds=kinds, priority=0)
        router.register(stac_conn, kinds={SourceRefKind.CATALOG_ITEM})
        router.register(pg_conn, kinds={SourceRefKind.DATABASE_REF})

        matches = router.select("s3://bucket/dem.tif")
        # → [ConnectorMatch(cog_conn, KIND_MATCH, rank=0)]

        matches = router.select("/data/dem.tif")
        # → [ConnectorMatch(cog_conn, KIND_MATCH, rank=0),
        #    ConnectorMatch(local_conn, KIND_MATCH, rank=10)]
    """

    def __init__(self) -> None:
        self._registrations: list[_Registration] = []

    def register(
        self,
        connector: Connector,
        *,
        kinds: set[SourceRefKind],
        priority: int = 5,
        fallback: bool = False,
    ) -> None:
        """Register a connector with its routing metadata.

        Args:
            connector: The connector instance.
            kinds: Which SourceRefKinds this connector handles.
            priority: Lower = preferred when multiple connectors match the same kind.
                      Default 5. COG might be 0, LocalFile 10.
            fallback: If True, this connector is also eligible for UNKNOWN refs.
        """
        self._registrations.append(
            _Registration(
                connector=connector,
                kinds=frozenset(kinds),
                priority=priority,
                fallback=fallback,
            )
        )

    def select(self, source: SourceRef | str) -> list[ConnectorMatch]:
        """Return ranked eligible connectors for a source reference.

        Args:
            source: A SourceRef or raw string (auto-inferred via SourceRef.infer).

        Returns:
            List of ConnectorMatch, sorted by rank (lowest first = best match).
            Empty list if no connectors match.
        """
        ref = SourceRef.infer(source) if isinstance(source, str) else source

        matches: list[ConnectorMatch] = []
        for reg in self._registrations:
            if ref.kind in reg.kinds:
                matches.append(
                    ConnectorMatch(
                        connector=reg.connector,
                        reason=MatchReason.KIND_MATCH,
                        rank=reg.priority,
                    )
                )
            elif ref.kind == SourceRefKind.UNKNOWN and reg.fallback:
                matches.append(
                    ConnectorMatch(
                        connector=reg.connector,
                        reason=MatchReason.FALLBACK,
                        rank=reg.priority + 1000,
                    )
                )

        matches.sort()
        return matches

    def select_one(self, source: SourceRef | str) -> ConnectorMatch:
        """Return the single best connector, or raise NoConnectorError.

        Convenience for callers that want exactly one answer.
        """
        ref = SourceRef.infer(source) if isinstance(source, str) else source
        matches = self.select(ref)
        if not matches:
            raise NoConnectorError(ref)
        return matches[0]

    @property
    def registrations(self) -> list[tuple[str, frozenset[SourceRefKind], int, bool]]:
        """Inspect registered connectors (for debugging/testing)."""
        return [(r.connector.name, r.kinds, r.priority, r.fallback) for r in self._registrations]

"""ConnectorRouter pressure test.

Lane: registry

Stress points:
1. Local GeoTIFF ambiguity — both COG and LocalFile match, COG wins on priority
2. Remote COG URI — only COG matches
3. STAC ref — only STAC matches
4. PostGIS ref — only PostGIS matches
5. Unknown/unsupported source — no match or fallback-only
6. Backward compat — raw strings auto-inferred via SourceRef.infer()
7. Priority ordering — lower priority number = higher rank
8. Multiple matches returned in rank order
9. select_one raises NoConnectorError when nothing matches
10. Fallback connectors only match UNKNOWN, ranked below kind matches
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from quarry_core.connector import Connector, ConnectorCapability, MaterializeResult
from quarry_core.router import ConnectorMatch, ConnectorRouter, MatchReason, NoConnectorError
from quarry_core.source_ref import SourceRef, SourceRefKind

# ---------------------------------------------------------------------------
# Stub connectors — minimal implementations for routing tests
# ---------------------------------------------------------------------------


@dataclass
class StubConnector:
    """Minimal connector that satisfies the protocol."""

    _name: str

    @property
    def name(self) -> str:
        return self._name

    @property
    def capabilities(self) -> ConnectorCapability:
        return ConnectorCapability.MATERIALIZE

    def materialize(
        self,
        source_ref: SourceRef | str,
        workspace: Path,
        *,
        lazy: bool = False,
    ) -> MaterializeResult:
        raise NotImplementedError("Stub — routing only, no execution")


def _make_router() -> tuple[
    ConnectorRouter,
    StubConnector,
    StubConnector,
    StubConnector,
    StubConnector,
]:
    """Build a standard router with all 4 connector types registered."""
    local = StubConnector("local_file")
    cog = StubConnector("cog")
    stac = StubConnector("stac")
    postgis = StubConnector("postgis")

    router = ConnectorRouter()
    router.register(
        cog,
        kinds={SourceRefKind.LOCAL_RASTER, SourceRefKind.REMOTE_URI},
        priority=0,
    )
    router.register(
        local,
        kinds={SourceRefKind.LOCAL_PATH, SourceRefKind.LOCAL_RASTER, SourceRefKind.LOCAL_VECTOR},
        priority=10,
        fallback=True,
    )
    router.register(
        stac,
        kinds={SourceRefKind.CATALOG_ITEM},
        priority=5,
    )
    router.register(
        postgis,
        kinds={SourceRefKind.DATABASE_REF},
        priority=5,
    )

    return router, local, cog, stac, postgis


# ---------------------------------------------------------------------------
# 1. Local GeoTIFF ambiguity — both COG and LocalFile match
# ---------------------------------------------------------------------------


class TestLocalGeoTIFFAmbiguity:
    """A local .tif should match both COG and LocalFile, COG ranked first."""

    def test_local_tif_matches_both(self):
        router, local, cog, _, _ = _make_router()
        matches = router.select(SourceRef.local("/data/dem.tif"))
        names = [m.connector.name for m in matches]
        assert "cog" in names
        assert "local_file" in names

    def test_cog_ranked_above_local(self):
        router, _, _, _, _ = _make_router()
        matches = router.select(SourceRef.local("/data/dem.tif"))
        assert matches[0].connector.name == "cog"
        assert matches[1].connector.name == "local_file"

    def test_both_are_kind_match(self):
        router, _, _, _, _ = _make_router()
        matches = router.select(SourceRef.local("/data/dem.tif"))
        assert all(m.reason == MatchReason.KIND_MATCH for m in matches)

    def test_select_one_returns_cog(self):
        router, _, _, _, _ = _make_router()
        best = router.select_one(SourceRef.local("/data/dem.tif"))
        assert best.connector.name == "cog"

    def test_local_geojson_matches_only_local_file(self):
        """Vector paths should not be eligible for raster-only connectors."""
        router, _, _, _, _ = _make_router()
        matches = router.select(SourceRef.local("/data/parcels.geojson"))
        assert len(matches) == 1
        assert matches[0].connector.name == "local_file"


# ---------------------------------------------------------------------------
# 2. Remote COG URI — only COG matches
# ---------------------------------------------------------------------------


class TestRemoteCOGURI:
    """Remote URIs should only match COG connector."""

    def test_https_uri(self):
        router, _, _, _, _ = _make_router()
        ref = SourceRef.uri("https://storage.example.com/dem.tif")
        matches = router.select(ref)
        assert len(matches) == 1
        assert matches[0].connector.name == "cog"

    def test_s3_uri(self):
        router, _, _, _, _ = _make_router()
        ref = SourceRef.uri("s3://bucket/path/dem.tif")
        matches = router.select(ref)
        assert len(matches) == 1
        assert matches[0].connector.name == "cog"

    def test_gs_uri(self):
        router, _, _, _, _ = _make_router()
        ref = SourceRef.uri("gs://bucket/dem.tif")
        matches = router.select(ref)
        assert len(matches) == 1
        assert matches[0].connector.name == "cog"

    def test_remote_rank_is_zero(self):
        router, _, _, _, _ = _make_router()
        ref = SourceRef.uri("https://example.com/dem.tif")
        match = router.select_one(ref)
        assert match.rank == 0


# ---------------------------------------------------------------------------
# 3. STAC ref — only STAC matches
# ---------------------------------------------------------------------------


class TestSTACRef:
    """STAC catalog items should only match STAC connector."""

    def test_stac_collection_item(self):
        router, _, _, _, _ = _make_router()
        ref = SourceRef.stac("sentinel-2-l2a", "S2A_20230615_T10SGD")
        matches = router.select(ref)
        assert len(matches) == 1
        assert matches[0].connector.name == "stac"

    def test_stac_with_asset(self):
        router, _, _, _, _ = _make_router()
        ref = SourceRef.stac("sentinel-2-l2a", "S2A_20230615_T10SGD", asset="B04")
        matches = router.select(ref)
        assert len(matches) == 1
        assert matches[0].connector.name == "stac"

    def test_stac_reason_is_kind_match(self):
        router, _, _, _, _ = _make_router()
        ref = SourceRef.stac("naip", "item-123")
        match = router.select_one(ref)
        assert match.reason == MatchReason.KIND_MATCH


# ---------------------------------------------------------------------------
# 4. PostGIS ref — only PostGIS matches
# ---------------------------------------------------------------------------


class TestPostGISRef:
    """Database references should only match PostGIS connector."""

    def test_schema_table(self):
        router, _, _, _, _ = _make_router()
        ref = SourceRef.postgis("public", "parcels")
        matches = router.select(ref)
        assert len(matches) == 1
        assert matches[0].connector.name == "postgis"

    def test_sql_query(self):
        router, _, _, _, _ = _make_router()
        ref = SourceRef.postgis_query("SELECT * FROM public.parcels LIMIT 10")
        matches = router.select(ref)
        assert len(matches) == 1
        assert matches[0].connector.name == "postgis"

    def test_postgis_reason_is_kind_match(self):
        router, _, _, _, _ = _make_router()
        ref = SourceRef.postgis("geo", "rivers")
        match = router.select_one(ref)
        assert match.reason == MatchReason.KIND_MATCH


# ---------------------------------------------------------------------------
# 5. Unknown/unsupported source
# ---------------------------------------------------------------------------


class TestUnknownSource:
    """UNKNOWN refs should only match fallback connectors."""

    def test_unknown_ref_matches_fallback(self):
        router, local, _, _, _ = _make_router()
        ref = SourceRef(raw="???something???", kind=SourceRefKind.UNKNOWN)
        matches = router.select(ref)
        assert len(matches) == 1
        assert matches[0].connector.name == "local_file"
        assert matches[0].reason == MatchReason.FALLBACK

    def test_unknown_fallback_ranked_low(self):
        router, _, _, _, _ = _make_router()
        ref = SourceRef(raw="mystery", kind=SourceRefKind.UNKNOWN)
        matches = router.select(ref)
        assert matches[0].rank >= 1000

    def test_no_fallback_means_empty(self):
        """Router with no fallback connectors returns empty for UNKNOWN."""
        router = ConnectorRouter()
        router.register(
            StubConnector("strict"),
            kinds={SourceRefKind.LOCAL_RASTER},
            priority=0,
        )
        ref = SourceRef(raw="mystery", kind=SourceRefKind.UNKNOWN)
        assert router.select(ref) == []

    def test_select_one_raises_for_no_match(self):
        router = ConnectorRouter()
        ref = SourceRef(raw="nothing", kind=SourceRefKind.UNKNOWN)
        with pytest.raises(NoConnectorError) as exc_info:
            router.select_one(ref)
        assert "nothing" in str(exc_info.value)

    def test_no_connector_error_contains_kind(self):
        router = ConnectorRouter()
        ref = SourceRef.local("/data/missing.tif")
        with pytest.raises(NoConnectorError) as exc_info:
            router.select_one(ref)
        assert "local_raster" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 6. Backward compat — raw strings auto-inferred
# ---------------------------------------------------------------------------


class TestRawStringBackwardCompat:
    """Passing a raw string should auto-infer and route correctly."""

    def test_local_path_string(self):
        router, _, _, _, _ = _make_router()
        matches = router.select("/data/dem.tif")
        assert matches[0].connector.name == "cog"

    def test_https_string(self):
        router, _, _, _, _ = _make_router()
        matches = router.select("https://example.com/dem.tif")
        assert len(matches) == 1
        assert matches[0].connector.name == "cog"

    def test_stac_pattern_string(self):
        router, _, _, _, _ = _make_router()
        matches = router.select("sentinel-2-l2a/S2A_20230615_T10SGD")
        assert len(matches) == 1
        assert matches[0].connector.name == "stac"

    def test_postgis_pattern_string(self):
        router, _, _, _, _ = _make_router()
        matches = router.select("public.parcels")
        assert len(matches) == 1
        assert matches[0].connector.name == "postgis"

    def test_sql_string(self):
        router, _, _, _, _ = _make_router()
        matches = router.select("SELECT * FROM rivers")
        assert len(matches) == 1
        assert matches[0].connector.name == "postgis"

    def test_select_one_with_raw_string(self):
        router, _, _, _, _ = _make_router()
        match = router.select_one("/data/dem.tif")
        assert match.connector.name == "cog"


# ---------------------------------------------------------------------------
# 7. Priority ordering
# ---------------------------------------------------------------------------


class TestPriorityOrdering:
    """Lower priority number = higher rank."""

    def test_custom_priority_reversal(self):
        """If LocalFile has lower priority than COG, it wins."""
        router = ConnectorRouter()
        cog = StubConnector("cog")
        local = StubConnector("local_file")
        router.register(cog, kinds={SourceRefKind.LOCAL_RASTER}, priority=10)
        router.register(local, kinds={SourceRefKind.LOCAL_RASTER}, priority=0)
        matches = router.select(SourceRef.local("/data/dem.tif"))
        assert matches[0].connector.name == "local_file"
        assert matches[1].connector.name == "cog"

    def test_same_priority_both_returned(self):
        router = ConnectorRouter()
        a = StubConnector("a")
        b = StubConnector("b")
        router.register(a, kinds={SourceRefKind.LOCAL_RASTER}, priority=5)
        router.register(b, kinds={SourceRefKind.LOCAL_RASTER}, priority=5)
        matches = router.select(SourceRef.local("/data/x.tif"))
        assert len(matches) == 2


# ---------------------------------------------------------------------------
# 8. ConnectorMatch is sortable
# ---------------------------------------------------------------------------


class TestConnectorMatchSorting:
    """ConnectorMatch should sort by rank."""

    def test_sort_order(self):
        a = ConnectorMatch(StubConnector("a"), MatchReason.KIND_MATCH, rank=10)
        b = ConnectorMatch(StubConnector("b"), MatchReason.KIND_MATCH, rank=0)
        c = ConnectorMatch(StubConnector("c"), MatchReason.FALLBACK, rank=1005)
        result = sorted([a, c, b])
        assert [m.connector.name for m in result] == ["b", "a", "c"]


# ---------------------------------------------------------------------------
# 9. Registration introspection
# ---------------------------------------------------------------------------


class TestRegistrationIntrospection:
    """Router exposes its registrations for debugging."""

    def test_registrations_property(self):
        router, _, _, _, _ = _make_router()
        regs = router.registrations
        assert len(regs) == 4
        names = [r[0] for r in regs]
        assert "cog" in names
        assert "local_file" in names
        assert "stac" in names
        assert "postgis" in names

    def test_empty_router(self):
        router = ConnectorRouter()
        assert router.registrations == []


# ---------------------------------------------------------------------------
# 10. Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Boundary conditions and degenerate inputs."""

    def test_empty_string(self):
        router, local, _, _, _ = _make_router()
        matches = router.select("")
        # Empty string infers to UNKNOWN, only fallback matches
        assert len(matches) == 1
        assert matches[0].reason == MatchReason.FALLBACK

    def test_source_ref_passes_through(self):
        """SourceRef input is not re-inferred."""
        router, _, _, _, _ = _make_router()
        ref = SourceRef(raw="not-a-path", kind=SourceRefKind.LOCAL_RASTER)
        matches = router.select(ref)
        # Should match declared local-raster connectors even though raw looks weird
        assert len(matches) == 2
        assert matches[0].connector.name == "cog"

    def test_stub_satisfies_protocol(self):
        """StubConnector is a valid Connector."""
        assert isinstance(StubConnector("test"), Connector)

"""
Pressure test: ConnectorRouter integration across all connector types.

Validates that the router correctly routes different SourceRef kinds
to the appropriate connectors according to priority and kind matching.

This test ensures the "false states must be impossible" principle:
every SourceRef must route to exactly one connector, and that connector
must be the correct one for the source type.
"""

import os

import pytest
from quarry_connectors.cog import COGConnector
from quarry_connectors.duckdb_connector import DuckDBConnector
from quarry_connectors.local_file import LocalFileConnector
from quarry_connectors.postgis import PostGISConnector
from quarry_connectors.stac import STACConnector
from quarry_core.router import ConnectorRouter, NoConnectorError
from quarry_core.source_ref import SourceRef, SourceRefKind


def _get_test_router():
    """Build router with full connector configuration matching CLI."""
    router = ConnectorRouter()

    # COGConnector: priority 0 for rasters and remote URIs
    router.register(
        COGConnector(),
        priority=0,
        kinds=[SourceRefKind.LOCAL_RASTER, SourceRefKind.REMOTE_URI],
    )

    # STACConnector: priority 0 for catalog items
    stac_api = os.environ.get("STAC_API_URL", "https://test-stac.example.com/v1")
    router.register(
        STACConnector(api_url=stac_api),
        priority=0,
        kinds=[SourceRefKind.CATALOG_ITEM],
    )

    # PostGISConnector: priority 0 for database refs
    router.register(
        PostGISConnector(),
        priority=0,
        kinds=[SourceRefKind.DATABASE_REF],
    )

    # DuckDBConnector: priority 0 for DuckDB database files
    router.register(
        DuckDBConnector(),
        priority=0,
        kinds=[SourceRefKind.DUCKDB],
    )

    # LocalFileConnector: priority 10 (fallback) for local files
    router.register(
        LocalFileConnector(),
        priority=10,
        kinds=[
            SourceRefKind.LOCAL_PATH,
            SourceRefKind.LOCAL_RASTER,
            SourceRefKind.LOCAL_VECTOR,
        ],
    )

    return router


class TestRouterLocalPathRouting:
    """Validate LOCAL_PATH routing through ConnectorRouter."""

    def test_local_path_routes_to_local_file_connector(self):
        """Non-raster, non-vector local files should route to LocalFileConnector."""
        router = _get_test_router()
        source = SourceRef.local("/data/config.txt")

        match = router.select_one(source)

        assert match is not None, "Expected a connector match"
        assert isinstance(match.connector, LocalFileConnector)


class TestRouterRasterRouting:
    """Validate LOCAL_RASTER routing with priority resolution."""

    def test_tif_file_routes_to_cog_connector(self):
        """.tif files should route to COGConnector (priority 0 beats LocalFile at 10)."""
        router = _get_test_router()
        source = SourceRef.local("/data/dem.tif")

        match = router.select_one(source)

        assert match is not None, "Expected a connector match"
        assert isinstance(match.connector, COGConnector), (
            f"Expected COGConnector, got {type(match.connector).__name__}"
        )
        # COGConnector wins due to lower priority value (0 vs 10)

    def test_tiff_file_routes_to_cog_connector(self):
        """.tiff extension should also route to COGConnector."""
        router = _get_test_router()
        source = SourceRef.local("/data/elevation.tiff")

        match = router.select_one(source)

        assert match is not None
        assert isinstance(match.connector, COGConnector)


class TestRouterRemoteURIRouting:
    """Validate REMOTE_URI routing to COGConnector."""

    def test_s3_uri_routes_to_cog_connector(self):
        """S3 URIs should route to COGConnector for cloud-optimized access."""
        router = _get_test_router()
        source = SourceRef.uri("s3://bucket/prefix/dem.tif")

        match = router.select_one(source)

        assert match is not None, "Expected S3 URI to route"
        assert isinstance(match.connector, COGConnector), (
            f"Expected COGConnector for S3, got {type(match.connector).__name__}"
        )

    def test_https_uri_routes_to_cog_connector(self):
        """HTTPS URLs should route to COGConnector."""
        router = _get_test_router()
        source = SourceRef.uri("https://example.com/data/raster.tif")

        match = router.select_one(source)

        assert match is not None
        assert isinstance(match.connector, COGConnector)

    def test_gs_uri_routes_to_cog_connector(self):
        """Google Cloud Storage URIs should route to COGConnector."""
        router = _get_test_router()
        source = SourceRef.uri("gs://bucket/dem.tif")

        match = router.select_one(source)

        assert match is not None
        assert isinstance(match.connector, COGConnector)


class TestRouterSTACRouting:
    """Validate CATALOG_ITEM routing to STACConnector."""

    def test_stac_item_routes_to_stac_connector(self):
        """STAC items via SourceRef.stac() should route to STACConnector."""
        router = _get_test_router()
        source = SourceRef.stac("sentinel2-l1c", "S2A_T44TKR_20240315T052859")

        match = router.select_one(source)

        assert match is not None, "Expected STAC routing to match"
        assert isinstance(match.connector, STACConnector), (
            f"Expected STACConnector, got {type(match.connector).__name__}"
        )

    def test_stac_item_with_asset_routes_to_stac_connector(self):
        """STAC items with asset specification should route to STACConnector."""
        router = _get_test_router()
        source = SourceRef.stac(
            "landsat-c2-l2", "LC08_L2SP_044034_20240315_20240403_02_T1", asset="red"
        )

        match = router.select_one(source)

        assert match is not None
        assert isinstance(match.connector, STACConnector)


class TestRouterPostGISRouting:
    """Validate DATABASE_REF routing to PostGISConnector."""

    def test_postgis_table_routes_to_postgis_connector(self):
        """PostGIS tables via SourceRef.postgis() should route to PostGISConnector."""
        router = _get_test_router()
        source = SourceRef.postgis("public", "watersheds")

        match = router.select_one(source)

        assert match is not None, "Expected PostGIS routing to match"
        assert isinstance(match.connector, PostGISConnector), (
            f"Expected PostGISConnector, got {type(match.connector).__name__}"
        )

    def test_postgis_query_routes_to_postgis_connector(self):
        """PostGIS queries via SourceRef.postgis_query() should route to PostGISConnector."""
        router = _get_test_router()
        source = SourceRef.postgis_query("SELECT * FROM watersheds WHERE area > 1000")

        match = router.select_one(source)

        assert match is not None
        assert isinstance(match.connector, PostGISConnector)


class TestRouterDuckDBRouting:
    """Validate DUCKDB routing to DuckDBConnector."""

    def test_duckdb_table_routes_to_duckdb_connector(self):
        """DuckDB tables via SourceRef.duckdb() should route to DuckDBConnector."""
        router = _get_test_router()
        source = SourceRef.duckdb("/data/analytics.duckdb", "measurements")

        match = router.select_one(source)

        assert match is not None, "Expected DuckDB routing to match"
        assert isinstance(match.connector, DuckDBConnector), (
            f"Expected DuckDBConnector, got {type(match.connector).__name__}"
        )

    def test_duckdb_query_routes_to_duckdb_connector(self):
        """DuckDB queries via SourceRef.duckdb_query() should route to DuckDBConnector."""
        router = _get_test_router()
        source = SourceRef.duckdb_query("/data/analytics.duckdb", "SELECT * FROM t")

        match = router.select_one(source)

        assert match is not None
        assert isinstance(match.connector, DuckDBConnector)


class TestRouterErrorHandling:
    """Validate error cases: unmatched sources must fail explicitly."""

    def test_unsupported_source_kind_raises_error(self):
        """Empty router should raise NoConnectorError for any source."""
        router = ConnectorRouter()  # Empty router - no connectors registered
        source = SourceRef.local("/data/config.txt")

        with pytest.raises(NoConnectorError):
            router.select_one(source)

    def test_no_match_for_unknown_kind(self):
        """SourceRef with UNKNOWN kind should not match any connector."""
        router = _get_test_router()
        source = SourceRef(raw="something_ambiguous", kind=SourceRefKind.UNKNOWN, params={})

        with pytest.raises(NoConnectorError):
            router.select_one(source)


class TestRouterPrioritySystem:
    """Validate priority-based selection works correctly."""

    def test_lower_priority_wins(self):
        """Lower priority number = higher precedence."""
        router = ConnectorRouter()
        cog = COGConnector()
        local = LocalFileConnector()

        # Register COG first with priority 0
        router.register(cog, priority=0, kinds=[SourceRefKind.LOCAL_RASTER])
        # Register LocalFile second with priority 10
        router.register(local, priority=10, kinds=[SourceRefKind.LOCAL_RASTER])

        source = SourceRef.local("/data/dem.tif")
        match = router.select_one(source)

        assert match.connector is cog
        # COGConnector wins regardless of registration order

    def test_reverse_registration_order_same_result(self):
        """Priority determines winner, not registration order."""
        router = ConnectorRouter()
        local = LocalFileConnector()
        cog = COGConnector()

        # Register LocalFile first with priority 10
        router.register(local, priority=10, kinds=[SourceRefKind.LOCAL_RASTER])
        # Register COG second with priority 0 (should still win)
        router.register(cog, priority=0, kinds=[SourceRefKind.LOCAL_RASTER])

        source = SourceRef.local("/data/dem.tif")
        match = router.select_one(source)

        assert match.connector is cog
        # COGConnector wins regardless of registration order


class TestRouterCLIIntegration:
    """Validate that CLI's _get_router() produces correct configuration."""

    def test_cli_router_matches_expected_configuration(self):
        """CLI router must match our test configuration exactly."""
        from quarry_cli.main import _get_router

        router = _get_router()

        # Test all routing scenarios
        test_cases = [
            ("local_path", "/local/path.txt", SourceRefKind.LOCAL_PATH, LocalFileConnector),
            ("local_raster", "/data/dem.tif", SourceRefKind.LOCAL_RASTER, COGConnector),
            ("remote_uri", "s3://bucket/file.tif", SourceRefKind.REMOTE_URI, COGConnector),
            (
                "catalog_item",
                "sentinel2-l1c/S2A_T44TKR_20240315T052859",
                SourceRefKind.CATALOG_ITEM,
                STACConnector,
            ),
            ("database_ref", "public.watersheds", SourceRefKind.DATABASE_REF, PostGISConnector),
            ("duckdb", "/data/my.duckdb::table1", SourceRefKind.DUCKDB, DuckDBConnector),
        ]

        for test_name, uri, expected_kind, expected_type in test_cases:
            if test_name == "local_path":
                source = SourceRef.local(uri)
            elif test_name == "local_raster":
                source = SourceRef.local(uri)
            elif test_name == "remote_uri":
                source = SourceRef.uri(uri)
            elif test_name == "catalog_item":
                parts = uri.split("/")
                source = SourceRef.stac(parts[0], parts[1])
            elif test_name == "database_ref":
                parts = uri.split(".")
                source = SourceRef.postgis(parts[0], parts[1])
            elif test_name == "duckdb":
                db_part, tbl_part = uri.split("::")
                source = SourceRef.duckdb(db_part, tbl_part)

            match = router.select_one(source)
            assert match is not None, f"No match for {uri} ({test_name})"
            assert isinstance(match.connector, expected_type), (
                f"Expected {expected_type.__name__} for {uri}, got {type(match.connector).__name__}"
            )

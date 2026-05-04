"""
Pressure test: ConnectorRouter integration across default CLI connector types.

Validates that the router correctly routes different SourceRef kinds
to the appropriate connectors according to priority, kind matching, and
source-shape filters.

This test ensures the "false states must be impossible" principle:
every default CLI SourceRef must select one best connector, and that connector
must be correct for the source type.
"""

import pytest
from quarry_connectors.cog import COGConnector
from quarry_connectors.csv_xy import CSVXYConnector
from quarry_connectors.duckdb_connector import DuckDBConnector
from quarry_connectors.excel_xy import ExcelXYConnector
from quarry_connectors.flatgeobuf import FlatGeobufConnector
from quarry_connectors.geopackage import GeoPackageConnector
from quarry_connectors.hdf5 import HDF5Connector
from quarry_connectors.local_file import LocalFileConnector
from quarry_connectors.netcdf import NetCDFConnector
from quarry_connectors.object_store import ObjectStoreConnector
from quarry_connectors.ogc_services import OGCServicesConnector
from quarry_connectors.opentopography import OpenTopographyConnector
from quarry_connectors.overture import OvertureConnector
from quarry_connectors.postgis import PostGISConnector
from quarry_connectors.router import build_default_router
from quarry_connectors.shapefile import ShapefileConnector
from quarry_connectors.stac import STACConnector
from quarry_connectors.topojson import TopoJSONConnector
from quarry_connectors.zarr_connector import ZarrConnector
from quarry_core.router import ConnectorRouter, NoConnectorError
from quarry_core.source_ref import SourceRef, SourceRefKind


def _get_test_router():
    """Build the shared default CLI connector router."""
    return build_default_router(stac_api_url="https://test-stac.example.com/v1")


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


class TestRouterExtensionRouting:
    """Validate specialized connector routing by source extension."""

    @pytest.mark.parametrize(
        ("source", "expected_type"),
        [
            ("/data/watersheds.shp", ShapefileConnector),
            ("/data/layers.gpkg::watersheds", GeoPackageConnector),
            ("/data/points.csv", CSVXYConnector),
            ("/data/workbook.xlsx::Sheet1", ExcelXYConnector),
            ("/data/product.h5::/science/elevation", HDF5Connector),
            ("/data/grid.nc::elevation", NetCDFConnector),
            ("/data/tiles.zarr/", ZarrConnector),
            ("/data/basins.topojson::huc12", TopoJSONConnector),
        ],
    )
    def test_local_extensions_route_to_specialized_connectors(self, source, expected_type):
        router = _get_test_router()

        match = router.select_one(source)

        assert isinstance(match.connector, expected_type), (
            f"Expected {expected_type.__name__} for {source}, got {type(match.connector).__name__}"
        )

    @pytest.mark.parametrize(
        ("source", "expected_type"),
        [
            ("/data/watersheds.geojson", LocalFileConnector),
            ("/data/basins.json", LocalFileConnector),
        ],
    )
    def test_ambiguous_extensions_do_not_route_to_specialized_connectors(
        self, source, expected_type
    ):
        router = _get_test_router()

        match = router.select_one(source)

        assert isinstance(match.connector, expected_type)


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

    def test_non_cog_s3_uri_routes_to_object_store_connector(self):
        """Remote non-COG geospatial files should route to ObjectStoreConnector."""
        router = _get_test_router()
        source = SourceRef.uri("s3://bucket/prefix/vector.gpkg")

        match = router.select_one(source)

        assert isinstance(match.connector, ObjectStoreConnector)

    def test_https_flatgeobuf_routes_to_flatgeobuf_connector(self):
        """HTTP(S) FlatGeobuf has a specialized connector ahead of ObjectStore."""
        router = _get_test_router()
        source = SourceRef.uri("https://example.com/data/roads.fgb")

        match = router.select_one(source)

        assert isinstance(match.connector, FlatGeobufConnector)


class TestRouterPrefixRouting:
    """Validate service/provider prefixes before broad catalog routing."""

    @pytest.mark.parametrize(
        ("source", "expected_type"),
        [
            ("wms::https://example.com/geoserver/wms::workspace:layer", OGCServicesConnector),
            ("wfs::https://example.com/geoserver/wfs::workspace:layer", OGCServicesConnector),
            ("overture://buildings/building", OvertureConnector),
            ("opentopo://SRTMGL1?bbox=-120,35,-119,36", OpenTopographyConnector),
        ],
    )
    def test_provider_prefixes_route_to_specialized_connectors(self, source, expected_type):
        router = _get_test_router()

        match = router.select_one(source)

        assert isinstance(match.connector, expected_type)


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

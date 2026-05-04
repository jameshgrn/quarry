"""Default connector router registrations.

Lane: registry

The router contract lives in quarry-core. This module owns connector-specific
registration metadata so adapters do not duplicate or invent routing policy.
"""

from __future__ import annotations

from collections.abc import Iterable

from quarry_core.connector import Connector
from quarry_core.router import ConnectorRouter
from quarry_core.source_ref import SourceRefKind

from quarry_connectors.cog import COGConnector
from quarry_connectors.csv_xy import CSVXYConnector
from quarry_connectors.duckdb_connector import DuckDBConnector
from quarry_connectors.excel_xy import ExcelXYConnector
from quarry_connectors.flatgeobuf import FlatGeobufConnector
from quarry_connectors.geojsonseq import GeoJSONSeqConnector
from quarry_connectors.geopackage import GeoPackageConnector
from quarry_connectors.geoparquet import GeoParquetConnector
from quarry_connectors.gpx import GPXConnector
from quarry_connectors.hdf5 import HDF5Connector
from quarry_connectors.kmz import KMZConnector
from quarry_connectors.las import LASPointCloudConnector
from quarry_connectors.local_file import LocalFileConnector
from quarry_connectors.mbtiles import MBTilesConnector
from quarry_connectors.netcdf import NetCDFConnector
from quarry_connectors.object_store import ObjectStoreConnector
from quarry_connectors.ogc_services import OGCServicesConnector
from quarry_connectors.opentopography import OpenTopographyConnector
from quarry_connectors.overture import OvertureConnector
from quarry_connectors.postgis import PostGISConnector
from quarry_connectors.shapefile import ShapefileConnector
from quarry_connectors.spatialite import SpatiaLiteConnector
from quarry_connectors.stac import STACConnector
from quarry_connectors.topojson import TopoJSONConnector
from quarry_connectors.zarr_connector import ZarrConnector

DEFAULT_STAC_API_URL = "https://earth-search.aws.element84.com/v0"

_LOCAL_RASTER_KINDS = {SourceRefKind.LOCAL_PATH, SourceRefKind.LOCAL_RASTER}
_LOCAL_VECTOR_KINDS = {SourceRefKind.LOCAL_PATH, SourceRefKind.LOCAL_VECTOR}
_LOCAL_TABLE_KINDS = {SourceRefKind.LOCAL_PATH, SourceRefKind.LOCAL_VECTOR}
_PREFIX_KINDS = {
    SourceRefKind.CATALOG_ITEM,
    SourceRefKind.REMOTE_URI,
    SourceRefKind.UNKNOWN,
}
_REMOTE_OBJECT_SCHEMES = {"http", "https", "s3", "gs", "az"}


def _register_local(
    router: ConnectorRouter,
    connector: Connector,
    *,
    kinds: Iterable[SourceRefKind],
    extensions: set[str],
) -> None:
    router.register(
        connector,
        kinds=kinds,
        priority=0,
        extensions=extensions,
    )


def build_default_router(*, stac_api_url: str | None = None) -> ConnectorRouter:
    """Build the canonical default connector router.

    This is explicit source-shape routing, not connector discovery:
    extension/prefix/scheme metadata is registered here and kept out of
    quarry-core so core remains zero-dependency.
    """
    router = ConnectorRouter()

    cog = COGConnector()
    router.register(
        cog,
        kinds=_LOCAL_RASTER_KINDS,
        priority=0,
        extensions={".tif", ".tiff", ".geotiff"},
    )
    router.register(
        cog,
        kinds={SourceRefKind.REMOTE_URI},
        priority=0,
        extensions={".tif", ".tiff", ".geotiff"},
        schemes=_REMOTE_OBJECT_SCHEMES,
    )

    flatgeobuf = FlatGeobufConnector()
    _register_local(
        router,
        flatgeobuf,
        kinds=_LOCAL_VECTOR_KINDS,
        extensions={".fgb"},
    )
    router.register(
        flatgeobuf,
        kinds={SourceRefKind.REMOTE_URI},
        priority=0,
        extensions={".fgb"},
        schemes={"http", "https"},
    )

    _register_local(
        router,
        ShapefileConnector(),
        kinds=_LOCAL_VECTOR_KINDS,
        extensions={".shp"},
    )
    _register_local(
        router,
        GeoPackageConnector(),
        kinds=_LOCAL_VECTOR_KINDS,
        extensions={".gpkg"},
    )
    _register_local(
        router,
        GeoParquetConnector(),
        kinds=_LOCAL_VECTOR_KINDS,
        extensions={".parquet", ".geoparquet"},
    )
    _register_local(router, GPXConnector(), kinds=_LOCAL_VECTOR_KINDS, extensions={".gpx"})
    _register_local(router, KMZConnector(), kinds=_LOCAL_VECTOR_KINDS, extensions={".kmz"})
    _register_local(
        router,
        LASPointCloudConnector(),
        kinds=_LOCAL_VECTOR_KINDS,
        extensions={".las", ".laz"},
    )
    _register_local(
        router,
        SpatiaLiteConnector(),
        kinds=_LOCAL_VECTOR_KINDS,
        extensions={".sqlite", ".spatialite"},
    )
    _register_local(
        router,
        TopoJSONConnector(),
        kinds=_LOCAL_VECTOR_KINDS,
        extensions={".topojson"},
    )

    _register_local(
        router,
        CSVXYConnector(),
        kinds=_LOCAL_TABLE_KINDS,
        extensions={".csv", ".tsv"},
    )
    _register_local(
        router,
        ExcelXYConnector(),
        kinds=_LOCAL_TABLE_KINDS,
        extensions={".xls", ".xlsx"},
    )
    _register_local(
        router,
        GeoJSONSeqConnector(),
        kinds=_LOCAL_VECTOR_KINDS,
        extensions={".geojsonl", ".geojsonseq", ".ndjson"},
    )

    _register_local(
        router,
        HDF5Connector(),
        kinds=_LOCAL_RASTER_KINDS,
        extensions={".h5", ".hdf5", ".hdf", ".he5"},
    )
    _register_local(
        router,
        NetCDFConnector(),
        kinds=_LOCAL_RASTER_KINDS,
        extensions={".nc", ".nc4"},
    )
    _register_local(
        router,
        MBTilesConnector(),
        kinds=_LOCAL_RASTER_KINDS,
        extensions={".mbtiles"},
    )
    _register_local(
        router,
        ZarrConnector(),
        kinds=_LOCAL_RASTER_KINDS,
        extensions={".zarr"},
    )

    router.register(
        OGCServicesConnector(),
        kinds=_PREFIX_KINDS,
        priority=0,
        prefixes={"wms::", "wfs::"},
    )
    router.register(
        OvertureConnector(),
        kinds=_PREFIX_KINDS,
        priority=0,
        prefixes={"overture://"},
    )
    router.register(
        OpenTopographyConnector(),
        kinds=_PREFIX_KINDS,
        priority=0,
        prefixes={"opentopo://"},
    )

    router.register(
        ObjectStoreConnector(),
        kinds={SourceRefKind.REMOTE_URI},
        priority=50,
        schemes=_REMOTE_OBJECT_SCHEMES,
    )
    router.register(
        STACConnector(api_url=stac_api_url or DEFAULT_STAC_API_URL),
        kinds={SourceRefKind.CATALOG_ITEM},
        priority=50,
    )
    router.register(PostGISConnector(), kinds={SourceRefKind.DATABASE_REF}, priority=0)
    router.register(DuckDBConnector(), kinds={SourceRefKind.DUCKDB}, priority=0)
    router.register(
        LocalFileConnector(),
        kinds={
            SourceRefKind.LOCAL_PATH,
            SourceRefKind.LOCAL_RASTER,
            SourceRefKind.LOCAL_VECTOR,
        },
        priority=100,
    )

    return router

# Quarry

Geospatial execution substrate. Typed contracts from ingest to output — connectors, operators, executors, and a registry that remembers everything.

**Status: work in progress.** Substrate phase is complete. CLI adapter is functional. Not yet packaged for distribution.

## Architecture

```
quarry/
  packages/
    quarry-core/         # Contracts: Artifact, Connector, Operator, Executor, Check (zero deps)
    quarry-connectors/   # 29 connector implementations
    quarry-operators/    # 16 operator implementations + HydrologyFlow
    quarry-registry/     # DuckDB-backed artifact/run/check/lineage persistence
    quarry-cli/          # CLI adapter (argparse, no new deps)
```

```
quarry-core (zero external deps)
  ^
quarry-connectors (+ rasterio, fiona, pystac-client, psycopg, shapely, duckdb)
quarry-operators  (+ rasterio, fiona, shapely)
quarry-registry   (+ duckdb)
  ^
quarry-cli        (adapter — all four above)
```

### Core contracts

- **Artifact** — internal unit of truth. Identity is a UUID, not a file path. Carries spatial descriptor, lineage, and validation state.
- **Connector** — sacred gateway. No geospatial object enters except through a connector. Materializes source references into artifacts.
- **ConnectorRouter** — registry-lane selector. Default routing uses explicit extension, scheme, and provider-prefix filters; ambiguous semantic products stay explicit.
- **Operator** — typed transformation. Declares input/output types, validates before execution, emits fresh metadata from actual output.
- **Executor** — dispatches operator execution. Captures full lifecycle (pending → running → completed/failed) as a RunRecord.
- **Check** — validation rule applied to artifacts and runs. Truth lives in the registry, not embedded in objects.
- **Registry** — DuckDB-backed persistent memory. Four tables: artifacts, runs, checks, lineage. Atomic cascading writes.

### Connectors

The default connector router lives in `quarry-connectors` and maps common source refs to connectors by extension, URI scheme, and provider prefix. It does not auto-route semantic product formats where the same extension can mean different products.

| Connector | Sources |
|-----------|---------|
| LocalFile | Local raster (GeoTIFF) and vector (GeoJSON, GeoPackage, Shapefile) |
| COG | Cloud-Optimized GeoTIFF — local and remote, with validation |
| STAC | SpatioTemporal Asset Catalog search + asset materialization |
| PostGIS | PostgreSQL/PostGIS tables and queries |
| DuckDB | DuckDB tables and spatial queries |
| CSVXY | CSV tables with detected X/Y or lon/lat columns |
| ExcelXY | Excel sheets with detected X/Y or lon/lat columns |
| GeoJSONSeq | GeoJSON sequence / newline-delimited GeoJSON |
| GeoPackage | GeoPackage layers |
| GeoParquet | Apache GeoParquet files |
| FlatGeobuf | FlatGeobuf vector files |
| FOFStack | Frequency-of-flooding NetCDF stacks |
| GPX | GPS Exchange Format tracks, routes, and waypoints |
| HDF5 | HDF5 scientific arrays |
| KMZ | Compressed KML/KMZ vectors |
| LASPointCloud | LAS/LAZ lidar point clouds |
| MBTiles | MBTiles raster/vector tile packages |
| NetCDF | NetCDF scientific raster data |
| ObjectStore | S3/GCS/Azure object-store paths via GDAL virtual filesystems |
| OGCServices | OGC WMS/WFS services |
| OpenTopography | OpenTopography API DEM downloads |
| Overture | Overture Maps Foundation data via DuckDB |
| PIXC | SWOT PIXC HDF5 point-cloud rasters |
| Sentinel2 | Sentinel-2 band mapper over STAC assets |
| Shapefile | ESRI Shapefile with sidecar validation |
| SLC | SWOT SLC HDF5 products |
| SpatiaLite | SpatiaLite databases |
| TopoJSON | TopoJSON vector objects |
| Zarr | Zarr stores |

### Operators

| Operator | What it does |
|----------|-------------|
| ClipRaster | Clip raster to bounds or mask geometry |
| Reproject | CRS transformation for raster and vector |
| FillDepressions | Priority-Flood DEM sink filling |
| Slope | Terrain slope from DEM |
| Aspect | Terrain aspect from DEM |
| Hillshade | Terrain hillshade from DEM |
| D8FlowDirection | Steepest-descent flow routing with flat resolution |
| FlowAccumulation | Topologically-sorted upstream contributing area |
| ZonalStats | Raster summarization per vector zone |
| SpatialJoin | Vector-on-vector left join (intersects) |
| SampleRaster | Extract raster values at point locations |
| RasterizeVector | Burn vector polygons to raster grid |
| BuildCOG | Normalize any raster to Cloud-Optimized GeoTIFF |
| GeocodeSLC | Geocode SWOT SLC rasters |
| SLCCalibration | Calibrate SLC real/imaginary bands |
| WaterElevationMosaic | Mosaic water elevation from SWOT-like rasters |

**HydrologyFlow** composes FillDepressions → D8FlowDirection → FlowAccumulation into a single chain with registry persistence at each step.

## Usage

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```sh
git clone https://github.com/jameshgrn/quarry.git
cd quarry
uv sync
```

### Run the canonical example

```sh
uv run python examples/watershed_analysis.py
```

This creates a synthetic DEM and zone polygons, runs the full hydrology chain, computes zonal statistics, exports to COG, and walks the lineage graph — all with synthetic data, no external dependencies.

### CLI

```sh
# List registered artifacts
uv run quarry artifacts list

# Run hydrology chain on a DEM
uv run quarry run hydrology --dem path/to/dem.tif

# Run zonal statistics
uv run quarry run zonal --raster path/to/raster.tif --zones path/to/zones.gpkg

# Inspect lineage
uv run quarry lineage <artifact-id>

# Run any operator generically
uv run quarry run operator --name Reproject --input path/to/data.tif --param target_crs=EPSG:4326
```

## Tests

1,907 pressure tests covering contracts, adversarial inputs, and end-to-end flows. Use `just stats` for the current count.

```sh
# Run a specific test file
just test tests/pressure_test/test_end_to_end.py

# Run the full gate
just test-all
```

## Design decisions

- **quarry-core has zero external dependencies.** All protocols are pure Python. Implementation packages bring their own deps.
- **Artifacts are not files.** An artifact has an identity (UUID), a backing store (which might be a file), spatial metadata read from actual data, and lineage recording how it was created.
- **Output metadata is always fresh.** Operators read spatial properties from the actual output — never copied from input.
- **The registry is the source of truth.** Checks, lineage edges, and run records live in DuckDB, not scattered across objects.
- **Connectors are the only entry point.** No geospatial data enters the system except through a connector's `materialize()` method.
- **Artifact metadata is not spatial truth.** CRS, extent, resolution, feature counts, and band counts live in `Artifact.spatial`; duplicate top-level metadata keys are stripped at artifact construction.

## License

Apache License 2.0. See [LICENSE](LICENSE).

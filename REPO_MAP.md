# Repo Map

```
quarry/                          # Monorepo root
├── CLAUDE.md                    # Doctrine + agent rules (canonical)
├── CONTRACTS.md                 # Human-readable contract semantics
├── REPO_MAP.md                  # This file
├── PRESSURE_TESTS.md            # Test history log
├── AGENTS.md                    # Agent behavior rules
├── justfile                     # Canonical commands
├── pyproject.toml               # Root manifest (uv workspace config)
├── uv.lock                      # Lockfile (committed)
│
├── packages/
│   ├── quarry-core/             # ZERO external deps
│   │   └── src/quarry_core/
│   │       ├── artifact.py      # Artifact, BackingStore, SpatialDescriptor, Lineage, CheckResult
│   │       ├── connector.py     # Connector protocol, MaterializeResult, ConnectorCapability
│   │       ├── source_ref.py    # SourceRef, SourceRefKind (typed envelope for source references)
│   │       ├── router.py        # ConnectorRouter, ConnectorMatch (selection layer)
│   │       ├── operator.py      # Operator protocol, OperatorSpec, OperatorResult
│   │       ├── executor.py      # Executor protocol, RunRecord, RunStatus
│   │       ├── check.py         # Check protocol + CRSValid, ExtentSane, BackingStoreAccessible
│   │       └── executors/
│   │           └── local.py     # LocalExecutor (synchronous, in-process)
│   │
│   ├── quarry-connectors/       # Deps: rasterio, fiona, pystac-client, psycopg, shapely, duckdb, ...
│   │   └── src/quarry_connectors/
│   │       ├── local_file.py    # LocalFileConnector (raster + vector, eager + lazy)
│   │       ├── cog.py           # COGConnector (local/remote COG, validation, I/O metrics)
│   │       ├── stac.py          # STACConnector (catalog search, asset selection, lazy/eager)
│   │       ├── postgis.py       # PostGISConnector (schema.table, queries, geometry/non-geometry)
│   │       ├── duckdb_connector.py # DuckDBConnector (path.duckdb::table, geometry/non-geometry)
│   │       ├── geopackage.py    # GeoPackageConnector (GPKG layers)
│   │       ├── shapefile.py     # ShapefileConnector (.shp/.shx/.dbf bundle)
│   │       ├── flatgeobuf.py    # FlatGeobufConnector (.fgb vector)
│   │       ├── geoparquet.py    # GeoParquetConnector (columnar vector/geometry)
│   │       ├── geojsonseq.py    # GeoJSONSeqConnector (newline-delimited GeoJSON)
│   │       ├── topojson.py      # TopoJSONConnector (topology-encoded vector)
│   │       ├── csv_xy.py        # CSVXYConnector (CSV with X/Y coordinate columns)
│   │       ├── excel_xy.py      # ExcelXYConnector (Excel with X/Y coordinate columns)
│   │       ├── gpx.py           # GPXConnector (GPS exchange format)
│   │       ├── kmz.py           # KMZConnector (KML/KMZ archives)
│   │       ├── las.py           # LASPointCloudConnector (LiDAR point clouds)
│   │       ├── mbtiles.py       # MBTilesConnector (map tile packages)
│   │       ├── netcdf.py        # NetCDFConnector (multidimensional arrays)
│   │       ├── zarr_connector.py # ZarrConnector (chunked array storage)
│   │       ├── spatialite.py    # SpatiaLiteConnector (SQLite + spatial)
│   │       ├── object_store.py  # ObjectStoreConnector (S3/GCS/Azure blob)
│   │       ├── ogc_services.py  # OGCServicesConnector (WMS/WFS/WCS)
│   │       ├── opentopography.py # OpenTopographyConnector (DEM API)
│   │       └── overture.py      # OvertureConnector (Overture Maps)
│   │
│   ├── quarry-operators/        # Deps: rasterio, fiona, shapely
│   │   └── src/quarry_operators/
│   │       ├── clip_raster.py       # ClipRasterOperator (bounds + mask)
│   │       ├── reproject.py         # ReprojectOperator (raster + vector CRS transform)
│   │       ├── fill_depressions.py  # FillDepressionsOperator (Priority-Flood DEM preprocessing)
│   │       ├── slope.py             # SlopeOperator (terrain slope from DEM)
│   │       ├── aspect.py            # AspectOperator (terrain aspect from DEM)
│   │       ├── hillshade.py         # HillshadeOperator (illumination from DEM)
│   │       ├── d8_flow_direction.py # D8FlowDirectionOperator (steepest descent + flat resolution)
│   │       ├── flow_accumulation.py # FlowAccumulationOperator (toposort upstream area)
│   │       ├── zonal_stats.py       # ZonalStatsOperator (raster+vector → per-zone CSV stats)
│   │       ├── spatial_join.py      # SpatialJoinOperator (vector×vector left join, intersects)
│   │       ├── sample_raster.py     # SampleRasterOperator (raster+points → per-point CSV values)
│   │       ├── rasterize_vector.py  # RasterizeVectorOperator (vector polygons → raster grid)
│   │       ├── build_cog.py         # BuildCOGOperator (raster → COG normalization)
│   │       ├── buffer.py            # BufferOperator (vector geometry buffer by distance)
│   │       ├── dissolve.py          # DissolveOperator (merge features by attribute)
│   │       ├── clip_vector.py       # ClipVectorOperator (clip features to mask boundary)
│   │       ├── simplify.py          # SimplifyOperator (Douglas-Peucker simplification)
│   │       ├── checks.py           # Standalone checks (InternalOutletCount)
│   │       └── hydrology_flow.py   # HydrologyFlow (fill→D8→accumulation chain)
│   │
│   ├── quarry-registry/         # Deps: duckdb
│   │   └── src/quarry_registry/
│   │       └── registry.py      # DuckDB persistence (artifacts, runs, checks, lineage)
│   │
│   └── quarry-cli/              # Deps: quarry-core + connectors + operators + registry
│       └── src/quarry_cli/
│           └── main.py          # argparse CLI: artifacts list/show, lineage, run hydrology/zonal
│
├── tests/
│   ├── pressure_test/           # Substrate pressure tests (1601 tests)
│   │   ├── conftest.py          # PYTHONPATH setup for dev
│   │   ├── test_end_to_end.py   # Kernel: connector → operator → executor (15)
│   │   ├── test_registry.py     # Registry round-trips (18)
│   │   ├── test_stac_connector.py # STAC adversarial (22)
│   │   ├── test_reproject.py    # Reproject stress (19)
│   │   ├── test_postgis_connector.py # PostGIS adversarial (25)
│   │   ├── test_cog_connector.py # COG adversarial (24)
│   │   ├── test_source_ref.py  # SourceRef contract (34)
│   │   ├── test_connector_router.py # ConnectorRouter routing (34)
│   │   ├── test_fill_depressions.py # FillDepressions hydrology op (30)
│   │   ├── test_slope.py            # Slope terrain op (31)
│   │   ├── test_aspect.py           # Aspect terrain op (28)
│   │   ├── test_d8_flow_direction.py # D8 flow direction + chain tests (27)
│   │   ├── test_flow_accumulation.py # Flow accumulation + full chain (27)
│   │   ├── test_hydrology_flow.py # Hydrology chain composition (27+15)
│   │   ├── test_hydrology_adversarial.py # 27 pathological DEM fixtures
│   │   ├── test_internal_outlet_check.py # Standalone check tests
│   │   ├── test_zonal_stats.py  # ZonalStats raster+vector (21)
│   │   ├── test_sample_raster.py # SampleRaster raster+points (22)
│   │   ├── test_spatial_join.py # SpatialJoin vector×vector (20)
│   │   ├── test_build_cog.py   # BuildCOG normalization (22)
│   │   ├── test_rasterize_vector.py # RasterizeVector polygon→raster (25)
│   │   ├── test_cli.py          # CLI adapter: list/show/lineage/run hydrology (19)
│   │   ├── test_cli_zonal.py    # CLI adapter: run zonal end-to-end (12)
│   │   ├── test_cli_inspection.py # CLI adapter: runs list/show, checks show (20)
│   │   ├── test_cli_sample.py   # CLI adapter: run sample end-to-end (19)
│   │   ├── test_cli_rasterize.py # CLI adapter: run rasterize end-to-end (26)
│   │   ├── test_router_integration.py # ConnectorRouter integration across all connectors (15)
│   │   └── test_duckdb_connector.py # DuckDB connector: table/query/spatial/lazy/discover (42)
│   │   ├── test_hillshade.py        # Hillshade illumination op (51)
│   │   └── test_*.py               # + connector pressure tests (24 connectors)
│   └── fixtures/                # Test data (gitignored binaries)
│
├── examples/
│   └── watershed_analysis.py    # Canonical end-to-end: ingest→hydro→zonal→COG→lineage
│
└── hydrops/                     # RAIDING SOURCE — not a package, not integrated
                                 # Tiled hydrology harness (checks, COG I/O, schedulers)
                                 # Extract one piece at a time; pressure-test against contracts
```

## Package dependency graph

```
quarry-core (zero deps)
  ↑
quarry-connectors (+ rasterio, fiona, pystac-client, psycopg, shapely, duckdb, ...)
quarry-operators  (+ rasterio, fiona, shapely)
quarry-registry   (+ duckdb)
  ↑
quarry-cli        (adapter — all four packages above)
```

All implementation packages depend on quarry-core. quarry-cli depends on all four. No circular deps.

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
│   ├── quarry-connectors/       # Deps: rasterio, fiona, pystac-client, psycopg, shapely
│   │   └── src/quarry_connectors/
│   │       ├── cog.py              # COGConnector (local/remote COG, validation)
│   │       ├── csv_xy.py           # CSVXYConnector (CSV coordinates)
│   │       ├── duckdb_connector.py # DuckDBConnector (path.duckdb::table/query)
│   │       ├── excel_xy.py         # ExcelXYConnector (spreadsheet coordinates)
│   │       ├── flatgeobuf.py       # FlatGeobufConnector
│   │       ├── fof_stack.py        # FOFStackConnector
│   │       ├── geojsonseq.py       # GeoJSONSeqConnector
│   │       ├── geopackage.py       # GeoPackageConnector
│   │       ├── geoparquet.py       # GeoParquetConnector
│   │       ├── gpx.py              # GPXConnector
│   │       ├── hdf5.py             # HDF5Connector
│   │       ├── kmz.py              # KMZConnector
│   │       ├── las.py              # LASPointCloudConnector
│   │       ├── local_file.py       # LocalFileConnector
│   │       ├── mbtiles.py          # MBTilesConnector
│   │       ├── netcdf.py           # NetCDFConnector
│   │       ├── object_store.py     # ObjectStoreConnector
│   │       ├── ogc_services.py     # OGCServicesConnector
│   │       ├── opentopography.py   # OpenTopographyConnector
│   │       ├── overture.py         # OvertureConnector
│   │       ├── pixc.py             # PIXCConnector
│   │       ├── postgis.py          # PostGISConnector
│   │       ├── router.py           # Default ConnectorRouter registrations
│   │       ├── sentinel2.py        # Sentinel2Connector
│   │       ├── shapefile.py        # ShapefileConnector
│   │       ├── slc.py              # SLCConnector
│   │       ├── spatialite.py       # SpatiaLiteConnector
│   │       ├── stac.py             # STACConnector
│   │       ├── topojson.py         # TopoJSONConnector
│   │       └── zarr_connector.py   # ZarrConnector
│   │
│   ├── quarry-operators/        # Deps: rasterio, fiona, shapely
│   │   └── src/quarry_operators/
│   │       ├── clip_raster.py       # ClipRasterOperator (bounds + mask)
│   │       ├── reproject.py         # ReprojectOperator (raster + vector CRS transform)
│   │       ├── fill_depressions.py  # FillDepressionsOperator (Priority-Flood DEM preprocessing)
│   │       ├── slope.py             # SlopeOperator (terrain slope from DEM)
│   │       ├── aspect.py            # AspectOperator (terrain aspect from DEM)
│   │       ├── hillshade.py         # HillshadeOperator (terrain illumination)
│   │       ├── d8_flow_direction.py # D8FlowDirectionOperator (steepest descent + flat resolution)
│   │       ├── flow_accumulation.py # FlowAccumulationOperator (toposort upstream area)
│   │       ├── geocode_slc.py       # GeocodeSLCOperator
│   │       ├── slc_calibration.py   # SLCCalibrationOperator
│   │       ├── water_elevation_mosaic.py # WaterElevationMosaicOperator
│   │       ├── zonal_stats.py       # ZonalStatsOperator (raster+vector → per-zone CSV stats)
│   │       ├── spatial_join.py      # SpatialJoinOperator (vector×vector left join, intersects)
│   │       ├── sample_raster.py     # SampleRasterOperator (raster+points → per-point CSV values)
│   │       ├── rasterize_vector.py   # RasterizeVectorOperator (vector polygons → raster grid)
│   │       ├── build_cog.py        # BuildCOGOperator (raster → COG normalization)
│   │       ├── checks.py           # Standalone checks (InternalOutletCount)
│   │       └── hydrology_flow.py   # HydrologyFlow (fill→D8→accumulation chain)
│   │
│   ├── quarry-registry/         # Deps: duckdb
│   │   └── src/quarry_registry/
│   │       └── registry.py      # DuckDB persistence (artifacts, runs, checks, lineage)
│   │
│   └── quarry-cli/              # Deps: quarry-core + connectors + operators + registry
│       └── src/quarry_cli/
│           └── main.py          # argparse CLI: inspect registry, run flows/operators
│
├── tests/
│   ├── pressure_test/           # Substrate pressure tests; use `just stats` for current count
│   │   ├── test_*_connector.py  # Connector pressure tests
│   │   ├── test_*.py            # Operator, flow, adapter, registry, router tests
│   │   └── conftest.py          # PYTHONPATH setup for dev
│   └── fixtures/                # Test data (gitignored binaries)
│
├── packages/quarry-connectors/tests/ # Connector pressure tests included by pytest.ini
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
quarry-connectors (+ rasterio, fiona, pystac-client, psycopg, shapely, duckdb)
quarry-operators  (+ rasterio, fiona, shapely)
quarry-registry   (+ duckdb)
  ↑
quarry-cli        (adapter — all four packages above)
```

All implementation packages depend on quarry-core. quarry-cli depends on all four. No circular deps.

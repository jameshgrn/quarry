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
│   │       ├── operator.py      # Operator protocol, OperatorSpec, OperatorResult
│   │       ├── executor.py      # Executor protocol, RunRecord, RunStatus
│   │       ├── check.py         # Check protocol + CRSValid, ExtentSane, BackingStoreAccessible
│   │       └── executors/
│   │           └── local.py     # LocalExecutor (synchronous, in-process)
│   │
│   ├── quarry-connectors/       # Deps: rasterio, fiona, pystac-client, psycopg, shapely
│   │   └── src/quarry_connectors/
│   │       ├── local_file.py    # LocalFileConnector (raster + vector, eager + lazy)
│   │       ├── stac.py          # STACConnector (catalog search, asset selection, lazy/eager)
│   │       ├── postgis.py       # PostGISConnector (schema.table, queries, geometry/non-geometry)
│   │       └── cog.py           # COGConnector (local/remote COG, validation, I/O metrics)
│   │
│   ├── quarry-operators/        # Deps: rasterio, fiona, shapely
│   │   └── src/quarry_operators/
│   │       ├── clip_raster.py       # ClipRasterOperator (bounds + mask)
│   │       ├── reproject.py         # ReprojectOperator (raster + vector CRS transform)
│   │       ├── fill_depressions.py  # FillDepressionsOperator (Priority-Flood DEM preprocessing)
│   │       ├── d8_flow_direction.py # D8FlowDirectionOperator (steepest descent + flat resolution)
│   │       ├── flow_accumulation.py # FlowAccumulationOperator (toposort upstream area)
│   │       ├── zonal_stats.py       # ZonalStatsOperator (raster+vector → per-zone CSV stats)
│   │       ├── spatial_join.py      # SpatialJoinOperator (vector×vector left join, intersects)
│   │       ├── sample_raster.py      # SampleRasterOperator (raster+points → per-point CSV values)
│   │       ├── build_cog.py        # BuildCOGOperator (raster → COG normalization)
│   │       ├── checks.py           # Standalone checks (InternalOutletCount)
│   │       └── hydrology_flow.py   # HydrologyFlow (fill→D8→accumulation chain)
│   │
│   └── quarry-registry/         # Deps: duckdb
│       └── src/quarry_registry/
│           └── registry.py      # DuckDB persistence (artifacts, runs, checks, lineage)
│
├── src/georuntime/              # Legacy prototype (DO NOT MODIFY — migration deferred)
│
├── tests/
│   ├── pressure_test/           # Substrate pressure tests (405 tests, 16 suites)
│   │   ├── conftest.py          # PYTHONPATH setup for dev
│   │   ├── test_end_to_end.py   # Kernel: connector → operator → executor (15)
│   │   ├── test_registry.py     # Registry round-trips (18)
│   │   ├── test_stac_connector.py # STAC adversarial (22)
│   │   ├── test_reproject.py    # Reproject stress (19)
│   │   ├── test_postgis_connector.py # PostGIS adversarial (25)
│   │   ├── test_cog_connector.py # COG adversarial (24)
│   │   ├── test_source_ref.py  # SourceRef contract (34)
│   │   ├── test_fill_depressions.py # FillDepressions hydrology op (30)
│   │   ├── test_d8_flow_direction.py # D8 flow direction + chain tests (27)
│   │   ├── test_flow_accumulation.py # Flow accumulation + full chain (27)
│   │   ├── test_hydrology_flow.py # Hydrology chain composition (27+15)
│   │   ├── test_hydrology_adversarial.py # 27 pathological DEM fixtures
│   │   ├── test_internal_outlet_check.py # Standalone check tests
│   │   ├── test_zonal_stats.py  # ZonalStats raster+vector (21)
│   │   ├── test_sample_raster.py # SampleRaster raster+points (22)
│   │   ├── test_spatial_join.py # SpatialJoin vector×vector (20)
│   │   └── test_build_cog.py   # BuildCOG normalization (22)
│   └── fixtures/                # Test data (gitignored binaries)
│
└── hydrops/                     # RAIDING SOURCE — not a package, not integrated
                                 # Tiled hydrology harness (checks, COG I/O, schedulers)
                                 # Extract one piece at a time; pressure-test against contracts
```

## Package dependency graph

```
quarry-core (zero deps)
  ↑
quarry-connectors (+ rasterio, fiona, pystac-client, psycopg, shapely)
quarry-operators  (+ rasterio, fiona, shapely)
quarry-registry   (+ duckdb)
```

All implementation packages depend on quarry-core. No circular deps.

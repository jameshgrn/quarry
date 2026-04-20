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
│   │       ├── operator.py      # Operator protocol, OperatorSpec, OperatorResult
│   │       ├── executor.py      # Executor protocol, RunRecord, RunStatus
│   │       ├── check.py         # Check protocol + CRSValid, ExtentSane, BackingStoreAccessible
│   │       └── executors/
│   │           └── local.py     # LocalExecutor (synchronous, in-process)
│   │
│   ├── quarry-connectors/       # Deps: rasterio, fiona, pystac-client
│   │   └── src/quarry_connectors/
│   │       ├── local_file.py    # LocalFileConnector (raster + vector, eager + lazy)
│   │       └── stac.py          # STACConnector (catalog search, asset selection, lazy/eager)
│   │
│   ├── quarry-operators/        # Deps: rasterio, fiona, shapely
│   │   └── src/quarry_operators/
│   │       ├── clip_raster.py   # ClipRasterOperator (bounds + mask)
│   │       └── reproject.py     # ReprojectOperator (raster + vector CRS transform)
│   │
│   └── quarry-registry/         # Deps: duckdb
│       └── src/quarry_registry/
│           └── registry.py      # DuckDB persistence (artifacts, runs, checks, lineage)
│
├── src/georuntime/              # Legacy prototype (DO NOT MODIFY — migration deferred)
│
└── tests/
    ├── pressure_test/           # Substrate pressure tests (74 tests)
    │   ├── conftest.py          # PYTHONPATH setup for dev
    │   ├── test_end_to_end.py   # Kernel: connector → operator → executor (15)
    │   ├── test_registry.py     # Registry round-trips (18)
    │   ├── test_stac_connector.py # STAC adversarial (22)
    │   └── test_reproject.py    # Reproject stress (19)
    └── fixtures/                # Test data (gitignored binaries)
```

## Package dependency graph

```
quarry-core (zero deps)
  ↑
quarry-connectors (+ rasterio, fiona, pystac-client)
quarry-operators  (+ rasterio, fiona, shapely)
quarry-registry   (+ duckdb)
```

All implementation packages depend on quarry-core. No circular deps.

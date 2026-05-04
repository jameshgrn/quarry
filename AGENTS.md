# Quarry

Canonical geospatial execution substrate. This file is the canonical source of truth for agent behavior and project guidance.

## Doctrine

1. **Canonical pathways only.** No geospatial object enters except through a Connector. No transformation happens except through an Operator. No execution happens except through an Executor.
2. **Artifacts over files.** The system thinks in artifacts. A file is one possible backing store.
3. **Connectors are sacred.** First-class objects defining the canonical way to access a source.
4. **Operations only consume and emit canonical artifacts.**
5. **Execution is orthogonal.** Logic doesn't care where it runs.
6. **Storage is not the center.** Local disk first. Object storage later.
7. **Agents use the same substrate as humans.** No magic hidden routes.

## Lane Declaration Rule

Every idea must declare which lane it belongs to before it enters the repo:

- `connector` — how data gets in/out
- `artifact` — internal unit of truth
- `operator` — typed transformation
- `flow` — composition of operators
- `executor` — where it runs
- `check` — validation layer
- `adapter` — exposure surface (QGIS, API, agent)
- `registry` — what remembers everything

If an idea cannot name its lane, it is not ready.

## Monorepo Structure

```
quarry/
  packages/
    quarry-core/       # Contracts: Artifact, Connector, Operator, Executor, Check (ZERO DEPS)
    quarry-registry/   # DuckDB-backed artifact + run registry
    quarry-connectors/ # Connector implementations
    quarry-operators/  # Operator implementations
    quarry-cli/        # CLI adapter (argparse, depends on all four above)
```

## Canonical Commands

```sh
just test F        # Run a targeted pressure test file or subset
just test-all      # Run the full pressure gate
just test-file F   # Alias for targeted test execution
just lint          # Ruff check
just fmt           # Ruff format
just lock          # uv lock
just tree          # Show package dependency graph
```

## Toolchain

- `uv` for package management (workspace mode)
- `ruff check` + `ruff format` (target-version = py310)
- `ty check` for type checking
- `pytest -q` for tests
- `just` for canonical commands
- No packages < 7 days old

## Behavior Rules

Prime directive: Preserve the ontology. Do not invent parallel abstractions.

1. **Lane declaration required.** Every change must name its lane (connector, artifact, operator, flow, executor, check, adapter, registry). If it can't, it's not ready.
2. **No contract changes without adversarial proof.** If a contract feels weak, prove it with a pressure test first. Do not generalize from smell alone.
3. **Prefer the smallest change that satisfies the test.** Do not refactor surrounding code. Do not add features. Do not add "while I'm here" improvements.
4. **No hidden pathways.** Everything enters through a connector. Everything transforms through an operator. Everything executes through an executor. Everything persists through the registry.
5. **Output metadata is always fresh.** Read it from the actual output file. Never copy from input.
6. **Run targeted pressure tests before claiming completion.** Use `just test <target>` for iteration. Use `just test-all` only as a deliberate full-surface gate.
7. **Do not bypass the registry.** If an artifact exists, it must be registered. If a run happens, it must be recorded.
8. **Do not generalize on anticipation.** Only build what is needed now. The 3x rule applies: no abstraction until the same code appears three times.
9. **Do not introduce new dependencies to quarry-core.** It must remain zero-dep.
10. **Update docs when ontology changes.** If a contract changes, update CONTRACTS.md and PRESSURE_TESTS.md.

## What NOT to build

- UI/viewer
- Workflow builder / drag-and-drop
- Chat interface
- Full serving stack
- Plugin marketplace
- MCP integration
- Distributed execution (until forced by real workload)
- Agent layer (until substrate is complete)

## Scheduled Debt

- `OperatorSpec.output_type` singular — multi-output deferred until tile-splitting
- `Lineage` single object — may need append-only for multi-stage provenance
- FillDepressions pure-Python loops — numba acceleration deferred until perf measured
- Flat gradient uses naive BFS — Barnes et al. (2015) optimal flat resolution deferred
- ZonalStats per-zone rasterization O(zones×pixels) — vectorized groupby deferred until perf measured
- SpatialJoin O(left×right) brute force — STRtree spatial index deferred until perf measured
- SpatialJoin only supports `intersects` predicate — `contains`, `within`, `touches` deferred
- BuildCOG only tested with GeoTIFF input — other rasterio formats untested
- RasterizeVector only tested with polygons — line/point rasterization deferred
- RasterizeVector single-band only — multi-band output deferred
- RasterizeVector no all_touched option — deferred until needed
- CLI plain text output only — JSON mode deferred until needed
- Operator string params (`compress`, `resampling`, `predicate`) validated against hardcoded tuples — `Literal` types deferred (large surface area)
- `HydrologyFlow._execute_step` mutates input lists AND returns a value — mixed contract, single caller, low urgency
- Semantic product connectors (FOFStack, PIXC, SLC, Sentinel2) are not auto-routed by generic extensions/catalog strings — require explicit connector use until a semantic SourceRef pressure test forces routing
- `WaterElevationMosaic._fill_water_mask` iterative dilation O(n×max(h,w)) — scipy.ndimage or numba deferred until perf measured on real SWOT tiles
- `WaterElevationMosaic._resample_to_grid` uses array-index nearest-neighbor — coordinate-aware resampling deferred until multi-extent inputs tested
- `GeocodeSLC._find_bracket` + `_range_doppler_to_latlon` pure-Python per-pixel bisection — vectorized or C-extension deferred until perf measured

## Substrate Phase — COMPLETE

Substrate phase is complete. All criteria met:
- Core ontology stable across the pressure surface
- 29 connectors:
  - COG
  - CSVXY
  - DuckDB
  - ExcelXY
  - FlatGeobuf
  - FOFStack
  - GeoJSONSeq
  - GeoPackage
  - GeoParquet
  - GPX
  - HDF5
  - KMZ
  - LASPointCloud
  - LocalFile
  - MBTiles
  - NetCDF
  - ObjectStore
  - OGCServices
  - OpenTopography
  - Overture
  - PIXC
  - PostGIS
  - Sentinel2
  - Shapefile
  - SLC
  - SpatiaLite
  - STAC
  - TopoJSON
  - Zarr
- 16 operators:
  - Aspect
  - BuildCOG
  - ClipRaster
  - D8FlowDirection
  - FillDepressions
  - FlowAccumulation
  - GeocodeSLC
  - Hillshade
  - RasterizeVector
  - Reproject
  - SampleRaster
  - SLCCalibration
  - Slope
  - SpatialJoin
  - WaterElevationMosaic
  - ZonalStats
- 1 flow: HydrologyFlow (fill → D8 → accumulation)
- 1 executor: LocalExecutor
- ConnectorRouter for default CLI source-ref selection with extension/scheme/prefix filters
- DuckDB-backed registry with lineage graph
- End-to-end flow works (HydrologyFlow + zonal stats + COG export)
- Use `just stats` for current collected test counts

## Raiding source: hydrops/

`hydrops/` lives in repo root as an unintegrated reference codebase. It is NOT a package.
Trimmed 2026-04-23: all dead reports, analysis scripts, demo scripts, CSV/JSON data, and PDFs removed. What remains is extractable source, reference docs, and supporting tests.

**Extractable targets (ranked by substrate value):**
1. Check patterns — `contracts/backend_compliance.py`, `evals/eval5_accum.py` (conservation residuals, seam mismatch detection)
2. Tile scheduler concepts — `topology/tilestagegraph.py`, `local_scheduler.py` (multi-stage dependency graph, memory-bounded batching)
3. D8 boundary export — `engine.py` (tile-to-global flow stitching, boundary flow ring export)
4. Backend protocol — `contracts/backend_contract.py`, `contracts/backend_protocol.py` (MVB validation, runtime-checkable engine interface)

**Reference docs (read when extracting):**
- `docs/backend_boundary_note.md` — normative spec for backend compliance
- `docs/ARCHITECTURE_TILEGRAPH.md` — TileGraph concept and execution model
- `docs/ARCHITECTURE_MEMORY_AWARE_PIPELINE.md` — memory budget formulas

**Rules:** Extract one piece at a time. Rewrite to fit quarry contracts. Pressure-test before merging. Do NOT import wholesale.

## Commit Conventions

- Imperative mood, ≤72 char subject
- One logical change per commit
- Prefix with package name when relevant: `quarry-core: add Check protocol`

## Branch Strategy

- `main` — stable
- Feature branches off main
- No long-lived branches

## Related Docs

- `CONTRACTS.md` — human-readable contract semantics
- `REPO_MAP.md` — package ownership and file layout
- `PRESSURE_TESTS.md` — test history and ontology evolution log
- `HYDROLOGY_PACK.md` — D8 hydrology chain reference (operators, checks, invariants, limitations)
- `examples/watershed_analysis.py` — canonical end-to-end workflow (ingest→process→analyze→export→inspect)

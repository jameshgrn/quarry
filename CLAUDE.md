# Quarry

Canonical geospatial execution substrate.

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

- Preserve the ontology unless a pressure test breaks it
- Prefer the smallest change that satisfies the test
- Do not introduce new abstractions without naming the lane
- Do not generalize on smell alone — prove weakness with adversarial test
- Output metadata always fresh from actual file, never copied from input
- Run tests before proposing completion
- Do not bypass the registry
- Do not introduce hidden pathways around connectors/operators
- Do not add dependencies to quarry-core (must remain zero-dep)
- Update CONTRACTS.md and PRESSURE_TESTS.md when ontology changes

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
- CLI exposes hydrology + zonal + sample + rasterize flows — generic operator dispatch deferred
- CLI plain text output only — JSON mode deferred until needed
- Operator string params (`compress`, `resampling`, `predicate`) validated against hardcoded tuples — `Literal` types deferred (large surface area)
- `HydrologyFlow._execute_step` mutates input lists AND returns a value — mixed contract, single caller, low urgency

## Substrate Phase — COMPLETE

Substrate phase is complete. All criteria met:
- Core ontology stable across the pressure surface
- 4 connectors: LocalFile, STAC, PostGIS, COG
- 10 operators: ClipRaster, Reproject, FillDepressions, D8FlowDirection, FlowAccumulation, ZonalStats, SpatialJoin, BuildCOG, SampleRaster, RasterizeVector
- Registry persists artifacts/runs/checks/lineage
- End-to-end flow works (HydrologyFlow + zonal stats + COG export)
- Use `just stats` for current collected test counts

## v0.2.0 Milestone — Consolidation & Legibility — COMPLETE

- Canonical example: `examples/watershed_analysis.py` (ingest→process→analyze→export→inspect)
- Docs refreshed: AGENTS.md, CONTRACTS.md, REPO_MAP.md current with actual state
- No new contracts, operators, or connectors — consolidation only

## v0.3.0 Milestone — CLI Adapter

- `quarry-cli` package: minimal CLI invocation surface (lane: adapter)
- Commands: `quarry artifacts list/show`, `quarry lineage`, `quarry runs list/show`, `quarry checks show`, `quarry run hydrology`, `quarry run zonal`, `quarry run sample`, `quarry run rasterize`
- Zero new dependencies (argparse only)
- Adapter behavior is pressure-tested across the CLI surface

## What NOT to build yet

- UI/viewer
- Workflow builder
- Chat interface
- Full serving stack
- Plugin marketplace
- MCP integration
- Distributed execution (until forced)
- Agent layer (until substrate complete)

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
- `AGENTS.md` — agent behavior rules and debt list
- `HYDROLOGY_PACK.md` — D8 hydrology chain reference (operators, checks, invariants, limitations)
- `examples/watershed_analysis.py` — canonical end-to-end workflow (ingest→process→analyze→export→inspect)

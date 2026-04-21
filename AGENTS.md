# Agent Behavior Rules

Rules for any agent (Claude, Codex, FirePass, dgov workers) operating in this repo.

## Prime directive

Preserve the ontology. Do not invent parallel abstractions.

## Rules

1. **Lane declaration required.** Every change must name its lane (connector, artifact, operator, flow, executor, check, adapter, registry). If it can't, it's not ready.

2. **No contract changes without adversarial proof.** If a contract feels weak, prove it with a pressure test first. Do not generalize from smell alone.

3. **Prefer the smallest change that satisfies the test.** Do not refactor surrounding code. Do not add features. Do not add "while I'm here" improvements.

4. **No hidden pathways.** Everything enters through a connector. Everything transforms through an operator. Everything executes through an executor. Everything persists through the registry.

5. **Output metadata is always fresh.** Read it from the actual output file. Never copy from input.

6. **Run tests before claiming completion.** `just test` must pass.

7. **Do not bypass the registry.** If an artifact exists, it must be registered. If a run happens, it must be recorded.

8. **Do not generalize on anticipation.** Only build what is needed now. The 3x rule applies: no abstraction until the same code appears three times.

9. **Do not introduce new dependencies to quarry-core.** It must remain zero-dep.

10. **Update docs when ontology changes.** If a contract changes, update CONTRACTS.md and PRESSURE_TESTS.md.

## What not to build

- UI/viewer
- Workflow builder / drag-and-drop
- Chat interface
- Full serving stack
- Plugin marketplace
- MCP integration
- Distributed execution (until forced by real workload)
- Agent layer (until substrate is complete)

## Debt list (tolerated, not fires)

- `source_ref: str` in Connector protocol — SourceRef type exists, ConnectorRouter handles routing, protocol update deferred
- `OperatorSpec.output_type` is singular — will need multi-output when tile-splitting appears
- `Lineage` on Artifact is a single object — may need append-only for multi-stage provenance
- Legacy `src/georuntime/` — migration deferred until packages are stable
- FillDepressions pure-Python loops — numba acceleration deferred until perf measured
- Flat gradient uses naive BFS — Barnes et al. (2015) optimal flat resolution deferred
- ZonalStats per-zone rasterization O(zones×pixels) — vectorized groupby deferred until perf measured
- SpatialJoin O(left×right) brute force — STRtree spatial index deferred until perf measured
- SpatialJoin only supports `intersects` predicate — `contains`, `within`, `touches` deferred
- BuildCOG only tested with GeoTIFF input — other rasterio formats untested
- RasterizeVector only tested with polygons — line/point rasterization deferred
- RasterizeVector single-band only — multi-band output deferred
- RasterizeVector no all_touched option — deferred until needed

## Raiding source: hydrops/

`hydrops/` lives in repo root as an unintegrated reference codebase. It is NOT a package.

**Extractable targets (ranked by substrate value):**
1. Check patterns — backend compliance, conservation residuals, seam mismatch detection
2. Tile scheduler concepts — TileStageGraph, memory-bounded batching (future executor)

**Rules:** Extract one piece at a time. Rewrite to fit quarry contracts. Pressure-test before merging. Do NOT import wholesale.

## Current status (v0.2.0)

Substrate phase is complete. The current surface:
- 4 connectors: LocalFile, STAC, PostGIS, COG
- 10 operators: ClipRaster, Reproject, FillDepressions, D8FlowDirection, FlowAccumulation, ZonalStats, SpatialJoin, BuildCOG, SampleRaster, RasterizeVector
- 1 flow: HydrologyFlow (fill→D8→accumulation)
- 1 executor: LocalExecutor
- ConnectorRouter for source-ref-based connector selection
- DuckDB-backed registry with lineage graph
- 481 tests passing, 18 pressure test suites, zero contract changes

See `examples/watershed_analysis.py` for a canonical end-to-end workflow.

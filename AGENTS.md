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

6. **Run targeted pressure tests before claiming completion.** Use `just test <target>` for iteration. Use `just test-all` only as a deliberate full-surface gate.

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

- `OperatorSpec.output_type` is singular — will need multi-output when tile-splitting appears
- `Lineage` on Artifact is a single object — may need append-only for multi-stage provenance
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

## Current status

Substrate phase is complete. The current surface:
- 5 connectors: LocalFile, STAC, PostGIS, COG, DuckDB
- 12 operators: ClipRaster, Reproject, FillDepressions, Slope, Aspect, D8FlowDirection, FlowAccumulation, ZonalStats, SpatialJoin, BuildCOG, SampleRaster, RasterizeVector
- 1 flow: HydrologyFlow (fill→D8→accumulation)
- 1 executor: LocalExecutor
- ConnectorRouter for source-ref-based connector selection
- DuckDB-backed registry with lineage graph
- Pressure-test counts change; use `just stats` for the current surface

See `examples/watershed_analysis.py` for a canonical end-to-end workflow.

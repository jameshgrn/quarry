# Pressure Test Log

Each entry records what was tested, what was learned, and whether contracts needed changes.

## 1. Kernel (2026-04-20)

**Components:** LocalFileConnector + ClipRasterOperator + LocalExecutor
**Tests:** 15
**Contract changes:** None

**Proved:**
- Connector → Artifact pathway works
- Artifact identity ≠ path (two materializations of same file → distinct IDs)
- Lazy materialization works
- Operator validates inputs, rejects bad types and lazy handles
- Executor produces RunRecord with full lifecycle
- Checks attach to results naturally

**Signals watched:**
- Artifact vs run validation confusion → not triggered
- Single-output awkwardness → not triggered
- Mutability drift → not triggered
- Source ref too loose → not triggered (local paths are simple)
- Normalization/materialization confusion → not triggered

## 2. Registry (2026-04-20)

**Components:** DuckDB persistence for all contract objects
**Tests:** 18
**Contract changes:** None

**Proved:**
- Artifact serializes losslessly (backing store, spatial, metadata)
- Lazy artifacts persist and recover correctly
- RunRecord round-trips with status, timing, checks
- save_run cascades (artifact + checks + lineage)
- Check truth separation works (checks table, not embedded)
- Lineage edges form coherent graph (multi-generation ancestry)
- Stats queries work

## 3. STAC Connector (2026-04-20)

**Components:** STACConnector (adversarial remote/catalog source)
**Tests:** 22
**Contract changes:** None

**Proved:**
- source_ref parsing: collection/item, collection/item::asset, bare ID
- Asset selection: explicit > single > common keys > first geotiff > error
- Lazy gets full STAC metadata without downloading
- Eager downloads + hashes
- SpatialDescriptor populated from STAC extensions (proj:epsg, proj:transform, eo:bands)
- Lineage captures full STAC provenance
- Lazy STAC artifact survives registry round-trip

**Signals:**
- `source_ref: str` mild strain (:: convention for asset key) — scheduled debt, not blocking

## 4. Reproject Operator (2026-04-20)

**Components:** ReprojectOperator (raster + vector CRS transformation)
**Tests:** 19
**Contract changes:** None

**Proved:**
- CRS changes propagate to output spatial descriptor
- Extent transforms correctly (degrees → meters)
- Resolution recalculates
- Band count preserves
- Output metadata regenerated from actual file (not copied from input)
- Content hash differs from input (resampling changes data)
- Checks compose: crs_valid, crs_matches_target, extent_sane, backing_accessible
- Lineage captures: source_crs, target_crs, resampling, resolution override
- Lazy artifacts properly rejected
- Vector reprojection through same operator (feature count preserved)
- Full loop: materialize → reproject → execute → persist → recover → lineage query

## 5. PostGIS Connector (2026-04-20)

**Components:** PostGISConnector (adversarial connection-backed source)
**Tests:** 25
**Contract changes:** None

**Proved:**
- source_ref parsing: schema.table, bare table (default schema), SELECT query
- Geometry/non-geometry branching: VECTOR vs TABLE artifact type
- Lazy materialization → BackingStoreKind.POSTGIS with full spatial metadata
- Eager vector: dumps to GeoPackage via fiona (WKB → shapely → geo_interface)
- Eager table: dumps to CSV
- Metadata inspection: columns, geometry_type, SRID, extent, row_count
- Discover: lists tables in schema with geometry info
- Lineage captures PostGIS provenance (host, dbname, schema, table, query)
- PostGIS artifact survives registry round-trip (BackingStoreKind.POSTGIS persists)
- Error handling: connection failure, table-not-found

**Signals:**
- `source_ref: str` — **moderate strain**. Three distinct shapes (schema.table, bare table, query)
  parsed via heuristics (dot-split, SELECT prefix). Separator conflicts possible with dot-containing
  table names. Not broken, but meaningfully more awkward than STAC's `collection/item::asset`.
  SourceRef type would eliminate the heuristic parsing.
- Geometry type normalization needed (PostGIS: MULTILINESTRING → Fiona: MultiLineString) —
  handled internally, not a contract issue.
- BackingStoreKind.POSTGIS already existed in contracts — confirms forward-thinking design.
- ArtifactType.TABLE already existed — confirms vector/table branching is natural.

**Debt observed:**
- `source_ref: str` heuristic parsing is the clearest signal yet that SourceRef type
  is approaching "needed" rather than "nice to have." Two connectors now use convention-based
  parsing. A third would make the case definitive.

## 6. COG Connector (2026-04-20)

**Components:** COGConnector (adversarial range-request/validation source)
**Tests:** 24
**Contract changes:** None

**Proved:**
- source_ref as plain URI (path or URL) — trivial parsing, 4th distinct shape
- Local/remote branching via scheme classification
- COG validation: tiling + overviews → is_cog flag in metadata
- Strict mode rejects non-COGs; default mode accepts with flag
- Lazy = header-only (LAZY_HANDLE backing, zero data transfer)
- Eager local = wrap in place (same pattern as LocalFileConnector)
- I/O metrics in lineage: data_transferred, source_type
- COG-specific metadata: block_size, overview_levels, compression, dtype
- SpatialDescriptor fully populated from rasterio headers
- Non-COG GeoTIFFs still materialize (connector is lenient by default)
- Registry round-trip preserves LAZY_HANDLE COG artifacts

**Signals:**
- `source_ref: str` — **no additional strain**. COG uses plain URIs, trivially classified.
  But the *connector selection* problem is now visible: a local .tif can go through
  LocalFileConnector OR COGConnector. Nothing in the system helps pick which one.
  This is a different kind of strain than source_ref parsing.
- `BackingStoreKind.REMOTE_URI` never used — COG lazy uses LAZY_HANDLE (consistent with STAC).
  REMOTE_URI may be dead weight or may need a future HTTP-handle variant.
- COG connector largely subsumes LocalFileConnector for raster files.
  Specialization: LocalFile handles vectors, COG handles raster-specific concerns.

**Debt observed:**
- Connector selection: 4 connectors exist, no router/dispatcher. User must manually pick.
  Not a contract issue yet, but will become one when adapters (CLI, agents) need to auto-route.
- `BackingStoreKind.REMOTE_URI` unused — assess whether to keep or remove.

**Summary:** Six pressure tests, zero contract changes. Ontology stable at 4 connectors, 2 operators.
source_ref: str strain from PostGIS NOT repeated by COG — strain is connector-specific, not systemic.
The real emerging question is connector selection, not source_ref parsing.

## 7. SourceRef Contract (2026-04-20)

**Components:** SourceRef type in quarry-core (first contract evolution)
**Tests:** 34
**Contract changes:** YES — SourceRef added to quarry-core. Connector protocol unchanged.

**Proved:**
- Simple refs stay simple: `SourceRef.local("/path")`, `SourceRef.uri("https://...")`
- Complex refs gain clarity: `SourceRef.stac(collection, item, asset=)`, `SourceRef.postgis(schema, table)`
- Round-tripping works: construct → raw → same string, always
- `infer()` classifies raw strings reliably: paths, URLs, STAC patterns, SQL queries, schema.table
- Ambiguous inputs get UNKNOWN — inference is honest
- Backward compat proven: Connector protocol still takes str, SourceRef gives .raw
- Frozen, hashable, value-equality semantics

**Signals:**
- SourceRef justifies itself immediately for PostGIS and STAC (eliminates heuristic parsing)
- SourceRef is trivial for LocalFile and COG (thin wrapper, no benefit beyond classification)
- `infer()` handles the 80% case for routing; edge cases go UNKNOWN
- The overlap between "dotted name" (schema.table) and "dotted filename" (file.tif) is
  handled via file extension detection — pragmatic, not perfect

**Design choices made:**
- SourceRef is a utility type, NOT a protocol change. Connector.materialize stays str.
- Params dict is free-form (connector-specific) — no shared schema
- Frozen dataclass (immutable, hashable)
- Five kinds: LOCAL_PATH, REMOTE_URI, CATALOG_ITEM, DATABASE_REF, UNKNOWN
- Factory methods produce raw strings matching existing connector conventions

**Summary:** Seven pressure tests. First contract addition (SourceRef) to quarry-core.
Connector protocol unchanged. 157 total tests pass. Ontology enriched, not broken.

## 8. FillDepressions Operator (2026-04-20)

**Components:** FillDepressionsOperator (first hydrology domain operator)
**Tests:** 30
**Contract changes:** None

**Proved:**
- Priority-Flood (Wang & Liu 2006) implementation correct: single pits, multi-cell bowls, nested
- Boundary cells always treated as outlets (never raised)
- Elevation monotonically non-decreasing (never lowers cells)
- Zero interior pits remain after fill (correctness invariant holds on random 100x100)
- Nodata cells preserved through fill (masked, not processed)
- Flat gradient resolution via BFS creates D8-resolvable slopes
- Already-drained DEMs pass through unchanged (idempotent on valid input)
- Single-band enforcement (rejects multi-band rasters)
- Operator protocol fully satisfied (spec, validate, execute, declared_checks)
- Fresh metadata from output file (not copied from input)
- Lineage records algorithm parameters

**Signals:**
- Operator protocol handles domain-specific checks naturally (no_interior_pits, elevation_only_raised)
- Single-input single-output is comfortable for preprocessing ops
- ResourceScale.MEDIUM appropriate (O(n log n), can be expensive on large DEMs)
- numpy-only implementation (no numba) — sufficient for substrate proof, optimization later

**Debt observed:**
- Pure Python loops in Priority-Flood will be slow on large DEMs (>1000x1000).
  Numba acceleration deferred until performance is a measured problem.
- Flat gradient uses BFS from outlet edges — correct but naive. Barnes et al. (2015)
  is the canonical reference for optimal flat resolution. Deferred.

**Summary:** Eight pressure tests. Third operator added (fill_depressions). First hydrology
domain incision. Zero contract changes. 187 total tests passing.

## 9. D8 Flow Direction Operator (2026-04-20)

**Components:** D8FlowDirectionOperator (second hydrology operator, chains after FillDepressions)
**Tests:** 27
**Contract changes:** None

**Proved:**
- Steepest descent correctly picks direction across all 8 compass codes
- Diagonal distance weighting (sqrt(2)) affects direction selection appropriately
- Boundary cells with no lower neighbor → OUTLET code
- Nodata cells → NODATA code, never assigned direction
- Two-pass flat resolution: PIT cells on flat surfaces route to equal-elevation
  neighbors that can drain (iterative propagation until stable)
- Fill → D8 chain: zero PITs on both hand-crafted and random 50x50 DEMs
- Operator protocol fully satisfied
- Direction encoding documented in artifact metadata

**Signals:**
- Flat resolution requires iterative pass — single-pass D8 is insufficient for filled DEMs
- Operator sequence semantics emerging: fill_depressions → d8_flow_direction is a precondition chain
- Check "no_pits" uses WARN not INVALID — operator works on unfilled DEMs too, just flags the issue
- O(n) algorithm (two linear passes + iteration on PITs only) — fast enough for substrate

**Debt observed:**
- Pure Python loops same as FillDepressions — numba deferred
- Flat resolution iterates until convergence — worst case O(n) iterations on pathological flats.
  Could use BFS from draining cells instead. Deferred.
- No cycle detection yet — assumed acyclic after fill. Add if needed.

**Summary:** Nine pressure tests. Fourth operator added (d8_flow_direction). Hydrology chain
emerging: fill → D8. Zero contract changes. 214 total tests passing.

## 10. Flow Accumulation Operator (2026-04-20)

**Components:** FlowAccumulationOperator (third hydrology op, completes minimal spine)
**Tests:** 27
**Contract changes:** None

**Proved:**
- Topological sort (Kahn's) correctly accumulates upstream area
- Linear chains produce monotonically increasing accumulation
- Branching/confluence sums upstream contributions correctly
- Conservation: total outlet accumulation = total valid cell count × weight
- PIT cells retain self-weight, don't propagate downstream
- Nodata cells excluded from accumulation entirely
- Cycle detection rejects invalid flow grids (raises OperatorError)
- Custom weight parameter scales accumulation linearly
- Full chain: fill → D8 → accumulation on random 30x30 DEM (all checks pass)
- Conservation holds on simple sloped DEM through full chain

**Signals:**
- Operator protocol continues to absorb domain-specific checks naturally
- The conservation check is the first genuinely quantitative check in the system
  (not just existence/validity, but a mathematical invariant)
- Cycle detection as a hard failure (OperatorError) vs soft warning (pit check in D8)
  shows the check severity model working
- O(n) algorithm — linear in cell count, suitable for substrate

**Debt observed:**
- Same pure-Python loop performance debt as other hydrology ops
- Weight parameter could eventually be a per-cell raster (e.g., rainfall-weighted
  accumulation) — deferred until needed
- No watershed delineation yet — natural extension of accumulation (threshold + trace)

**Summary:** Ten pressure tests. Fifth operator (flow_accumulation). Hydrology spine complete:
fill → D8 → accumulation. Zero contract changes. 241 total tests passing.

## 11. Hydrology Flow Integration + InternalOutletCount (2026-04-20)

**Components:** HydrologyFlow (chain composition), InternalOutletCount (standalone check)
**Tests:** 42 (27 flow integration + 15 check)
**Contract changes:** None

**Proved:**
- Full chain: fill → D8 → accumulation composes through Executor + Registry
- Every intermediate artifact, run, check, and lineage edge persists correctly
- Lineage graph walkable: accumulation → D8 → filled → input DEM
- Conservation holds through full chain (pit DEM, sloped, random, custom weight)
- Check propagation: all operator checks present in flow result and registry
- Failure isolation: bad input stops chain at correct step, reports step name
- Registry round-trip: artifacts/runs survive save→load cycle
- Flow works without registry (pure execution mode)
- InternalOutletCount standalone check agrees with D8 operator-internal check
- Standalone check: zero false positives on clean grids, correctly detects nodata leaks
- Check protocol compliance: name, description, returns CheckResult

**Signals:**
- HydrologyFlow validates the Flow lane concept — composition through executor works naturally
- Standalone + operator-internal check duality works: same logic, different entry points
- Registry cascading save handles multi-step chains without manual intervention
- Failure isolation stops the chain cleanly — no partial output from failed steps

**Debt observed:**
- HydrologyFlow is hard-coded to three operators — no generic DAG composition yet
- Registry save happens inside the flow — no transactional rollback on partial failure

**Summary:** Eleven pressure tests. Hydrology chain composition + standalone check.
Zero contract changes. 283 total tests passing.

## 12. Adversarial DEM Fixtures (2026-04-20)

**Components:** Full hydrology chain on 27 pathological DEM surfaces
**Tests:** 27
**Contract changes:** None

**Proved:**
- Tiny DEMs (2×2 through 5×1): hand-verifiable accumulation values correct
- L-shaped nodata, nodata corners, boundary strip: chain handles irregular masks
- Thin diagonal channel: D8 routes through 1-cell-wide diagonal paths
- Sinuous channel: D8 handles winding drainage paths
- Narrow valley: accumulation concentrates at valley floor
- Raised rim with single spill point: all interior drains through constrained exit
- Boundary shelf: flat resolution handles flat region AT the grid boundary
- Nodata cross bisection: 4 disconnected quadrants each drain independently
- Single valid cell in nodata sea: degenerate case handles correctly (acc=1)
- Two adjacent valid cells: minimal drainage pair works (higher→lower)
- 15×15 flat DEM: flat resolution assigns directions from interior to boundary
- Checkerboard nodata: maximally fragmented mask doesn't crash chain
- Scattered nodata holes: random 10% nodata on sloped surface preserves conservation
- Corner-to-corner slope: diagonal drainage routes to SE corner correctly
- Saddle point: flow splits between two valleys
- Ridge line: flow splits east and west of ridge

**Limitation documented:**
- Priority-Flood cannot reach valid cells disconnected from grid boundary by nodata.
  Island test documents this: chain completes, conservation holds, but interior pits
  on disconnected islands remain unfilled. D8 assigns PIT codes. Not a bug — a known
  algorithm boundary condition.

**Signals:**
- Conservation invariant holds across all 27 pathological surfaces (where applicable)
- Flat resolution handles boundary shelves, single-spill-point plateaus
- Nodata mask handling is robust across irregular shapes and fragmentation patterns
- Degenerate single-cell and two-cell cases don't crash or produce NaN

**Summary:** Twelve pressure tests. 27 adversarial fixtures. One known limitation
documented (disconnected valid regions). Zero contract changes. 320 total tests passing.

## 13. SpatialJoin Operator (2026-04-21)

**Components:** SpatialJoinOperator (vector × vector spatial join)
**Tests:** 20
**Contract changes:** None

**Proved:**
- Left join semantics: all left features preserved, unmatched get null right attrs
- One-to-many: left duplicated per matching right (correct cardinality)
- Many-to-many: cross-product of overlapping pairs (2×2 → 4 output)
- No-overlap: left features kept with null right columns
- Empty geometries: preserved in output, treated as no-match
- Empty right layer: left features pass through with no right columns
- CRS mismatch rejected at validation
- Schema collision: colliding right columns renamed with `_right` suffix
- Point-in-polygon: intersects predicate handles mixed geometry types
- Unsupported predicate rejected at validation
- Output is VECTOR (preserves geometry), not TABLE
- Fresh metadata from actual output file
- Lineage records predicate and collision renames
- Left join invariant: output count >= left count (check enforced)

**Signals:**
- Operator protocol absorbs vector-vector ops naturally (no protocol changes needed)
- OperatorSpec with two VECTOR inputs works cleanly (same pattern as ZonalStats raster+vector)
- Schema collision is a WARN not INVALID — resolved automatically, flagged for awareness
- GeoJSON driver discards schema on empty layers — operator handles gracefully

**Debt observed:**
- Only `intersects` predicate supported (v1). `contains`, `within`, `touches` deferred.
- Right features loaded into memory (no spatial index). O(left × right) — acceptable
  for substrate, spatial index (STRtree) deferred until perf measured.
- No inner/right/cross join modes — left join only for v1.

**Summary:** Thirteen pressure tests. Seventh operator (spatial_join). First vector×vector
operator. Zero contract changes. 361 total tests passing.

## 14. ZonalStats Operator (2026-04-21)

**Components:** ZonalStatsOperator (raster + vector → table)
**Tests:** 21
**Contract changes:** None

**Proved:**
- Per-zone raster statistics (count, sum, mean, min, max, std) correct on hand-verifiable grids
- CRS mismatch rejected at validation
- Empty geometries produce NaN rows, row count preserved
- All-nodata zone produces NaN row
- Partial overlap — stats computed only for covered pixels
- Zone fully outside raster → NaN row
- Schema always complete — all stat columns present regardless of data
- Row count always equals input feature count (stable)
- Single-pixel zone returns exact pixel value
- Multi-band raster — band param selects correct band
- Nodata pixels excluded (both numeric and NaN nodata)
- zone_id_field extracts zone ID from feature properties
- Lineage records operation params (band, zone_id_field, stat_columns)
- Output is TABLE (CSV), fresh metadata from actual file

**Signals:**
- Operator protocol handles mixed-type inputs (RASTER + VECTOR) cleanly
- OperatorSpec with two different input types works naturally
- Per-zone rasterization via geometry_mask is correct but O(zones × pixels)

**Debt observed:**
- Per-zone rasterization is O(zones × pixels) — vectorized groupby deferred until perf measured

**Summary:** Fourteen pressure tests. Sixth operator (zonal_stats). First raster+vector
cross-type operator. Zero contract changes. 361 total tests passing.

## 15. BuildCOG Operator (2026-04-21)

**Components:** BuildCOGOperator (raster → COG normalization)
**Tests:** 22
**Contract changes:** None

**Proved:**
- Non-COG GeoTIFF → valid COG (tiled + overviews)
- Base resolution pixel values identical after COG build (lossless)
- CRS preserved unchanged
- Nodata preserved (numeric and NaN)
- No-nodata rasters handled correctly (None → None)
- Multi-band preservation (count and per-band data)
- Already-a-COG idempotence — valid COG in, valid COG out, data unchanged
- Tiny raster smaller than blocksize — no crash, data preserved
- Overview levels computed and present on large rasters
- Compression applied — deflate output smaller than uncompressed
- Lazy artifact rejected at validation
- Unsupported compression rejected at validation
- Lineage records blocksize, compress, overview_resampling
- Fresh metadata from actual output file

**Signals:**
- Operator protocol handles representation transforms naturally — no protocol changes needed
- "Same data, different storage contract" fits cleanly under existing Operator semantics
- BuildCOG completes the ingest → process → export story
- The distinction between semantic and representation transforms is visible but
  does not require formalization — both are just Operators

**Debt observed:**
- Only GeoTIFF input tested — other rasterio-readable formats (NetCDF, HDF5) untested
- COG layout uses rasterio.shutil.copy with copy_src_overviews — relies on GDAL COG driver behavior
- No tile-level validation (checking internal IFD ordering) — trusting GDAL's COG layout

**Summary:** Fifteen pressure tests. Eighth operator (build_cog). First representation/
normalization operator. Zero contract changes. 383 total tests passing.

## 16. SampleRaster Operator (2026-04-21)

**Components:** SampleRasterOperator (raster + vector points → table)
**Tests:** 22
**Contract changes:** None

**Proved:**
- Point sampling returns exact pixel values at known locations
- CRS mismatch rejected at validation
- Points outside raster extent → NaN, row count preserved
- Nodata cells (numeric and NaN) → NaN in output
- Explicit band selection — pick subset of bands from multiband raster
- Empty bands param samples all bands (default behavior)
- Row count always equals input point count (stable invariant)
- Empty input layer → zero rows, schema check WARN
- Single point at pixel center → exact value
- Point near raster boundary edge → valid sample
- NaN nodata handling (isnan comparison)
- Nodata override via params (overrides raster native nodata)
- Schema always complete: point_id + band_N columns
- Lineage records bands and nodata_value params
- Output is TABLE (CSV), fresh metadata from actual file
- point_id sequential 0..N-1 regardless of sample success

**Signals:**
- Operator protocol handles point-based sampling naturally — no protocol changes needed
- TABLE output for point samples is correct choice (no geometry in output, just values)
- ResourceScale.LIGHT appropriate (single rasterio read per point per band)

**Debt observed:**
- Per-point window read may be slow for large point sets — batch read with array indexing deferred
- Only point geometries supported — centroid sampling for polygons/lines deferred

**Summary:** Sixteen pressure tests. Ninth operator (sample_raster). Second raster+vector
cross-type operator. Zero contract changes. 405 total tests passing.

## 17. ConnectorRouter (2026-04-21)

**Components:** ConnectorRouter (connector selection/routing layer)
**Tests:** 34
**Contract changes:** YES — ConnectorRouter added to quarry-core. Connector protocol unchanged.

**Proved:**
- Local GeoTIFF ambiguity: both COG and LocalFile match, COG ranked first (priority 0 vs 10)
- Remote COG URIs (https, s3, gs): only COG matches
- STAC catalog items: only STAC matches (collection/item, with asset)
- PostGIS refs: only PostGIS matches (schema.table, SQL query)
- Unknown/unsupported source: only fallback connectors match (ranked +1000)
- No fallback → empty result for UNKNOWN
- select_one raises NoConnectorError with kind and raw in message
- Raw string backward compat: strings auto-inferred via SourceRef.infer()
- Priority ordering: lower number = higher rank, custom priorities reverse default order
- ConnectorMatch is sortable by rank
- Registration introspection for debugging
- Edge cases: empty string, SourceRef passthrough (not re-inferred), stub protocol compliance

**Signals:**
- Router resolves the "connector selection" debt from pressure test #6 (COG connector)
- SourceRef.infer() + kind-affinity registration is sufficient for v1 routing
- No connector protocol changes needed — router operates alongside, not inside
- Fallback +1000 penalty ensures kind matches always beat fallbacks

**Design choices made:**
- Router lives in quarry-core (zero deps, operates on protocol types only)
- Lane: registry (remembers what connectors exist, selects among them)
- Registration-based, not introspection-based (connectors don't need `can_handle`)
- Priority is caller-controlled, not auto-derived
- No execution — callers use the match to call materialize themselves

**Summary:** Seventeen pressure tests. ConnectorRouter added to quarry-core.
Connector protocol unchanged. 439 total tests passing.

## 18. RasterizeVector Operator (2026-04-21)

**Components:** RasterizeVectorOperator (vector polygons → raster grid)
**Tests:** 25
**Contract changes:** None

**Proved:**
- Constant burn: polygon burned at specified value, background at nodata
- Attribute burn: per-feature numeric property burned, different polygons get different values
- CRS preserved from vector input to output raster
- Empty geometries skipped without crash; valid polygons still burned
- Polygons partially outside extent clipped to grid — only in-grid pixels burned
- Nodata/background: uncovered pixels == nodata value, covered pixels != nodata
- Grid alignment: dimensions = ceil(extent_span / resolution), verified exact
- Explicit extent overrides vector bounding box
- No extent → derives from vector bounds automatically
- Missing burn attribute → feature skipped, others still burned
- Non-numeric burn attribute → feature skipped gracefully
- Invalid resolution (zero/negative) rejected at validation
- Invalid extent (degenerate) rejected at validation
- Neither burn_value nor burn_attribute → validation error
- Zero-feature vector → all-nodata raster (no crash)
- Overlapping polygons → last-write-wins (rasterio default behavior)
- Small resolution on large extent → large but valid grid
- Wrong input type rejected (raster passed as vector)
- Unmaterialized input rejected (lazy handle)
- Lineage records all params (resolution, extent, burn_value, burn_attribute, nodata, dtype)
- Output metadata fresh from actual file (not params echo)
- All declared checks pass on happy path (crs_valid, dimensions_sane, nodata_background)

**Signals:**
- Operator protocol handles vector→raster conversion naturally — no protocol changes needed
- Single VECTOR input, single RASTER output — simplest spec shape
- rasterio.features.rasterize does the heavy lifting; operator adds artifact/lineage/check wrapping
- dtype parameter exposes control over output precision (uint8 through float64)

**Debt observed:**
- Only polygon geometries tested — line/point rasterization deferred
- No all_touched option (rasterio supports it) — deferred until needed
- No multi-band output — single band only for v1

**Summary:** Eighteen pressure tests. Tenth operator (rasterize_vector). First vector→raster
operator. Zero contract changes. 464 total tests passing.

## 19. CLI Adapter (2026-04-21)

**Components:** quarry-cli package (argparse CLI over existing substrate)
**Tests:** 19
**Contract changes:** None

**Lane:** adapter

**Proved:**
- `artifacts list` renders empty and populated registries correctly
- `artifacts list --type` filters by ArtifactType
- `artifacts show` displays full artifact detail; returns 1 for missing ID
- `lineage` walks full ancestor chain; shows "no ancestors" for root artifacts
- `lineage` returns 1 for missing artifact
- `run hydrology` end-to-end: DEM in → filled + D8 + accumulation out + registry populated
- `run hydrology` with `--no-gradient` and `--weight` flag variants
- `run hydrology` returns 1 for missing DEM path
- `--workspace` flag respected across all commands (registry + outputs land in specified dir)
- Full round-trip: run → list → show → lineage through CLI entry point
- No-command and missing-subcommand print help, return 0
- Parser construction and prog name correct

**Signals:**
- Existing substrate primitives wire to CLI without any protocol changes
- Registry, LocalFileConnector, HydrologyFlow, LocalExecutor compose cleanly from outside
- argparse is sufficient — no click/typer dependency needed
- CLI is pure glue: 200 lines, zero new abstractions

**Debt observed:**
- Only HydrologyFlow exposed via `run` — generic operator dispatch deferred
- No JSON output mode — plain text tables only for v1
- No `run list` / `run show` commands — deferred until someone needs run inspection from CLI

**Summary:** Nineteen pressure tests. First adapter-lane package (quarry-cli).
Zero contract changes. 483 total tests passing.

## 20. CLI Zonal Flow (2026-04-21)

**Components:** `quarry run zonal` CLI command (raster + polygon zones → CSV)
**Tests:** 12
**Contract changes:** None

**Lane:** adapter

**Proved:**
- `run zonal` end-to-end: raster + zones → CSV output + registry populated
- Output CSV has correct schema (zone_id + 6 stat columns) and row count matches zones
- Registry contains 3 artifacts (raster + zones + output table)
- Lineage: output table has 2 ancestors (raster + zones)
- `--band` flag selects correct raster band (verified with multiband raster)
- `--zone-id-field` extracts named zone IDs from feature properties
- `--workspace` flag respected (output + registry land in specified dir)
- Returns 1 for missing raster path
- Returns 1 for missing zones path
- Nodata pixels excluded from statistics
- Full round-trip: run zonal → artifacts list → artifacts show → lineage
- `--type table` filter shows exactly 1 artifact after zonal run

**Signals:**
- Two-input operator (raster + vector) wires to CLI as naturally as single-input flow
- No flow composition class needed — CLI directly wires connector → executor → operator → registry
- GeoPackage fixtures needed (GeoJSON normalizes CRS to 4326 per spec)
- The substrate's connector → operator → registry pathway works identically for single-step and multi-step flows

**Debt observed:**
- Generic operator dispatch still deferred — each flow is hand-wired in CLI
- No `run sample` command yet (SampleRaster is the obvious third flow)

**Summary:** Twenty pressure tests. Second CLI flow (zonal stats). Exercises two-input
operator pattern through CLI. Zero contract changes. 495 total tests passing.

## 21. CLI Inspection Commands (2026-04-21)

**Components:** CLI adapter (runs list/show, checks show) + Registry
**Tests:** 20
**Contract changes:** None

**Proved:**
- `runs list` shows table of runs from registry with ID, operator, status, submitted, duration
- `runs list --status completed` filters correctly; `--status failed` returns empty on clean run
- `runs list --limit 1` respects limit
- `runs show <run-id>` displays full run detail: operator, status, timing, inputs, params, output, checks
- Output artifact resolved via direct DB query (workaround for `get_run()` not reconstructing OperatorResult)
- Returns 1 for nonexistent run ID
- `checks show <artifact-id>` displays checks for artifact with state, name, message, timestamp
- `checks show <run-id>` displays checks for run
- Returns 1 for nonexistent ID (neither artifact nor run)
- Artifact with no checks returns 0 with "(no checks)" message
- `runs` and `checks` with no subcommand return 0 (help)
- Full round-trip: run hydrology → runs list → runs show → checks show

**Signals:**
- Registry already had `list_runs()`, `get_run()`, `get_checks()` — pure adapter work, no substrate changes
- `get_run()` doesn't reconstruct `output` field — CLI works around via direct `output_artifact_id` query
- Auto-detection of artifact vs run ID works cleanly for `checks show`

**Debt observed:**
- Registry `_row_to_run()` should reconstruct output OperatorResult for full fidelity
- CLI plain text only — JSON output mode still deferred

**Summary:** Twenty pressure tests. Three new CLI inspection commands. Zero contract changes. 515 total tests passing.

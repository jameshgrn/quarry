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

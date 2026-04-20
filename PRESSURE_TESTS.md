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

**Summary:** Four pressure tests, zero contract changes. Ontology is stable.

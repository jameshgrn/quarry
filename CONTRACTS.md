# Quarry Contracts

Canonical definitions. If the code and this doc disagree, the code wins — but update this doc.

## Artifact

The internal unit of truth. NOT a file.

- **Identity** is the `id` field (UUID). Not the path. Not the name.
- **BackingStore** describes where data physically lives (local_file, remote_uri, lazy_handle, memory, duckdb, postgis). This is NOT identity.
- **SpatialDescriptor** holds CRS, extent, resolution, feature_count, band_count. Populated from actual data, not guessed.
- **Lineage** records how the artifact was created (operation, input IDs, params, timestamp).
- **Checks** accumulate from the checks table. Artifact exposes `validation_state` derived from accumulated check results.
- **Metadata** is an extensible dict for driver info, domain tags, platform info, etc.

Mutability rules:
- `id`, `type`: immutable
- `backing`: mostly immutable (set once at creation)
- `lineage`: immutable (set at creation)
- `checks`: appendable (loaded from registry)
- `metadata`: mutable (tags, annotations)

## Connector

The sacred gateway. No geospatial object enters except through a connector.

- **One required method**: `materialize(source_ref, workspace, lazy=False) -> MaterializeResult`
- **Materialize does NOT always mean download.** It can mean: copy, wrap in place, stage, create a lazy handle, or normalize format.
- **MaterializeResult** carries: artifact + strategy + source_ref + notes. Strategy is provenance ("wrapped_local", "fetched_remote", "lazy_handle", "normalized").
- **Capabilities** are explicit flags: MATERIALIZE, DISCOVER, AUTHENTICATE, STREAM, MATERIALIZE_LAZY, METADATA_ONLY.
- **Optional protocols**: Discoverable, Authenticatable, MetadataEmitter.

`source_ref` remains `str` in the Connector protocol (backward compat). `SourceRef` type exists
as an explicit construction/routing utility — see below.

## SourceRef

Typed envelope for source references. Lives in quarry-core (zero deps).

- **raw**: the original string, always preserved, always round-trippable via `str(ref)`
- **kind**: classification tag (LOCAL_PATH, REMOTE_URI, CATALOG_ITEM, DATABASE_REF, UNKNOWN)
- **params**: optional parsed fields (connector-specific structure)

Factory methods:
- `SourceRef.local(path)` → LOCAL_PATH
- `SourceRef.uri(url)` → REMOTE_URI
- `SourceRef.stac(collection, item, asset=)` → CATALOG_ITEM
- `SourceRef.postgis(schema, table)` → DATABASE_REF
- `SourceRef.postgis_query(sql)` → DATABASE_REF
- `SourceRef.infer(raw)` → best-effort classification from raw string

What SourceRef is NOT:
- Not a replacement for `source_ref: str` in the protocol
- Not a connector selector/router
- Not a validator (bad refs are still representable)
- Not a class hierarchy

Connector protocol still receives `str`. Callers pass `ref.raw` or `str(ref)`.
SourceRef helps callers construct refs explicitly and helps routing layers classify them.

## ConnectorRouter

Selection layer: given a SourceRef (or raw string), returns ranked eligible connectors.
Lives in quarry-core (zero deps). Lane: registry.

- **Input**: `SourceRef | str` — raw strings auto-inferred via `SourceRef.infer()`
- **Output**: `list[ConnectorMatch]` — ranked by priority (lower = better)
- **No execution** — selection only. Does not call `materialize`.
- Connectors register with explicit `kinds: set[SourceRefKind]` + `priority: int`
- `fallback=True` connectors also match UNKNOWN refs (ranked +1000)
- `select()` returns all matches sorted by rank
- `select_one()` returns best match or raises `NoConnectorError`

Precedence rules (with standard registration):
- Local GeoTIFF → COG (priority 0) + LocalFile (priority 10) — both match, COG preferred
- Remote URI → COG only
- STAC catalog item → STAC only
- Database ref → PostGIS only
- Unknown → fallback connectors only (if any registered)

What ConnectorRouter is NOT:
- Not an executor (does not call materialize)
- Not a factory (does not create connector instances)
- Not a protocol change (Connector protocol unchanged)

## Operator

A typed transformation: artifacts in, artifact out.

- **Declares** accepted input types, output type, min/max inputs, resource scale.
- **validate_inputs()** returns list of error strings (empty = valid). Called before execute.
- **execute()** runs the transformation, returns `OperatorResult` (artifact + checks + warnings + timing).
- **declared_checks()** lists check names the operator runs on its output.
- Output artifact gets fresh spatial metadata from the actual output file — never copied from input.
- Output artifact gets its own identity, lineage, and backing store.

## Executor

Dispatches operator execution to a compute backend.

- **submit()** takes operator + inputs + params, returns RunRecord.
- **status()** and **wait()** query/block on run completion.
- **RunRecord** is the full lifecycle object: pending → running → completed/failed, with timing, checks, error info.
- Executor does not care what the operator does. It just runs it and captures the result.
- LocalExecutor is synchronous. Future executors (Dask, SLURM) may be async.

## Check

Validation rule applied to an artifact.

- **run(artifact) -> CheckResult** with state (valid/invalid/warn) and message.
- Check truth lives in the **registry checks table**, not embedded in artifacts.
- Checks can be attached to both artifacts AND runs (dual residence via foreign keys).
- Checks can be run independently of operators or executors.

## Registry

Persistent memory. DuckDB-backed.

- **Four tables**: artifacts, runs, checks, lineage.
- **save_run()** cascades: persists output artifact + checks + lineage edges in one call.
- **Lineage** stored as edges (parent_id → child_id) with operation and run_id. Graph-walkable.
- **get_full_lineage()** walks the full ancestor chain recursively.
- Artifacts loaded from registry carry their checks (loaded from checks table).

## Flow

Composition of operators into a multi-step chain with registry persistence.

- **HydrologyFlow** is the first implementation: fill → D8 → accumulation.
- Each step: build operator + params → submit to executor → persist RunRecord to registry → feed output to next step.
- Flow result carries all intermediate artifacts, runs, checks, and failure info.
- Registry persistence is optional — flows can run without a registry for testing.
- Generic DAG-based flow composition is not yet implemented — HydrologyFlow is purpose-built.

## Adapter (not yet implemented)

Exposure surface. How artifacts/operators become usable from QGIS, API, CLI, or agents.

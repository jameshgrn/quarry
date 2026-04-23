# Quarry Contracts

Canonical definitions. If the code and this doc disagree, the code wins — but update this doc.

## Artifact

The internal unit of truth. NOT a file.

- **Identity** is the `id` field (UUID). Not the path. Not the name.
- **BackingStore** describes where data physically lives (local_file, remote_uri, lazy_handle, memory, duckdb, postgis). This is NOT identity.
- **SpatialDescriptor** holds CRS, extent, resolution, feature_count, band_count. Populated from actual data, not guessed.
- **Lineage** records how the artifact was created (operation, input IDs, params, timestamp) and persists with the artifact itself.
- **Checks** accumulate from the checks table. Artifact exposes `validation_state` derived from accumulated check results.
- **Metadata** is an extensible dict for driver info, domain tags, platform info, etc.

Mutability rules:
- `id`, `type`: immutable
- `backing`: mostly immutable (set once at creation)
- `lineage`: immutable (set at creation)
- `checks`: immutable (evolved via `with_check()`)
- `metadata`: immutable (frozen at creation/load)

## Connector

The sacred gateway. No geospatial object enters except through a connector.

- **One required method**: `materialize(source_ref, workspace, lazy=False) -> MaterializeResult`
- **source_ref** is `SourceRef | str`. Connectors handle both.
- **Materialize does NOT always mean download.** It can mean: copy, wrap in place, stage, create a lazy handle, or normalize format.
- **MaterializeResult** carries: artifact + strategy + source_ref + notes. Strategy is provenance ("wrapped_local", "fetched_remote", "lazy_handle", "normalized").
- **Capabilities** are explicit flags: MATERIALIZE, DISCOVER, AUTHENTICATE, STREAM, MATERIALIZE_LAZY, METADATA_ONLY.
- **Optional protocols**: Discoverable, Authenticatable, MetadataEmitter.

`SourceRef` is the primary input type for the Connector protocol. Raw strings are still supported for backward compatibility and auto-inferred via `SourceRef.infer()`.

## SourceRef

Typed envelope for source references. Lives in quarry-core (zero deps).

- **raw**: the original string, always preserved, always round-trippable via `str(ref)`
- **kind**: classification tag (LOCAL_PATH, LOCAL_RASTER, LOCAL_VECTOR, REMOTE_URI, CATALOG_ITEM, DATABASE_REF, DUCKDB, UNKNOWN)
- **params**: optional parsed fields (connector-specific structure)

Factory methods:
- `SourceRef.local(path)` → LOCAL_RASTER, LOCAL_VECTOR, or LOCAL_PATH depending on known extension
- `SourceRef.uri(url)` → REMOTE_URI
- `SourceRef.stac(collection, item, asset=)` → CATALOG_ITEM
- `SourceRef.postgis(schema, table)` → DATABASE_REF
- `SourceRef.postgis_query(sql)` → DATABASE_REF
- `SourceRef.duckdb(db_path, table)` → DUCKDB
- `SourceRef.duckdb_query(db_path, sql)` → DUCKDB
- `SourceRef.infer(raw)` → best-effort classification from raw string

What SourceRef is NOT:
- Not a connector selector/router
- Not a validator (bad refs are still representable)
- Not a class hierarchy

SourceRef helps callers construct refs explicitly and helps routing layers classify them.
The Connector protocol receives it and can use its `kind` and `params` for structured dispatch.

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
- Local GeoJSON → LocalFile only
- Remote URI → COG only
- STAC catalog item → STAC only
- Database ref → PostGIS only
- DuckDB ref → DuckDB only
- Unknown → fallback connectors only (if any registered)

What ConnectorRouter is NOT:
- Not an executor (does not call materialize)
- Not a factory (does not create connector instances)
- Not a protocol change (Connector protocol unchanged)

## Operator

A typed transformation: artifacts in, artifact out.

- **Declares** accepted input types, output type, min/max inputs, resource scale.
- **validate_inputs()** returns list of error strings (empty = valid). Called before execute.
- **execute()** runs the transformation, returns `OperatorResult` (artifact + checks + warnings + timing + metadata).
- **declared_checks()** lists check names the operator runs on its output.
- Output artifact gets fresh spatial metadata from the actual output file — never copied from input.
- Output artifact gets its own identity, lineage, and backing store.
- **OperatorResult** fields (timing, warnings, metadata) persist through the registry.

## Executor

Dispatches operator execution to a compute backend.

- **submit()** takes operator + inputs + params, returns RunRecord.
- **status()** and **wait()** query/block on run completion.
- **RunRecord** is the full lifecycle object: pending → running → completed/failed, with timing, checks, error info.
- Validation or operator failure is represented as `RunStatus.FAILED` on the returned `RunRecord`, not as a fabricated side channel.
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
- **save_run()** cascades transactionally: persists the run plus output artifact + checks + lineage edges in one atomic call.
- **Lineage** stored as edges (parent_id → child_id) with operation and run_id. Graph-walkable.
- Artifact-level lineage payload also persists losslessly with the artifact row.
- **get_full_lineage()** walks the full ancestor chain recursively.
- Artifacts loaded from registry carry their checks and artifact-level lineage.

## Flow

Composition of operators into a multi-step chain with registry persistence.

- **HydrologyFlow** is the first implementation: fill → D8 → accumulation.
- Each step: build operator + params → submit to executor → persist RunRecord to registry → feed output to next step.
- Flow result carries all intermediate artifacts, runs, checks, and failure info.
- Registry persistence is optional — flows can run without a registry for testing.
- Generic DAG-based flow composition is not yet implemented — HydrologyFlow is purpose-built.

## Adapter

Exposure surface. How artifacts/operators become usable from CLI, QGIS, API, or agents.

- **Current adapter** is the CLI package (`quarry-cli`).
- Adapters do not bypass the substrate: they materialize through connectors, execute through executors, and inspect/persist through the registry.
- **Exit codes are adapter truth semantics**, not executor lifecycle:
  - `0` = completed with valid or warn-only checks
  - `1` = input, validation, or execution failure
  - `2` = completed execution but one or more checks are `INVALID`
- Adapters may expose inspection surfaces over artifacts, runs, lineage, and checks without changing the core contracts.

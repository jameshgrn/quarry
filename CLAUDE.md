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
  src/georuntime/      # Legacy prototype (DO NOT MODIFY)
```

## Canonical Commands

```sh
just test          # Run all pressure tests
just test-file F   # Run specific test file
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

- `source_ref: str` in Connector protocol — SourceRef type exists, protocol update deferred
- `OperatorSpec.output_type` singular — multi-output deferred until tile-splitting
- `Lineage` single object — may need append-only for multi-stage provenance
- Legacy `src/georuntime/` — migration deferred
- FillDepressions pure-Python loops — numba acceleration deferred until perf measured
- Flat gradient uses naive BFS — Barnes et al. (2015) optimal flat resolution deferred

## Substrate Phase Definition of Done

Substrate phase is complete when:
- Core ontology remains stable (no breaking contract changes)
- 3–5 connectors exist (currently: 4 — LocalFile, STAC, PostGIS, COG)
- 3–5 operators exist (currently: 4 — ClipRaster, Reproject, FillDepressions, D8FlowDirection)
- Registry persists artifacts/runs/checks/lineage (done)
- One end-to-end flow works across local + one remote source (done)
- No UI work beyond minimal debug CLI

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

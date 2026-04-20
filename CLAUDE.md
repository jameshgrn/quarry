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
    quarry-core/       # Contracts: Artifact, Connector, Operator, Executor, Check
    quarry-registry/   # DuckDB-backed artifact + run registry
    quarry-connectors/ # Connector implementations
    quarry-operators/  # Operator implementations
    quarry-cli/        # CLI surface
  src/georuntime/      # Legacy prototype (to be migrated into packages)
```

## Toolchain

- `uv` for package management
- `ruff check` + `ruff format`
- `ty check` for type checking
- `pytest -q` for tests
- No packages < 7 days old

## Commit Conventions

- Imperative mood, ≤72 char subject
- One logical change per commit
- Prefix with package name when relevant: `quarry-core: add Check protocol`

## Branch Strategy

- `main` — stable
- Feature branches off main
- No long-lived branches

## What NOT to build yet

- UI/viewer
- Workflow builder
- Chat interface
- Full serving stack
- Plugin marketplace
- MCP integration

These are surfaces. The substrate comes first.

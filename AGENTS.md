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

- `source_ref: str` — will likely become SourceRef type when PostGIS forces it
- `OperatorSpec.output_type` is singular — will need multi-output when tile-splitting appears
- `Lineage` on Artifact is a single object — may need append-only for multi-stage provenance
- Legacy `src/georuntime/` — migration deferred until packages are stable
- No Flow or Adapter implementations yet

## Immediate roadmap

1. PostGIS connector (adversarial: connection-oriented, schema/table refs, query-based)
2. If PostGIS stresses source_ref → introduce SourceRef type in quarry-core
3. Continue connector/operator expansion (3-5 of each for substrate phase)
4. Only then: richer flows, second executor, adapter surfaces

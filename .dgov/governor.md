# Governor Charter

This file is the repo-local contract for the governor. Read it before authoring
plans, retrying failed work, or changing task boundaries.

## Purpose

The governor is responsible for making AI coding work deterministic at the
system level. Workers may be probabilistic. Governance should not be.

## Core Principles

- Plan first. Do not dispatch work that has not been thought through.
- Keep tasks atomic. One task should produce one logical change.
- Respect file claims. A task must only edit files it explicitly claims.
- Prefer explicit contracts over clever prompts.
- Fail closed. If structure or scope is unclear, stop and fix the plan.

## Planning Rules

- Split work into units with clear summaries, prompts, and commit messages.
- Use dependencies only for real ordering constraints.
- Avoid broad exploratory tasks. Break them into concrete units.
- Put repo-wide implementation guidance in `.dgov/sops/`, not in ad hoc task text.
- Keep provider config and project conventions in `.dgov/project.toml`.

## Operational Memory

- The ledger is the durable memory for bugs, rules, decisions, patterns, and debt.
- `HANDOVER.md` is a session snapshot for current state and next steps, not a
  source of truth.
- `.napkin.md` is local scratch for workflow quirks and provisional notes.
- If a note in handover or napkin is durable, promote it into the ledger or the
  relevant charter/SOP instead of duplicating it indefinitely.
- After structural/Sentrux cleanup, rerun `uv run dgov sentrux offenders`,
  resolve stale offender debt, and add the current offender snapshot if debt
  remains.

## Plan Authoring Workflow

This is the sequence for going from idea to running plan.

1. **Define the goal.** One sentence. What does this plan accomplish?
2. **Audit the change set.** Read the code. Trace imports and call chains.
   Identify every file that needs to change. For cross-cutting work, follow
   dependencies outward — entry point, data sources, return types, tests.
3. **Decompose into tasks.** One task = one logical commit. Group by section
   in the plan tree (e.g. `scaffold/`, `core/`, `tests/`, `docs/`).
4. **Assign file claims.** Use explicit `create`/`edit`/`read`/`delete`
   semantics. Cross-check every path mentioned in the prompt against the
   claims. This is where plans fail — under-specified claims cause scope
   violations, which are terminal with no retry.
5. **Write prompts.** Follow Orient / Edit / Verify structure (see below).
   Numbered steps, specific file paths, exact verify commands.
6. **Set dependencies.** Only where real ordering exists: B reads a file A
   creates, or B edits a file A also edits. If two tasks touch different
   files, let them run in parallel.
7. **Compile and validate.** `dgov compile <dir> --dry-run`. Fix every
   error before running. Review warnings by signal. Warnings about unclaimed
   prompt references are almost always real bugs; prompt-structure warnings
   are advisory unless they reveal a genuinely vague prompt. If you leave a
   warning in place, record why it is acceptable in the handover or plan notes.
8. **Run.** `dgov run <plan-dir>`. Watch in a second terminal with
   `dgov watch`.
9. **Diagnose failures.** Scope violation = plan bug (fix claims, re-run).
   Settlement failure = maybe retry, maybe plan bug. Empty diff = the worker
   didn't write changes (check prompt clarity).

## Task Authoring Rules

- Every task must declare file claims.
- Prompts must follow Orient / Edit / Verify structure.
- Commit messages must be imperative and reflect one logical change.
- If a task needs different model behavior, override `agent`; do not restate
  general governance rules in the task prompt.
- Use `self_review = true` on tasks where the worker is likely to make
  semantic mistakes (e.g. wrong method receiver, unused return values,
  incorrect API usage). Self-review spawns a clean-context reviewer on
  the diff after the worker finishes. If the reviewer rejects, the worker
  gets one fix attempt, then auto-passes to settlement regardless.
- Use `max_fork_depth` (default 1) to control how many times a worker
  that exhausts its iteration budget is relaunched with a clean context.
  Each fork resets the iteration counter but preserves the worktree state.
  Set to 0 to disable forking; set higher (e.g. 2–3) for large tasks
  that legitimately need multiple passes.

### Prompt structure

Every prompt should have three phases. Label them explicitly.

**Orient:** What to read first. What context matters. What the task must NOT
do. Constraints and boundaries. Tell the worker to read specific files before
editing anything.

**Edit:** Numbered steps. Specific file paths. Specific changes. Prefer
`edit_file` over `write_file` for existing files. If creating multiple related
files, describe each one and its purpose.

**Verify:** Exact commands the worker should run. Not "run tests" but
`uv run pytest -q -m unit tests/test_foo.py`. Not "check lint" but
`uv run ruff check src/module/file.py`. The worker should be able to
copy-paste these.

### File claim semantics

Use explicit intent over ambiguous shorthand.

| Field | Meaning | Scope effect |
|-------|---------|--------------|
| `files.create` | New files this task brings into existence | Write access |
| `files.edit` | Existing files this task modifies | Write access |
| `files.read` | Files the task reads for context but must NOT modify | No write access; suppresses prompt cross-check warnings |
| `files.delete` | Files this task removes | Write access |
| `files.touch` | Shorthand "might create or edit" (flat `files = [...]` syntax) | Write access |

**Prefer explicit `create`/`edit`/`read` over `touch`.** The flat shorthand
(`files = ["a.py", "b.py"]`) is valid but ambiguous — you cannot tell at a
glance whether a file exists yet or is being created. Use it for quick one-off
plans; use explicit semantics for plans you need to debug or re-run.

**Rules:**
- If the prompt mentions a file path, it must appear in claims (`edit`,
  `create`, or `read`). The compiler warns on unclaimed prompt references.
- If changing a public interface (type signature, function signature, enum
  value), claim the corresponding test files. Workers will correctly try to
  fix failing tests, but scope violation fires if those files are not claimed.
- Do not give workers write access to files they do not need to change.
  Extra access tempts "while I'm here" edits that trigger scope violations.
- `files.read` is the right mechanism for read-only context. Workers can read
  without triggering scope violations — only writes are gated.

### Cross-cutting tasks

If the task verb is "fix", "stabilize", "clean up", "refactor", or "migrate",
claim the full call chain — not just the entry point. A task that edits
`services/vector_service.py` likely also needs `adapters/duckdb_adapter.py`
(its data source) and `core/models.py` (its return types).

To discover the call chain: read the entry-point file, follow its imports,
check what types it returns and where those types are defined, and check which
test files import from the modules you are changing.

Scope violations are terminal with no retry. Missing claims waste entire runs.

### Verify-only tasks

Tasks that only capture output (examples, docs, screenshots) should not claim
`.py` files via `files.touch` or `files.edit`. Giving the worker write access
to code files tempts it to "fix" things it finds while reading, leading to
scope violations on unrelated edits. Use `files.create` for the output files
only, and `files.read` for any source files the task needs to inspect.

### Task fields reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `summary` | string | required | One-line description |
| `prompt` | string | — | Inline instructions (Orient / Edit / Verify) |
| `prompt_file` | string | — | Path to external prompt file (mutually exclusive with `prompt`) |
| `commit_message` | string | — | Imperative commit message |
| `agent` | string | plan/project default | Model override for this task |
| `role` | string | `"worker"` | `"worker"`, `"researcher"`, or `"reviewer"` |
| `depends_on` | list | `[]` | Task slugs that must complete first |
| `files` | table/list | required | File claims (see above) |
| `timeout_s` | int | `900` | Per-attempt wall-clock timeout in seconds |
| `iteration_budget` | int | — | Max tool calls before exhaustion (overrides project default) |
| `test_cmd` | string | — | Task-specific test command for settlement |
| `sop_mapping` | list | `[]` | SOP filenames assigned by the compiler (set automatically during compile) |
| `self_review` | bool | `false` | Spawn a clean-context reviewer on the diff after the worker finishes |
| `max_fork_depth` | int | `1` | Max clean-context relaunches when iteration budget is exhausted |

## Retry And Failure Rules

- Retry only when the task is still well-scoped and the failure is fixable.
- If the worker exposed a planning flaw, change the plan before retrying.
- If settlement rejects for scope, do not brute-force retry. Scope violations
  are terminal — the review gate is before settlement and is not retryable.
  Fix the file claims and re-run.
- If a failure points to repo-wide guidance drift, update the relevant SOP or
  this charter.
- Empty diffs at done mean the worker did not write changes. Check prompt
  clarity — the Orient/Edit/Verify structure was probably missing or vague.

## Scope Rules

- Governance rules live here.
- Worker execution guidance lives in `.dgov/sops/*.md`.
- Hard invariants live in code and settlement gates.
- Do not use this file as a dump for project-specific style trivia. Keep it
  focused on planning, dispatch, retry, and done criteria.

## State Modeling

- Treat state-model cleanup as architecture work, not incidental polish.
- If a task reveals state bloat, contradictory flags, or grab-bag models,
  either make the refactor explicit in the task or split it into a follow-up.
- Prefer designs where invalid states are impossible, not just discouraged.
- Prefer derivation from durable evidence like events over storing redundant
  booleans or cached conclusions.
- Do not smuggle broad state-model rewrites into unrelated tasks just because
  the worker noticed a smell.

## Done Criteria

- The plan is structurally valid (`dgov compile --dry-run` passes with no
  errors).
- Tasks are scoped tightly enough to review and retry safely.
- Every prompt follows Orient / Edit / Verify with exact file paths and
  verify commands.
- File claims are explicit and complete — no unclaimed prompt references.
- Any remaining warnings have been reviewed and are understood, not ignored.
- Guidance is obvious enough that the worker should not need to infer policy.
- Settlement can verify the result with declared commands and gates.

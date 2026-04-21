# D8 Hydrology Pack v0

Canonical reference implementation for single-flow-direction hydrology in Quarry.

## The Chain

```
Raw DEM
  │
  ▼
FillDepressions          ← remove interior pits (Priority-Flood)
  │
  ▼
D8FlowDirection          ← steepest-descent direction codes
  │
  ▼
FlowAccumulation         ← upstream contributing area (topological sort)
```

Composed as `HydrologyFlow` (lane: flow), which runs all three through an Executor
and persists artifacts/runs/checks/lineage to a Registry.

## Operators

### FillDepressions

**Lane:** operator
**Algorithm:** Priority-Flood (Wang & Liu 2006), O(n log n)
**Input:** Single-band raster DEM (float64)
**Output:** Depression-filled DEM (float64)

**Assumes:**
- Input is materialized (not lazy)
- Nodata value either in params or source metadata
- Single-band raster

**Guarantees:**
- Elevation monotonically non-decreasing (no cell lowered)
- Zero interior pits remain on reachable cells
- Nodata cells preserved exactly (not filled, not moved)
- Output is fresh GeoTIFF with fresh metadata

**Params:**
- `nodata` — override source nodata value
- `apply_gradient` — enable micro-gradient on flat regions for D8 resolvability (default: true)
- `epsilon` — gradient increment per cell (default: 1e-5)

**Integrated checks:**
- `no_interior_pits` — count must be 0
- `elevation_only_raised` — no cell lowered
- `backing_accessible` — output file exists

### D8FlowDirection

**Lane:** operator
**Algorithm:** Steepest descent (O'Callaghan & Mark 1984) with flat resolution
**Input:** Single-band raster DEM (expects depression-filled, works on raw)
**Output:** Flow direction raster (int16)

**Direction encoding:**
```
5=NW  6=N  7=NE
4=W   ·    0=E
3=SW  2=S  1=SE

8=OUTLET  9=PIT  -1=NODATA
```

**Assumes:**
- Single-band raster
- Diagonal distance = sqrt(2) for slope calculation

**Guarantees:**
- Every valid cell assigned a code (0–7, 8, or 9)
- Boundary cells with no lower neighbor → OUTLET
- Interior cells with no lower neighbor → PIT (should be 0 after fill)
- Nodata cells → NODATA code (-1)
- Two-pass flat resolution: PITs on flat surfaces route to draining neighbors

**Integrated checks:**
- `valid_code_set` — all codes in {-1, 0–7, 8, 9}
- `no_pits` — PIT count should be 0 (WARN if not, not INVALID)
- `no_internal_outlets` — no valid cells flow into nodata
- `all_valid_assigned` — no valid cell has NODATA code
- `backing_accessible` — output file exists

### FlowAccumulation

**Lane:** operator
**Algorithm:** Topological sort (Kahn's algorithm), O(n)
**Input:** Single-band D8 flow direction raster (int16)
**Output:** Accumulation raster (float64)

**Assumes:**
- Input uses the D8 direction encoding above
- No cycles (hard failure if detected)

**Guarantees:**
- Each cell's value = sum of upstream accumulation (including self)
- Outlet cells collect all upstream flow
- PIT cells retain self-weight only (no downstream propagation)
- Conservation: sum of outlet+PIT accumulation = total valid cells × weight
- All valid cells >= weight

**Params:**
- `weight` — per-cell weight (default: 1.0, i.e., count upstream cells)

**Integrated checks:**
- `no_cycles` — always VALID (checked pre-execution, raises on failure)
- `nonnegative` — all valid cells >= weight
- `conservation` — outlet sum ≈ total weight (tolerance 1e-6)
- `backing_accessible` — output file exists

## Checks: Where They Live

Two kinds of validation exist in the hydrology pack:

### Operator-integrated checks

Run automatically during `execute()`. Declared via `declared_checks()`.
Returned in `OperatorResult.checks`. Persisted to registry via `save_run()`.

Every operator above declares its own checks. These are the primary validation path.

**Use when:** the check is tightly coupled to the operator's output and makes no sense
without the operator context (e.g., "no interior pits" only matters for fill output).

### Standalone reusable checks

Implement the `Check` protocol. Can run independently on any artifact.
Currently: `InternalOutletCount` (checks D8 flow direction artifacts).

**Use when:** the same validation applies across multiple operators or contexts,
or when you need to re-check an artifact after the fact (e.g., audit a registry artifact).

### When one wraps the other

D8FlowDirection's `no_internal_outlets` integrated check delegates to the same logic
as the standalone `InternalOutletCount`. The operator wraps the standalone check.
This is the right pattern when both forms are needed:
- Integrated: runs automatically in the chain, no extra code
- Standalone: available for ad-hoc validation, agents, audits

## Invariants Currently Enforced

1. **Elevation monotonicity** — fill never lowers a cell
2. **Zero interior pits** — on grid-connected valid cells after fill
3. **Valid direction assignment** — every valid cell gets a D8 code
4. **Conservation of flow** — total weight in = total weight at sinks
5. **Acyclicity** — flow network has no cycles (hard failure)
6. **Nodata preservation** — nodata cells pass through all operators unchanged

## Known Limitations

### Disconnected valid regions

Priority-Flood seeds from the grid boundary. Valid cells completely surrounded by
nodata (disconnected from any grid-edge cell) are never reached by the flood.
Interior pits on such islands remain unfilled. D8 assigns PIT codes to them.

**Impact:** Conservation still holds (PITs retain self-weight). The chain doesn't crash.
But the `no_interior_pits` check will flag the unfilled pit.

**Workaround:** None currently. A future enhancement could seed from nodata-adjacent
valid cells as secondary boundaries.

### Pure Python performance

All three operators use pure Python loops with numpy. Adequate for substrate proof
and DEMs up to ~1000×1000. Numba acceleration deferred until performance is measured.

### Flat resolution

Fill's flat gradient uses BFS from outlet edges — correct but naive.
D8's flat resolution iterates until convergence — worst case O(n) iterations on pathological flats.
Barnes et al. (2015) is the canonical reference for optimal flat resolution. Deferred.

## Deferred

- Rho8, D∞, MFD flow routing
- Per-cell weight rasters (e.g., rainfall-weighted accumulation)
- Watershed delineation (threshold + trace from accumulation)
- Numba/native acceleration
- Barnes et al. (2015) optimal flat resolution
- Distributed execution
- Multi-output operators (tile splitting)

## Test Coverage

| Suite | Tests | Focus |
|---|---|---|
| test_fill_depressions | 30 | Algorithm, protocol, edge cases |
| test_d8_flow_direction | 27 | Directions, diagonals, flat resolution, chain |
| test_flow_accumulation | 27 | Linear, branching, conservation, cycles, chain |
| test_hydrology_flow | 27 | End-to-end composition, registry, lineage, failure |
| test_internal_outlet_check | 15 | Standalone check, operator agreement |
| test_hydrology_adversarial | 27 | Pathological DEMs, nodata geometry, degenerate cases |

**Total: 153 hydrology-specific tests**

## Fixture Scenarios

### Hand-verifiable (tiny)
2×2, 3×3, 2×3, 1×5, 5×1

### Standard
Single pit, multi-cell depression, sloped (no pits), channel + depression,
nested depressions, cone, V-valley, flat surface

### Adversarial
L-shaped nodata, nodata island (disconnected), nodata at corners,
nodata cross bisection, checkerboard nodata, scattered nodata holes,
diagonal channel, sinuous channel, narrow valley, boundary plateau
with constrained spill, boundary shelf, saddle point, ridge line,
single valid cell in nodata sea, two adjacent valid cells

### Random / Monte Carlo
100×100 random (fill correctness), 50×50 random (fill→D8 chain),
30×30 random (full chain), 20×20 random (standalone/operator agreement)

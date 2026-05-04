# Quarry canonical commands

# Default: show available commands
default:
    @just --list

# Run a targeted pressure test file or subset
test TARGET:
    PYTHONPATH="packages/quarry-core/src:packages/quarry-connectors/src:packages/quarry-operators/src:packages/quarry-registry/src:packages/quarry-cli/src" uv run pytest {{TARGET}} -v

# Run the full pressure gate
test-all:
    PYTHONPATH="packages/quarry-core/src:packages/quarry-connectors/src:packages/quarry-operators/src:packages/quarry-registry/src:packages/quarry-cli/src" uv run pytest tests/pressure_test/ packages/quarry-connectors/tests/ -v

# Run a specific test file
test-file FILE:
    @just test {{FILE}}

# Run tests quietly (CI mode)
test-all-quiet:
    PYTHONPATH="packages/quarry-core/src:packages/quarry-connectors/src:packages/quarry-operators/src:packages/quarry-registry/src:packages/quarry-cli/src" uv run pytest tests/pressure_test/ packages/quarry-connectors/tests/ -q

# Lint all packages
lint:
    uv run ruff check packages/ tests/

# Format all packages
fmt:
    uv run ruff format packages/ tests/

# Type check (when ty is configured)
typecheck:
    uv run ty check packages/quarry-core/src/

# Lock dependencies
lock:
    uv lock

# Show package tree
tree:
    @echo "quarry-core (zero deps)"
    @echo "  ↑"
    @echo "quarry-connectors (+ rasterio, fiona, pystac-client, psycopg, shapely, requests)"
    @echo "quarry-operators  (+ rasterio, fiona, shapely)"
    @echo "quarry-registry   (+ duckdb)"
    @echo "quarry-cli        (quarry-core + connectors + operators + registry)"

# Show test counts
stats:
    @echo "Pressure tests:"
    @PYTHONPATH="packages/quarry-core/src:packages/quarry-connectors/src:packages/quarry-operators/src:packages/quarry-registry/src:packages/quarry-cli/src" uv run pytest tests/pressure_test/ packages/quarry-connectors/tests/ --collect-only -q 2>/dev/null | tail -1

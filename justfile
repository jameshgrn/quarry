# Quarry canonical commands

# Default: run all pressure tests
default: test

# Run all pressure tests
test:
    PYTHONPATH="packages/quarry-core/src:packages/quarry-connectors/src:packages/quarry-operators/src:packages/quarry-registry/src" uv run pytest tests/pressure_test/ -v

# Run a specific test file
test-file FILE:
    PYTHONPATH="packages/quarry-core/src:packages/quarry-connectors/src:packages/quarry-operators/src:packages/quarry-registry/src" uv run pytest {{FILE}} -v

# Run tests quietly (CI mode)
test-quiet:
    PYTHONPATH="packages/quarry-core/src:packages/quarry-connectors/src:packages/quarry-operators/src:packages/quarry-registry/src" uv run pytest tests/pressure_test/ -q

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

# Show test counts
stats:
    @echo "Pressure tests:"
    @PYTHONPATH="packages/quarry-core/src:packages/quarry-connectors/src:packages/quarry-operators/src:packages/quarry-registry/src" uv run pytest tests/pressure_test/ --collect-only -q 2>/dev/null | tail -1

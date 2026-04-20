# Quarry

Quarry is a personal geospatial runtime for humans and agents.

It is not a general GIS framework.
It is not a plugin platform.
It is not an abstraction project.

Current scope:
- one core operation: inspect
- one registry backend: DuckDB
- one interface: CLI

Design rules:
- prefer explicit code over reusable abstractions
- do not anticipate future operations
- keep outputs structured and machine-readable
- keep the workspace model simple

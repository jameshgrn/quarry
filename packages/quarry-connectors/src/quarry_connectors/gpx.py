"""GPXConnector — materializes GPX (GPS Exchange Format) files into canonical artifacts.

Lane: connector

GPX files contain tracks, routes, and waypoints. Fiona has a built-in GPX driver.
This connector handles multi-layer GPX with layer selection.

Source ref format:
- "path/to/file.gpx" → default layer (pick the one with most features; prefer "tracks" on tie)
- "path/to/file.gpx::waypoints" → specific layer
- "path/to/file.gpx::tracks" → specific layer

GPX layers in fiona: waypoints, routes, tracks, route_points, track_points

Design decisions:
- source_ref: local path with optional ::layer suffix
- Lazy = metadata-only with LAZY_HANDLE backing
- Eager local = wrap in place, LOCAL_FILE backing
- CRS is always WGS84 (EPSG:4326) for GPX — it's part of the GPX spec
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import fiona
from quarry_core.artifact import (
    Artifact,
    ArtifactType,
    BackingStore,
    BackingStoreKind,
    Lineage,
    SpatialDescriptor,
    content_hash,
)
from quarry_core.connector import (
    CatalogEntry,
    ConnectorCapability,
    MaterializeError,
    MaterializeResult,
)

if TYPE_CHECKING:
    from quarry_core.source_ref import SourceRef

# GPX layers in fiona (in order of preference for default selection)
GPX_LAYERS = ["waypoints", "routes", "tracks", "route_points", "track_points"]

# Default layer priority for tie-breaking (higher = preferred)
LAYER_PRIORITY = {"tracks": 3, "waypoints": 2, "routes": 1, "route_points": 0, "track_points": 0}


def _parse_gpx_source_ref(source_ref: SourceRef | str) -> tuple[str, str | None]:
    """Parse source_ref into (path, optional_layer).

    Returns:
        Tuple of (file_path, layer_name or None)
    """
    if hasattr(source_ref, "raw"):
        raw = source_ref.raw
    else:
        raw = str(source_ref)

    if "::" in raw:
        path_part, layer_part = raw.split("::", 1)
        return (path_part, layer_part if layer_part else None)
    return (raw, None)


def _is_gpx_file(path: Path) -> bool:
    """Check if file has .gpx extension (case-insensitive)."""
    return path.suffix.lower() == ".gpx"


def _get_layer_feature_counts(path: str) -> dict[str, int]:
    """Get feature counts for all layers in a GPX file.

    Args:
        path: Path to the GPX file

    Returns:
        Dict mapping layer name to feature count
    """
    counts: dict[str, int] = {}
    try:
        layers = fiona.listlayers(path)
    except Exception as e:
        raise MaterializeError(path, f"Failed to list GPX layers: {e}") from e

    for layer in layers:
        try:
            with fiona.open(path, layer=layer) as src:
                # GPX driver doesn't support len(), count by iterating
                counts[layer] = sum(1 for _ in src)
        except Exception:
            counts[layer] = 0

    return counts


def _select_default_layer(layer_counts: dict[str, int]) -> str | None:
    """Select the default layer based on feature counts and priority.

    Selection rules:
    1. Pick layer with most features
    2. On tie, prefer tracks > waypoints > routes > route_points/track_points

    Args:
        layer_counts: Dict mapping layer name to feature count

    Returns:
        Name of the default layer, or None if no layers have features
    """
    if not layer_counts:
        return None

    # Filter to layers with features
    populated = {name: count for name, count in layer_counts.items() if count > 0}

    if not populated:
        # Fall back to first available layer
        return next(iter(layer_counts.keys()), None)

    # Sort by count (descending), then by priority (descending)
    def sort_key(item: tuple[str, int]) -> tuple[int, int]:
        name, count = item
        priority = LAYER_PRIORITY.get(name, 0)
        return (-count, -priority)

    sorted_layers = sorted(populated.items(), key=sort_key)
    return sorted_layers[0][0]


def _read_gpx_layer_metadata(path: str, layer: str) -> dict[str, Any]:
    """Read metadata from a specific GPX layer.

    Args:
        path: Path to the GPX file
        layer: Layer name to read

    Returns:
        Dict with driver, crs, schema, bounds, feature_count, available_layers
    """
    try:
        with fiona.open(path, layer=layer) as src:
            # CRS is always WGS84 for GPX
            crs = "EPSG:4326"

            # Extract schema
            schema = src.schema

            # Feature count - GPX driver doesn't support len(), count by iterating
            count = sum(1 for _ in src)

            # Extract bounds (may fail on empty layers)
            try:
                bounds = src.bounds if count > 0 else None
            except Exception:
                bounds = None

            return {
                "driver": src.driver,
                "crs": crs,
                "schema": schema,
                "bounds": bounds,
                "feature_count": count,
            }
    except Exception as e:
        raise MaterializeError(f"{path}::{layer}", f"Failed to read GPX layer: {e}") from e


class GPXConnector:
    """Materializes GPX files into canonical Quarry artifacts.

    Supports multi-layer GPX files with automatic layer selection
    or explicit layer specification via ::layer suffix.
    """

    @property
    def name(self) -> str:
        return "gpx"

    @property
    def capabilities(self) -> ConnectorCapability:
        return (
            ConnectorCapability.MATERIALIZE
            | ConnectorCapability.DISCOVER
            | ConnectorCapability.MATERIALIZE_LAZY
            | ConnectorCapability.METADATA_ONLY
        )

    def materialize(
        self,
        source_ref: SourceRef | str,
        workspace: Path,
        *,
        lazy: bool = False,
    ) -> MaterializeResult:
        """Materialize a GPX file into a canonical artifact.

        source_ref: local path to .gpx file, optionally with ::layer suffix
        workspace: where to put materialized data if needed (unused for GPX)
        lazy: if True, return metadata-only artifact with LAZY_HANDLE backing
        """
        path_str, explicit_layer = _parse_gpx_source_ref(source_ref)
        path = Path(path_str)

        # Validate file exists and is a GPX file
        if not path.exists():
            raise MaterializeError(source_ref, f"File not found: {path}")

        if not path.is_file():
            raise MaterializeError(source_ref, f"Not a file: {path}")

        if not _is_gpx_file(path):
            raise MaterializeError(source_ref, f"Not a GPX file: {path}")

        # Get available layers and their feature counts
        try:
            layer_counts = _get_layer_feature_counts(str(path))
        except MaterializeError:
            raise
        except Exception as e:
            raise MaterializeError(source_ref, f"Failed to read GPX layers: {e}") from e

        if not layer_counts:
            raise MaterializeError(source_ref, "No layers found in GPX file")

        available_layers = list(layer_counts.keys())

        # Determine which layer to use
        if explicit_layer:
            if explicit_layer not in layer_counts:
                raise MaterializeError(
                    source_ref, f"Layer '{explicit_layer}' not found. Available: {available_layers}"
                )
            selected_layer = explicit_layer
        else:
            selected_layer = _select_default_layer(layer_counts)
            if selected_layer is None:
                raise MaterializeError(source_ref, "No suitable layer found in GPX file")

        # Read metadata for the selected layer
        try:
            meta = _read_gpx_layer_metadata(str(path), selected_layer)
        except MaterializeError:
            raise
        except Exception as e:
            raise MaterializeError(source_ref, f"Failed to read GPX metadata: {e}") from e

        # Build spatial descriptor
        bounds = meta["bounds"]
        spatial = SpatialDescriptor(
            crs=meta["crs"],
            extent=bounds if bounds else None,
            feature_count=meta["feature_count"],
        )

        # Build lineage params
        lineage_params: dict[str, Any] = {
            "source": "gpx",
            "path": str(path),
            "layer": selected_layer,
            "lazy": lazy,
            "available_layers": available_layers,
        }

        # Build artifact metadata
        artifact_meta = {
            "driver": meta["driver"],
            "schema": meta["schema"],
            "crs": meta["crs"],
            "feature_count": meta["feature_count"],
            "bounds": meta["bounds"],
            "selected_layer": selected_layer,
            "available_layers": available_layers,
            "layer_counts": layer_counts,
        }

        if lazy:
            # Lazy mode: metadata only, LAZY_HANDLE backing
            artifact = Artifact(
                type=ArtifactType.VECTOR,
                name=self._derive_name(path, selected_layer),
                backing=BackingStore(
                    kind=BackingStoreKind.LAZY_HANDLE,
                    uri=f"{path}::{selected_layer}",
                ),
                spatial=spatial,
                lineage=Lineage(operation="materialize", params=lineage_params),
                metadata=artifact_meta,
            )
            return MaterializeResult(
                artifact=artifact,
                strategy="lazy_handle",
                source_ref=source_ref,
                notes=(
                    f"GPX metadata only — layer '{selected_layer}' "
                    f"with {meta['feature_count']} features"
                ),
            )

        # Eager mode: wrap in place, LOCAL_FILE backing
        artifact = Artifact(
            type=ArtifactType.VECTOR,
            name=self._derive_name(path, selected_layer),
            backing=BackingStore(
                kind=BackingStoreKind.LOCAL_FILE,
                uri=str(path),
                size_bytes=path.stat().st_size,
                content_hash=content_hash(path),
            ),
            spatial=spatial,
            lineage=Lineage(operation="materialize", params=lineage_params),
            metadata=artifact_meta,
        )

        return MaterializeResult(
            artifact=artifact,
            strategy="wrapped_local",
            source_ref=source_ref,
            notes=(
                f"Local GPX wrapped — layer '{selected_layer}' "
                f"with {meta['feature_count']} features"
            ),
        )

    def discover(self, query: str | dict[str, Any] | None = None) -> list[CatalogEntry]:
        """Discover GPX data.

        Two modes:
        - If query points to a .gpx FILE → list available layers with feature counts
        - If query points to a DIRECTORY → list .gpx files

        query: file path or directory path as string, or dict with "path" key
        """
        if isinstance(query, dict):
            path_str = query.get("path")
        elif isinstance(query, str):
            path_str = query
        else:
            raise MaterializeError("discover", "No path specified")

        if not path_str:
            raise MaterializeError("discover", "No path specified")

        path = Path(path_str)

        if not path.exists():
            raise MaterializeError("discover", f"Path not found: {path}")

        # Mode 1: Query is a GPX file → list layers
        if path.is_file() and _is_gpx_file(path):
            try:
                layer_counts = _get_layer_feature_counts(str(path))
            except Exception as e:
                raise MaterializeError("discover", f"Failed to read GPX layers: {e}") from e

            entries = []
            for layer_name, count in layer_counts.items():
                entries.append(
                    CatalogEntry(
                        source_ref=f"{path}::{layer_name}",
                        name=layer_name,
                        description=f"GPX layer with {count} features",
                        metadata={
                            "feature_count": count,
                            "parent_file": str(path),
                        },
                    )
                )
            return entries

        # Mode 2: Query is a directory → list .gpx files
        if path.is_dir():
            seen: set[str] = set()
            entries = []
            for pattern in ("*.gpx", "*.GPX"):
                for file_path in path.glob(pattern):
                    resolved = str(file_path.resolve())
                    if resolved in seen:
                        continue
                    seen.add(resolved)

                    # Get layer counts for this file
                    try:
                        layer_counts = _get_layer_feature_counts(str(file_path))
                        total_features = sum(layer_counts.values())
                        available_layers = list(layer_counts.keys())
                    except Exception:
                        total_features = 0
                        available_layers = []

                    entries.append(
                        CatalogEntry(
                            source_ref=str(file_path),
                            name=file_path.stem,
                            description=(
                                f"GPX file with {total_features} features "
                                f"across {len(available_layers)} layers"
                            ),
                            spatial_hint={
                                "crs": "EPSG:4326",
                            },
                            metadata={
                                "extension": file_path.suffix,
                                "available_layers": available_layers,
                                "layer_counts": layer_counts,
                                "total_features": total_features,
                            },
                        )
                    )
            return entries

        raise MaterializeError("discover", f"Not a GPX file or directory: {path}")

    def metadata(self, source_ref: SourceRef | str) -> dict[str, Any]:
        """Get GPX metadata without materializing data.

        Returns per-layer metadata including feature counts and CRS.
        """
        path_str, explicit_layer = _parse_gpx_source_ref(source_ref)
        path = Path(path_str)

        if not path.exists():
            raise MaterializeError(source_ref, f"File not found: {path}")

        if not path.is_file():
            raise MaterializeError(source_ref, f"Not a file: {path}")

        if not _is_gpx_file(path):
            raise MaterializeError(source_ref, f"Not a GPX file: {path}")

        # Get all layer info
        try:
            layer_counts = _get_layer_feature_counts(str(path))
        except Exception as e:
            raise MaterializeError(source_ref, f"Failed to read GPX layers: {e}") from e

        available_layers = list(layer_counts.keys())

        # Build per-layer metadata
        layers_meta: dict[str, Any] = {}
        for layer_name in available_layers:
            try:
                layer_meta = _read_gpx_layer_metadata(str(path), layer_name)
                layers_meta[layer_name] = {
                    "feature_count": layer_meta["feature_count"],
                    "extent": layer_meta["bounds"],
                    "schema": layer_meta["schema"],
                }
            except Exception as e:
                layers_meta[layer_name] = {"error": str(e)}

        result: dict[str, Any] = {
            "driver": "GPX",
            "crs": "EPSG:4326",
            "available_layers": available_layers,
            "layer_counts": layer_counts,
            "layers": layers_meta,
        }

        # If a specific layer was requested, include its metadata at top level
        if explicit_layer:
            if explicit_layer in layers_meta:
                layer_info = layers_meta[explicit_layer]
                result["selected_layer"] = explicit_layer
                result["feature_count"] = layer_info.get("feature_count")
                result["extent"] = layer_info.get("extent")
                result["schema"] = layer_info.get("schema")
            else:
                result["error"] = f"Layer '{explicit_layer}' not found"

        return result

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _derive_name(self, path: Path, layer: str) -> str:
        """Derive artifact name from path and layer."""
        return f"{path.stem}_{layer}"

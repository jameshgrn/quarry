"""GeoPackageConnector — materializes GeoPackage (.gpkg) files into canonical artifacts.

Lane: connector

GeoPackage is an OGC standard format based on SQLite that can contain multiple
vector layers in a single file. This connector provides multi-layer discovery and
layer selection capabilities.

Key features:
- Multi-layer support: list and select layers within a GeoPackage
- Layer selection via :: separator: "file.gpkg::layer_name"
- Default layer: first layer if no :: specified
- Local file materialization (wrap in place or lazy handle)
- Uses fiona (GDAL) for all I/O — GeoPackage driver is built into GDAL

Design decisions:
- source_ref: local path with optional ::layer_name suffix
- Lazy = metadata-only with LAZY_HANDLE backing
- Eager local = wrap in place, LOCAL_FILE backing
- discover() has TWO modes:
  - Query points to .gpkg file → list layers within it
  - Query points to directory → list .gpkg files in it
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


def _is_gpkg_file(path: Path) -> bool:
    """Check if file has .gpkg extension (case-insensitive)."""
    return path.suffix.lower() == ".gpkg"


def _parse_gpkg_source_ref(source_ref: SourceRef | str) -> tuple[str, str | None]:
    """Parse source_ref into (path, optional_layer_name).

    Supports:
    - "path/to/file.gpkg" → (path, None) - use default (first) layer
    - "path/to/file.gpkg::layer_name" → (path, layer_name) - specific layer

    Returns:
        Tuple of (file_path, layer_name or None)
    """
    if isinstance(source_ref, str):
        raw = source_ref.strip()
    else:
        raw = source_ref.raw.strip()

    # Check for :: separator
    if "::" in raw:
        path_part, layer_name = raw.split("::", 1)
        return (path_part, layer_name)

    return (raw, None)


def _read_gpkg_metadata(path: str, layer_name: str | None = None) -> dict[str, Any]:
    """Read metadata from a GeoPackage file using fiona.

    Args:
        path: Local path to the .gpkg file
        layer_name: Optional layer name to read (default: first layer)

    Returns:
        Dict with driver, crs, schema, bounds, feature_count, all_layers
    """
    # List all layers first
    all_layers = fiona.listlayers(path)

    if not all_layers:
        raise MaterializeError(path, "GeoPackage contains no layers")

    # Determine which layer to read
    target_layer = layer_name if layer_name is not None else all_layers[0]

    if target_layer not in all_layers:
        raise MaterializeError(path, f"Layer '{target_layer}' not found. Available: {all_layers}")

    with fiona.open(path, layer=target_layer) as src:
        # Extract CRS
        crs = None
        if src.crs:
            if isinstance(src.crs, dict):
                if "init" in src.crs:
                    crs = src.crs["init"]
                else:
                    crs = str(src.crs)
            else:
                crs_str = str(src.crs)
                if crs_str:
                    crs = crs_str

        # Extract schema
        schema = src.schema

        # Feature count
        count = len(src)

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
            "all_layers": all_layers,
            "layer_name": target_layer,
        }


class GeoPackageConnector:
    """Materializes GeoPackage files into canonical Quarry artifacts.

    Supports multi-layer GeoPackages with layer selection via :: separator.
    """

    @property
    def name(self) -> str:
        return "geopackage"

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
        """Materialize a GeoPackage file into a canonical artifact.

        source_ref: local path to .gpkg file, optionally with ::layer_name suffix
        workspace: where to materialize if needed (unused for local files)
        lazy: if True, return metadata-only artifact with LAZY_HANDLE backing
        """
        path_str, layer_name = _parse_gpkg_source_ref(source_ref)
        path = Path(path_str)

        # Validate file exists
        if not path.exists():
            raise MaterializeError(source_ref, f"File not found: {path}")

        if not path.is_file():
            raise MaterializeError(source_ref, f"Not a file: {path}")

        if not _is_gpkg_file(path):
            raise MaterializeError(source_ref, f"Not a .gpkg file: {path}")

        # Read metadata
        try:
            meta = _read_gpkg_metadata(str(path), layer_name)
        except MaterializeError:
            raise
        except Exception as e:
            raise MaterializeError(source_ref, f"Failed to read GeoPackage metadata: {e}") from e

        # Build spatial descriptor
        bounds = meta["bounds"]
        spatial = SpatialDescriptor(
            crs=meta["crs"],
            extent=bounds if bounds else None,
            feature_count=meta["feature_count"],
        )

        # Build lineage params
        lineage_params: dict[str, Any] = {
            "source": "geopackage",
            "path": str(path),
            "layer": meta["layer_name"],
            "all_layers": meta["all_layers"],
            "lazy": lazy,
        }

        # Build artifact metadata
        artifact_meta = {
            "driver": meta["driver"],
            "schema": meta["schema"],
            "crs": meta["crs"],
            "feature_count": meta["feature_count"],
            "bounds": meta["bounds"],
            "layer_name": meta["layer_name"],
            "all_layers": meta["all_layers"],
        }

        if lazy:
            # Lazy mode: metadata only, LAZY_HANDLE backing
            artifact = Artifact(
                type=ArtifactType.VECTOR,
                name=self._derive_name(path, meta["layer_name"]),
                backing=BackingStore(
                    kind=BackingStoreKind.LAZY_HANDLE,
                    uri=f"{path}::{meta['layer_name']}",
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
                    f"GeoPackage metadata only — layer '{meta['layer_name']}' "
                    f"with {meta['feature_count']} features"
                ),
            )

        # Eager mode: wrap local file in place
        local_path = path.resolve()
        strategy = "wrapped_local"
        notes = (
            f"Local GeoPackage wrapped — layer '{meta['layer_name']}' "
            f"({local_path.stat().st_size} bytes)"
        )

        artifact = Artifact(
            type=ArtifactType.VECTOR,
            name=self._derive_name(path, meta["layer_name"]),
            backing=BackingStore(
                kind=BackingStoreKind.LOCAL_FILE,
                uri=str(local_path),
                size_bytes=local_path.stat().st_size,
                content_hash=content_hash(local_path),
            ),
            spatial=spatial,
            lineage=Lineage(operation="materialize", params=lineage_params),
            metadata=artifact_meta,
        )

        return MaterializeResult(
            artifact=artifact,
            strategy=strategy,
            source_ref=source_ref,
            notes=notes,
        )

    def discover(self, query: str | dict[str, Any] | None = None) -> list[CatalogEntry]:
        """Discover GeoPackage content.

        TWO modes:
        - If query points to a .gpkg FILE → list layers within it
        - If query points to a DIRECTORY → list .gpkg files in it

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
            raise MaterializeError("discover", f"Path not found: {path_str}")

        # Mode 1: Query is a .gpkg file → list layers within it
        if path.is_file() and _is_gpkg_file(path):
            try:
                layers = fiona.listlayers(str(path))
            except Exception as e:
                raise MaterializeError("discover", f"Failed to list layers: {e}") from e

            entries = []
            for layer in layers:
                entries.append(
                    CatalogEntry(
                        source_ref=f"{path}::{layer}",
                        name=layer,
                        description=f"Layer '{layer}' in {path.name}",
                        metadata={
                            "file": str(path),
                            "layer": layer,
                        },
                    )
                )
            return entries

        # Mode 2: Query is a directory → list .gpkg files in it
        if path.is_dir():
            seen: set[str] = set()
            entries = []
            for pattern in ("*.gpkg", "*.GPKG"):
                for file_path in path.glob(pattern):
                    resolved = str(file_path.resolve())
                    if resolved in seen:
                        continue
                    seen.add(resolved)
                    entries.append(
                        CatalogEntry(
                            source_ref=str(file_path),
                            name=file_path.stem,
                            metadata={
                                "extension": file_path.suffix,
                            },
                        )
                    )
            return entries

        # Neither file nor directory (shouldn't happen given exists() check)
        raise MaterializeError("discover", f"Not a file or directory: {path_str}")

    def metadata(self, source_ref: SourceRef | str) -> dict[str, Any]:
        """Get GeoPackage metadata without materializing data."""
        path_str, layer_name = _parse_gpkg_source_ref(source_ref)
        path = Path(path_str)

        if not path.exists():
            raise MaterializeError(source_ref, f"File not found: {path}")

        if not path.is_file():
            raise MaterializeError(source_ref, f"Not a file: {path}")

        if not _is_gpkg_file(path):
            raise MaterializeError(source_ref, f"Not a .gpkg file: {path}")

        try:
            meta = _read_gpkg_metadata(str(path), layer_name)
        except MaterializeError:
            raise
        except Exception as e:
            raise MaterializeError(source_ref, f"Failed to read GeoPackage metadata: {e}") from e

        return {
            "driver": meta["driver"],
            "crs": meta["crs"],
            "schema": meta["schema"],
            "feature_count": meta["feature_count"],
            "extent": meta["bounds"],
            "layer_name": meta["layer_name"],
            "all_layers": meta["all_layers"],
        }

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _derive_name(self, path: Path, layer_name: str) -> str:
        """Derive artifact name from path and layer name."""
        return f"{path.stem}_{layer_name}"

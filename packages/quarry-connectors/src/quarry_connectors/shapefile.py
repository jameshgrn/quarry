"""ShapefileConnector — materializes Shapefile (.shp) files into canonical artifacts.

Lane: connector

Shapefiles are multi-file formats (.shp + .shx + .dbf + optional .prj, .cpg).
This connector's unique value is sidecar validation, encoding detection, and
multi-file awareness.

Key features:
- Sidecar validation: ensures .shx and .dbf exist (required), warns on missing .prj
- Encoding detection: reads encoding from .cpg file if present
- Multi-file awareness: reports sidecar inventory in metadata
- Uses fiona (GDAL) for all I/O — Shapefile driver is built into GDAL

Sidecar files:
- .shx: index file (required)
- .dbf: attribute table (required)
- .prj: projection/CRS (optional, warns if missing)
- .cpg: encoding specification (optional)

Design decisions:
- source_ref: path to .shp file (sidecars detected automatically from same directory)
- Lazy = metadata-only with LAZY_HANDLE backing
- Eager local = wrap in place, LOCAL_FILE backing
- discover() finds .shp files and reports sidecar completeness per file
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


# Sidecar file extensions and their roles
SIDECAR_REQUIRED = {".shx", ".dbf"}
SIDECAR_OPTIONAL = {".prj", ".cpg"}
ALL_SIDECARS = SIDECAR_REQUIRED | SIDECAR_OPTIONAL


def _is_shp_file(path: Path) -> bool:
    """Check if file has .shp extension (case-insensitive)."""
    return path.suffix.lower() == ".shp"


def _get_sidecar_path(shp_path: Path, ext: str) -> Path:
    """Get the path to a sidecar file with given extension.

    Preserves the case of the original .shp file stem.
    """
    return shp_path.with_suffix(ext)


def _detect_sidecars(shp_path: Path) -> dict[str, bool]:
    """Detect which sidecar files exist for a shapefile.

    Returns:
        Dict mapping sidecar extension (without dot) to existence boolean.
        Keys: shx, dbf, prj, cpg
    """
    return {
        "shx": _get_sidecar_path(shp_path, ".shx").exists(),
        "dbf": _get_sidecar_path(shp_path, ".dbf").exists(),
        "prj": _get_sidecar_path(shp_path, ".prj").exists(),
        "cpg": _get_sidecar_path(shp_path, ".cpg").exists(),
    }


def _read_cpg_encoding(shp_path: Path) -> str | None:
    """Read encoding from .cpg file if it exists.

    Returns:
        Encoding string (e.g., "UTF-8") or None if .cpg missing/unreadable.
    """
    cpg_path = _get_sidecar_path(shp_path, ".cpg")
    if not cpg_path.exists():
        return None

    try:
        # .cpg files are single-line text files with encoding name
        content = cpg_path.read_text(encoding="utf-8").strip()
        return content if content else None
    except Exception:
        return None


def _validate_sidecars(shp_path: Path, sidecars: dict[str, bool]) -> None:
    """Validate that required sidecars exist.

    Raises:
        MaterializeError: If any required sidecar (.shx, .dbf) is missing.
    """
    missing = []
    for ext in SIDECAR_REQUIRED:
        key = ext.lstrip(".")
        if not sidecars.get(key, False):
            missing.append(ext)

    if missing:
        missing_str = ", ".join(sorted(missing))
        raise MaterializeError(
            str(shp_path),
            f"Missing required sidecar file(s): {missing_str}. "
            f"Shapefile requires .shp + .shx + .dbf",
        )


def _read_shp_metadata(shp_path: Path) -> dict[str, Any]:
    """Read metadata from a Shapefile using fiona.

    Args:
        shp_path: Path to the .shp file

    Returns:
        Dict with driver, crs, schema, bounds, feature_count, sidecars, encoding

    Raises:
        MaterializeError: If metadata cannot be read.
    """
    try:
        with fiona.open(str(shp_path)) as src:
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
            }
    except MaterializeError:
        raise
    except Exception as e:
        raise MaterializeError(shp_path, f"Failed to read Shapefile metadata: {e}") from e


class ShapefileConnector:
    """Materializes Shapefile (.shp) files into canonical Quarry artifacts.

    Provides sidecar validation, encoding detection, and multi-file awareness.
    """

    @property
    def name(self) -> str:
        return "shapefile"

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
        """Materialize a Shapefile into a canonical artifact.

        Args:
            source_ref: Path to .shp file (sidecars detected automatically)
            workspace: Where to materialize if needed (unused for local files)
            lazy: If True, return metadata-only artifact with LAZY_HANDLE backing

        Returns:
            MaterializeResult with the artifact and provenance.

        Raises:
            MaterializeError: If materialization fails or required sidecars missing.
        """
        # Parse source_ref to get path
        path_str = self._parse_source_ref(source_ref)
        shp_path = Path(path_str)

        # Validate .shp file exists
        if not shp_path.exists():
            raise MaterializeError(source_ref, f"Shapefile not found: {shp_path}")

        if not shp_path.is_file():
            raise MaterializeError(source_ref, f"Not a file: {shp_path}")

        if not _is_shp_file(shp_path):
            raise MaterializeError(source_ref, f"Not a .shp file: {shp_path}")

        # Detect sidecars
        sidecars = _detect_sidecars(shp_path)

        # Validate required sidecars exist
        _validate_sidecars(shp_path, sidecars)

        # Read encoding from .cpg if present
        encoding = _read_cpg_encoding(shp_path)

        # Read metadata using fiona
        try:
            meta = _read_shp_metadata(shp_path)
        except MaterializeError:
            raise
        except Exception as e:
            raise MaterializeError(source_ref, f"Failed to read Shapefile metadata: {e}") from e

        # Handle missing .prj warning
        missing_prj = not sidecars["prj"]
        if missing_prj:
            # CRS from fiona will likely be None, but we preserve that
            crs = None
        else:
            crs = meta["crs"]

        # Build spatial descriptor
        bounds = meta["bounds"]
        spatial = SpatialDescriptor(
            crs=crs,
            extent=bounds if bounds else None,
            feature_count=meta["feature_count"],
        )

        # Build lineage params
        lineage_params: dict[str, Any] = {
            "source": "shapefile",
            "path": str(shp_path),
            "lazy": lazy,
            "sidecars": sidecars,
            "encoding": encoding,
            "missing_prj": missing_prj,
        }

        # Build artifact metadata
        artifact_meta: dict[str, Any] = {
            "driver": meta["driver"],
            "schema": meta["schema"],
            "crs": crs,
            "feature_count": meta["feature_count"],
            "bounds": meta["bounds"],
            "sidecars": sidecars,
            "encoding": encoding,
        }

        if missing_prj:
            artifact_meta["missing_prj"] = True

        if lazy:
            # Lazy mode: metadata only, LAZY_HANDLE backing
            artifact = Artifact(
                type=ArtifactType.VECTOR,
                name=self._derive_name(shp_path),
                backing=BackingStore(
                    kind=BackingStoreKind.LAZY_HANDLE,
                    uri=str(shp_path),
                ),
                spatial=spatial,
                lineage=Lineage(operation="materialize", params=lineage_params),
                metadata=artifact_meta,
            )
            return MaterializeResult(
                artifact=artifact,
                strategy="lazy_handle",
                source_ref=source_ref,
                notes=f"Shapefile metadata only — {meta['feature_count']} features",
            )

        # Eager mode: wrap local file in place
        local_path = shp_path.resolve()
        strategy = "wrapped_local"
        notes = f"Local Shapefile wrapped ({local_path.stat().st_size} bytes)"

        artifact = Artifact(
            type=ArtifactType.VECTOR,
            name=self._derive_name(shp_path),
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
        """Find .shp files in a directory and report sidecar completeness per file.

        Args:
            query: Directory path as string or dict with "path" key

        Returns:
            List of CatalogEntry with sidecar info in metadata.

        Raises:
            MaterializeError: If no path specified or path not found.
        """
        if isinstance(query, dict):
            dir_path = query.get("path")
        elif isinstance(query, str):
            dir_path = query
        else:
            raise MaterializeError("discover", "No path specified")

        if not dir_path:
            raise MaterializeError("discover", "No path specified")

        path = Path(dir_path)
        if not path.exists():
            raise MaterializeError("discover", f"Path not found: {dir_path}")

        if not path.is_dir():
            raise MaterializeError("discover", f"Not a directory: {dir_path}")

        seen: set[str] = set()
        entries = []
        for pattern in ("*.shp", "*.SHP"):
            for file_path in path.glob(pattern):
                resolved = str(file_path.resolve())
                if resolved in seen:
                    continue
                seen.add(resolved)

                # Detect sidecars for this shapefile
                sidecars = _detect_sidecars(file_path)

                # Check completeness
                has_all_required = sidecars["shx"] and sidecars["dbf"]
                has_prj = sidecars["prj"]

                entries.append(
                    CatalogEntry(
                        source_ref=str(file_path),
                        name=file_path.stem,
                        metadata={
                            "extension": file_path.suffix,
                            "sidecars": sidecars,
                            "complete": has_all_required,
                            "has_prj": has_prj,
                        },
                    )
                )

        return entries

    def metadata(self, source_ref: SourceRef | str) -> dict[str, Any]:
        """Get Shapefile metadata without materializing data.

        Args:
            source_ref: Path to .shp file

        Returns:
            Metadata dict with schema, CRS, extent, sidecars, encoding.

        Raises:
            MaterializeError: If file not found or metadata cannot be read.
        """
        path_str = self._parse_source_ref(source_ref)
        shp_path = Path(path_str)

        # Validate .shp file exists
        if not shp_path.exists():
            raise MaterializeError(source_ref, f"Shapefile not found: {shp_path}")

        if not shp_path.is_file():
            raise MaterializeError(source_ref, f"Not a file: {shp_path}")

        if not _is_shp_file(shp_path):
            raise MaterializeError(source_ref, f"Not a .shp file: {shp_path}")

        # Detect sidecars
        sidecars = _detect_sidecars(shp_path)

        # Validate required sidecars
        _validate_sidecars(shp_path, sidecars)

        # Read encoding from .cpg if present
        encoding = _read_cpg_encoding(shp_path)

        # Read metadata
        try:
            meta = _read_shp_metadata(shp_path)
        except MaterializeError:
            raise
        except Exception as e:
            raise MaterializeError(source_ref, f"Failed to read Shapefile metadata: {e}") from e

        result: dict[str, Any] = {
            "driver": meta["driver"],
            "crs": meta["crs"],
            "schema": meta["schema"],
            "feature_count": meta["feature_count"],
            "extent": meta["bounds"],
            "sidecars": sidecars,
            "encoding": encoding,
        }

        if not sidecars["prj"]:
            result["missing_prj"] = True

        return result

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _parse_source_ref(self, source_ref: SourceRef | str) -> str:
        """Parse source_ref into a path string.

        Returns:
            Path string to the .shp file.
        """
        if isinstance(source_ref, str):
            return source_ref.strip()
        return source_ref.raw.strip()

    def _derive_name(self, shp_path: Path) -> str:
        """Derive artifact name from shapefile path."""
        return shp_path.stem

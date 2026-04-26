"""KMZConnector — materializes KMZ (zipped KML) files into canonical artifacts.

Lane: connector

KMZ is a zipped KML file. KML is readable by fiona, but KMZ needs extraction first.
This connector handles the zip extraction and then delegates to fiona's KML/LIBKML driver.

Source ref format:
- Path to .kmz file

Key implementation details:
- Uses stdlib zipfile to extract
- KMZ spec: the KML file is typically named "doc.kml" at the root of the zip
- Fallback: if no "doc.kml", look for any .kml file in the zip
- If no .kml found in zip, raise MaterializeError
- Extract to workspace / "{stem}_extracted/" directory
- Use fiona.open(extracted_kml_path) to read — fiona has KML driver
- CRS is always WGS84 (EPSG:4326) for KML/KMZ — it's part of the spec
- ArtifactType.VECTOR always
- Store original_kmz_path and extracted_kml_path in lineage params
- Strategy: "normalized" (since we're converting from KMZ to KML)
"""

from __future__ import annotations

import zipfile
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


def _is_kmz_file(path: Path) -> bool:
    """Check if file has .kmz extension (case-insensitive)."""
    return path.suffix.lower() == ".kmz"


def _find_kml_in_zip(zip_path: Path) -> str | None:
    """Find the KML file inside a KMZ zip archive.

    KMZ spec: the KML file is typically named "doc.kml" at the root.
    Fallback: look for any .kml file in the zip.

    Args:
        zip_path: Path to the KMZ file

    Returns:
        Name of the KML file inside the zip, or None if not found
    """
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            namelist = zf.namelist()

            # First, look for doc.kml at root (KMZ spec)
            for name in namelist:
                if name.lower() == "doc.kml":
                    return name

            # Fallback: look for any .kml file
            for name in namelist:
                if name.lower().endswith(".kml"):
                    return name

            return None
    except zipfile.BadZipFile:
        return None
    except Exception:
        return None


def _get_kml_driver() -> str:
    """Get the available KML driver name from fiona.

    Returns:
        Driver name ("LIBKML" or "KML")

    Raises:
        MaterializeError: If no KML driver is available
    """
    if "LIBKML" in fiona.supported_drivers:
        return "LIBKML"
    elif "KML" in fiona.supported_drivers:
        return "KML"
    else:
        raise RuntimeError("No KML driver available in fiona/GDAL")


def _extract_kmz(
    kmz_path: Path,
    workspace: Path,
) -> Path:
    """Extract KML from KMZ to workspace.

    Args:
        kmz_path: Path to the KMZ file
        workspace: Directory to extract to

    Returns:
        Path to the extracted KML file

    Raises:
        MaterializeError: If extraction fails or no KML found
    """
    # Validate KMZ file exists
    if not kmz_path.exists():
        raise MaterializeError(str(kmz_path), f"KMZ file not found: {kmz_path}")

    if not kmz_path.is_file():
        raise MaterializeError(str(kmz_path), f"Not a file: {kmz_path}")

    # Check if it's a valid zip file
    try:
        with zipfile.ZipFile(kmz_path, "r") as zf:
            _ = zf.namelist()
    except zipfile.BadZipFile as e:
        raise MaterializeError(str(kmz_path), f"Not a valid zip file: {e}") from e
    except Exception as e:
        raise MaterializeError(str(kmz_path), f"Failed to open KMZ: {e}") from e

    # Find KML file in zip
    kml_name = _find_kml_in_zip(kmz_path)
    if kml_name is None:
        raise MaterializeError(str(kmz_path), "No .kml file found in KMZ archive")

    # Create extraction directory: workspace / "{stem}_extracted/"
    extract_dir = workspace / f"{kmz_path.stem}_extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)

    # Extract the KML file
    extracted_kml_path = extract_dir / Path(kml_name).name

    try:
        with zipfile.ZipFile(kmz_path, "r") as zf:
            with zf.open(kml_name) as src:
                extracted_kml_path.write_bytes(src.read())
    except Exception as e:
        raise MaterializeError(str(kmz_path), f"Failed to extract KML from KMZ: {e}") from e

    return extracted_kml_path


def _read_kml_metadata(kml_path: Path) -> dict[str, Any]:
    """Read metadata from a KML file using fiona.

    Args:
        kml_path: Path to the extracted KML file

    Returns:
        Dict with driver, crs, schema, bounds, feature_count

    Raises:
        MaterializeError: If reading metadata fails
    """
    try:
        driver = _get_kml_driver()
    except RuntimeError as e:
        raise MaterializeError(str(kml_path), str(e)) from e

    try:
        with fiona.open(str(kml_path), driver=driver) as src:
            # CRS is always WGS84 for KML/KMZ per spec
            crs = "EPSG:4326"

            # Extract schema
            schema = src.schema

            # Extract bounds
            bounds = src.bounds

            # Feature count
            count = len(src)

            return {
                "driver": src.driver,
                "crs": crs,
                "schema": schema,
                "bounds": bounds,
                "feature_count": count,
            }
    except fiona.errors.DriverError as e:
        raise MaterializeError(
            str(kml_path), f"KML driver error (driver may not be available): {e}"
        ) from e
    except Exception as e:
        raise MaterializeError(str(kml_path), f"Failed to read KML metadata: {e}") from e


class KMZConnector:
    """Materializes KMZ files into canonical Quarry artifacts.

    Extracts KML from KMZ and delegates to fiona's KML/LIBKML driver.
    CRS is always WGS84 (EPSG:4326) per KML spec.
    """

    @property
    def name(self) -> str:
        return "kmz"

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
        """Materialize a KMZ file into a canonical artifact.

        source_ref: local path to .kmz file
        workspace: where to extract KML file
        lazy: if True, return metadata-only artifact with LAZY_HANDLE backing
        """
        path_str = str(source_ref) if not hasattr(source_ref, "raw") else source_ref.raw
        kmz_path = Path(path_str)

        # Validate file exists and is a KMZ file
        if not kmz_path.exists():
            raise MaterializeError(source_ref, f"File not found: {kmz_path}")

        if not kmz_path.is_file():
            raise MaterializeError(source_ref, f"Not a file: {kmz_path}")

        if not _is_kmz_file(kmz_path):
            raise MaterializeError(source_ref, f"Not a KMZ file: {kmz_path}")

        # Extract KML from KMZ
        try:
            extracted_kml_path = _extract_kmz(kmz_path, workspace)
        except MaterializeError:
            raise
        except Exception as e:
            raise MaterializeError(source_ref, f"Failed to extract KMZ: {e}") from e

        # Read metadata from extracted KML
        try:
            meta = _read_kml_metadata(extracted_kml_path)
        except MaterializeError:
            raise
        except Exception as e:
            raise MaterializeError(source_ref, f"Failed to read KML metadata: {e}") from e

        # Build spatial descriptor
        bounds = meta["bounds"]
        spatial = SpatialDescriptor(
            crs=meta["crs"],
            extent=bounds if bounds else None,
            feature_count=meta["feature_count"],
        )

        # Build lineage params
        lineage_params: dict[str, Any] = {
            "source": "kmz",
            "original_kmz_path": str(kmz_path.resolve()),
            "extracted_kml_path": str(extracted_kml_path.resolve()),
            "lazy": lazy,
        }

        # Build artifact metadata
        artifact_meta = {
            "driver": meta["driver"],
            "schema": meta["schema"],
            "crs": meta["crs"],
            "feature_count": meta["feature_count"],
            "bounds": meta["bounds"],
            "original_kmz": str(kmz_path),
        }

        if lazy:
            # Lazy mode: metadata only, LAZY_HANDLE backing
            artifact = Artifact(
                type=ArtifactType.VECTOR,
                name=self._derive_name(kmz_path),
                backing=BackingStore(
                    kind=BackingStoreKind.LAZY_HANDLE,
                    uri=str(extracted_kml_path),
                ),
                spatial=spatial,
                lineage=Lineage(operation="materialize", params=lineage_params),
                metadata=artifact_meta,
            )
            return MaterializeResult(
                artifact=artifact,
                strategy="lazy_handle",
                source_ref=source_ref,
                notes=f"KMZ metadata only — {meta['feature_count']} features, KML extracted",
            )

        # Eager mode: LOCAL_FILE backing pointing to extracted KML
        artifact = Artifact(
            type=ArtifactType.VECTOR,
            name=self._derive_name(kmz_path),
            backing=BackingStore(
                kind=BackingStoreKind.LOCAL_FILE,
                uri=str(extracted_kml_path),
                size_bytes=extracted_kml_path.stat().st_size,
                content_hash=content_hash(extracted_kml_path),
            ),
            spatial=spatial,
            lineage=Lineage(operation="materialize", params=lineage_params),
            metadata=artifact_meta,
        )

        return MaterializeResult(
            artifact=artifact,
            strategy="normalized",
            source_ref=source_ref,
            notes=f"KMZ normalized to KML — {meta['feature_count']} features",
        )

    def discover(self, query: str | dict[str, Any] | None = None) -> list[CatalogEntry]:
        """List .kmz files in a directory.

        query: directory path as string or dict with "path" key
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
        if not path.is_dir():
            raise MaterializeError("discover", f"Not a directory: {dir_path}")

        seen: set[str] = set()
        entries = []
        for pattern in ("*.kmz", "*.KMZ"):
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

    def metadata(self, source_ref: SourceRef | str) -> dict[str, Any]:
        """Get KMZ metadata without fully materializing.

        Extracts KML temporarily to read metadata, but doesn't persist
        the extracted file (just reads and returns metadata).
        """
        path_str = str(source_ref) if not hasattr(source_ref, "raw") else source_ref.raw
        kmz_path = Path(path_str)

        if not kmz_path.exists():
            raise MaterializeError(source_ref, f"File not found: {kmz_path}")

        if not kmz_path.is_file():
            raise MaterializeError(source_ref, f"Not a file: {kmz_path}")

        if not _is_kmz_file(kmz_path):
            raise MaterializeError(source_ref, f"Not a KMZ file: {kmz_path}")

        # Find KML in zip without extracting
        kml_name = _find_kml_in_zip(kmz_path)
        if kml_name is None:
            raise MaterializeError(source_ref, "No .kml file found in KMZ archive")

        # Create a temporary workspace for extraction
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_workspace = Path(tmp_dir)

            # Extract KML temporarily
            try:
                extracted_kml_path = _extract_kmz(kmz_path, temp_workspace)
            except MaterializeError:
                raise
            except Exception as e:
                raise MaterializeError(source_ref, f"Failed to extract KMZ: {e}") from e

            # Read metadata
            try:
                meta = _read_kml_metadata(extracted_kml_path)
            except MaterializeError:
                raise
            except Exception as e:
                raise MaterializeError(source_ref, f"Failed to read KML metadata: {e}") from e

        return {
            "driver": meta["driver"],
            "crs": meta["crs"],
            "schema": meta["schema"],
            "feature_count": meta["feature_count"],
            "extent": meta["bounds"],
            "kml_file_in_archive": kml_name,
            "original_kmz": str(kmz_path),
        }

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _derive_name(self, path: Path) -> str:
        """Derive artifact name from KMZ path."""
        return path.stem

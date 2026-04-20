"""LocalFileConnector — materializes local geospatial files into artifacts.

The simplest connector. Takes a local path, inspects it, wraps it as an artifact.
Does NOT copy the file — wraps it in place (strategy: "wrapped_local").

For rasters: uses rasterio to extract spatial metadata.
For vectors: uses fiona to extract spatial metadata.
"""

from __future__ import annotations

from pathlib import Path

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

RASTER_EXTENSIONS = {".tif", ".tiff", ".geotiff", ".jp2", ".hgt", ".nc", ".vrt"}
VECTOR_EXTENSIONS = {".shp", ".geojson", ".gpkg", ".kml", ".gml", ".fgb", ".parquet"}


class LocalFileConnector:
    """Materializes local geospatial files into canonical artifacts."""

    @property
    def name(self) -> str:
        return "local_file"

    @property
    def capabilities(self) -> ConnectorCapability:
        return (
            ConnectorCapability.MATERIALIZE
            | ConnectorCapability.DISCOVER
            | ConnectorCapability.METADATA_ONLY
        )

    def materialize(
        self,
        source_ref: str,
        workspace: Path,
        *,
        lazy: bool = False,
    ) -> MaterializeResult:
        """Wrap a local file as a canonical artifact.

        Does not copy. The file stays where it is.
        Inspects it to extract spatial metadata.
        """
        path = Path(source_ref).resolve()

        if not path.exists():
            raise MaterializeError(source_ref, f"File not found: {path}")

        ext = path.suffix.lower()
        if ext in RASTER_EXTENSIONS:
            artifact = self._materialize_raster(path, lazy)
        elif ext in VECTOR_EXTENSIONS:
            artifact = self._materialize_vector(path, lazy)
        else:
            raise MaterializeError(source_ref, f"Unsupported extension: {ext}")

        strategy = "lazy_handle" if lazy else "wrapped_local"
        return MaterializeResult(
            artifact=artifact,
            strategy=strategy,
            source_ref=source_ref,
        )

    def discover(self, query: str | dict | None = None) -> list[CatalogEntry]:
        """List geospatial files in a directory.

        Args:
            query: Path to a directory to scan (str), or dict with 'path' and optional 'recursive'.
        """
        if query is None:
            query = "."

        if isinstance(query, str):
            search_dir = Path(query).resolve()
            recursive = False
        else:
            search_dir = Path(query.get("path", ".")).resolve()
            recursive = query.get("recursive", False)

        if not search_dir.is_dir():
            return []

        all_extensions = RASTER_EXTENSIONS | VECTOR_EXTENSIONS
        entries = []

        pattern = "**/*" if recursive else "*"
        for p in search_dir.glob(pattern):
            if p.suffix.lower() in all_extensions:
                entries.append(
                    CatalogEntry(
                        source_ref=str(p),
                        name=p.stem,
                        metadata={"extension": p.suffix, "size_bytes": p.stat().st_size},
                    )
                )

        return entries

    def metadata(self, source_ref: str) -> dict:
        """Get metadata without full materialization."""
        path = Path(source_ref).resolve()
        if not path.exists():
            raise MaterializeError(source_ref, f"File not found: {path}")

        ext = path.suffix.lower()
        if ext in RASTER_EXTENSIONS:
            return self._raster_metadata(path)
        elif ext in VECTOR_EXTENSIONS:
            return self._vector_metadata(path)
        else:
            raise MaterializeError(source_ref, f"Unsupported extension: {ext}")

    # -----------------------------------------------------------------------
    # Private
    # -----------------------------------------------------------------------

    def _materialize_raster(self, path: Path, lazy: bool) -> Artifact:
        """Inspect raster and produce artifact."""
        if lazy:
            return Artifact(
                type=ArtifactType.RASTER,
                name=path.stem,
                backing=BackingStore(
                    kind=BackingStoreKind.LAZY_HANDLE,
                    uri=str(path),
                ),
                lineage=Lineage(operation="materialize", params={"lazy": True}),
            )

        import rasterio

        with rasterio.open(path) as src:
            bounds = src.bounds
            artifact = Artifact(
                type=ArtifactType.RASTER,
                name=path.stem,
                backing=BackingStore(
                    kind=BackingStoreKind.LOCAL_FILE,
                    uri=str(path),
                    size_bytes=path.stat().st_size,
                    content_hash=content_hash(path),
                ),
                spatial=SpatialDescriptor(
                    crs=str(src.crs) if src.crs else None,
                    extent=(bounds.left, bounds.bottom, bounds.right, bounds.top),
                    resolution=(src.res[0], src.res[1]),
                    band_count=src.count,
                ),
                lineage=Lineage(operation="materialize"),
                metadata={"driver": src.driver, "dtype": str(src.dtypes[0])},
            )

        return artifact

    def _materialize_vector(self, path: Path, lazy: bool) -> Artifact:
        """Inspect vector and produce artifact."""
        if lazy:
            return Artifact(
                type=ArtifactType.VECTOR,
                name=path.stem,
                backing=BackingStore(
                    kind=BackingStoreKind.LAZY_HANDLE,
                    uri=str(path),
                ),
                lineage=Lineage(operation="materialize", params={"lazy": True}),
            )

        import fiona

        with fiona.open(path) as src:
            bounds = src.bounds
            artifact = Artifact(
                type=ArtifactType.VECTOR,
                name=path.stem,
                backing=BackingStore(
                    kind=BackingStoreKind.LOCAL_FILE,
                    uri=str(path),
                    size_bytes=path.stat().st_size,
                    content_hash=content_hash(path),
                ),
                spatial=SpatialDescriptor(
                    crs=str(src.crs) if src.crs else None,
                    extent=(bounds[0], bounds[1], bounds[2], bounds[3]),
                    feature_count=len(src),
                ),
                lineage=Lineage(operation="materialize"),
                metadata={"driver": src.driver, "schema": dict(src.schema)},
            )

        return artifact

    def _raster_metadata(self, path: Path) -> dict:
        import rasterio

        with rasterio.open(path) as src:
            return {
                "crs": str(src.crs) if src.crs else None,
                "extent": (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top),
                "resolution": src.res,
                "band_count": src.count,
                "driver": src.driver,
                "shape": (src.height, src.width),
                "dtype": str(src.dtypes[0]),
            }

    def _vector_metadata(self, path: Path) -> dict:
        import fiona

        with fiona.open(path) as src:
            return {
                "crs": str(src.crs) if src.crs else None,
                "extent": src.bounds,
                "feature_count": len(src),
                "driver": src.driver,
                "schema": dict(src.schema),
            }

"""FOFStackConnector — materializes fof-compiler stack.nc files as artifacts.

Lane: connector

Reads fof-compiler output NetCDF files (HDF5 under the hood) with gridded
bands at 30m EPSG:4326. Key band: water_frequency (float32, 0-1).

Uses h5py directly — no xarray dependency.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
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


class FOFStackConnector:
    """Materializes fof-compiler stack.nc files as raster artifacts."""

    @property
    def name(self) -> str:
        return "fof_stack"

    @property
    def capabilities(self) -> ConnectorCapability:
        return (
            ConnectorCapability.MATERIALIZE
            | ConnectorCapability.DISCOVER
            | ConnectorCapability.METADATA_ONLY
            | ConnectorCapability.MATERIALIZE_LAZY
        )

    def materialize(
        self,
        source_ref: SourceRef | str,
        workspace: Path,
        *,
        lazy: bool = False,
        bands: list[str] | None = None,
    ) -> MaterializeResult:
        path = Path(source_ref).resolve()

        if path.suffix.lower() not in {".nc", ".nc4", ".h5", ".hdf5"}:
            raise MaterializeError(source_ref, f"Not a NetCDF/HDF5 file: {path.suffix}")

        if not path.exists():
            raise MaterializeError(source_ref, f"File not found: {path}")

        meta = self._read_metadata(path)

        if lazy:
            artifact = self._materialize_lazy(path, meta)
            strategy = "lazy_handle"
        else:
            artifact = self._materialize_eager(path, workspace, meta, bands)
            strategy = "normalized"

        return MaterializeResult(artifact=artifact, strategy=strategy, source_ref=source_ref)

    def discover(self, query: str | dict | None = None) -> list[CatalogEntry]:
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

        pattern = "**/*.nc" if recursive else "*.nc"
        entries = []
        for p in search_dir.glob(pattern):
            try:
                meta = self._read_metadata(p)
                entries.append(
                    CatalogEntry(
                        source_ref=str(p),
                        name=p.stem,
                        spatial_hint={
                            "crs": meta.get("crs"),
                            "extent": meta.get("extent"),
                        },
                        metadata={
                            "bands": meta.get("bands", []),
                            "size_bytes": p.stat().st_size,
                        },
                    )
                )
            except Exception:
                pass

        return entries

    def metadata(self, source_ref: SourceRef | str) -> dict[str, Any]:
        path = Path(source_ref).resolve()
        if not path.exists():
            raise MaterializeError(source_ref, f"File not found: {path}")
        return self._read_metadata(path)

    # -----------------------------------------------------------------------
    # Private
    # -----------------------------------------------------------------------

    def _read_metadata(self, path: Path) -> dict[str, Any]:
        import h5py

        with h5py.File(str(path), "r") as f:
            # Data bands = top-level datasets that are 2D (not coordinate arrays)
            coord_names = {"x", "y", "lat", "lon", "latitude", "longitude", "time"}
            data_bands = [
                k
                for k in f.keys()
                if isinstance(f[k], h5py.Dataset) and f[k].ndim >= 2 and k not in coord_names
            ]

            # Get spatial extent from x/y or lon/lat coordinate arrays
            crs = None
            extent = None
            resolution = None

            if "crs" in f.attrs:
                crs = str(f.attrs["crs"])

            x_ds = f.get("x") or f.get("lon") or f.get("longitude")
            y_ds = f.get("y") or f.get("lat") or f.get("latitude")

            if x_ds is not None and y_ds is not None:
                x_vals = x_ds[:]
                y_vals = y_ds[:]
                crs = crs or "EPSG:4326"
                extent = (
                    float(x_vals.min()),
                    float(y_vals.min()),
                    float(x_vals.max()),
                    float(y_vals.max()),
                )
                if len(x_vals) > 1:
                    resolution = (
                        float(abs(x_vals[1] - x_vals[0])),
                        float(abs(y_vals[1] - y_vals[0])),
                    )

            shape = None
            if data_bands:
                shape = list(f[data_bands[0]].shape)

        return {
            "bands": data_bands,
            "crs": crs or "EPSG:4326",
            "extent": extent,
            "resolution": resolution,
            "shape": shape,
        }

    def _materialize_lazy(self, path: Path, meta: dict[str, Any]) -> Artifact:
        return Artifact(
            type=ArtifactType.RASTER,
            name=path.stem,
            backing=BackingStore(kind=BackingStoreKind.LAZY_HANDLE, uri=str(path)),
            spatial=SpatialDescriptor(
                crs=meta.get("crs"),
                extent=meta.get("extent"),
                resolution=meta.get("resolution"),
                band_count=len(meta.get("bands", [])),
            ),
            lineage=Lineage(operation="materialize", params={"lazy": True}),
            metadata={"source": "fof_stack", "bands": meta.get("bands", [])},
        )

    def _materialize_eager(
        self,
        path: Path,
        workspace: Path,
        meta: dict[str, Any],
        bands: list[str] | None,
    ) -> Artifact:
        import h5py
        import rasterio
        from rasterio.transform import from_bounds

        with h5py.File(str(path), "r") as f:
            available = meta["bands"]
            extract = bands if bands else available
            extract = [b for b in extract if b in available]
            if not extract:
                raise MaterializeError(path, f"No valid bands. Available: {available}")

            # Read band data
            band_arrays = []
            for band_name in extract:
                arr = f[band_name][:]
                if arr.ndim == 3:
                    arr = arr[0]  # Take first slice of 3D
                band_arrays.append(arr.astype(np.float32))

        height, width = band_arrays[0].shape
        extent = meta.get("extent")
        if extent is None:
            extent = (0, 0, width, height)

        xmin, ymin, xmax, ymax = extent
        transform = from_bounds(xmin, ymin, xmax, ymax, width, height)

        out_dir = workspace / path.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / "stack.tif"

        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": len(extract),
            "dtype": "float32",
            "crs": meta.get("crs", "EPSG:4326"),
            "transform": transform,
            "compress": "deflate",
            "nodata": np.nan,
        }

        with rasterio.open(str(output_path), "w", **profile) as dst:
            for i, arr in enumerate(band_arrays, 1):
                dst.write(arr, i)
                dst.set_band_description(i, extract[i - 1])

        return Artifact(
            type=ArtifactType.RASTER,
            name=path.stem,
            backing=BackingStore(
                kind=BackingStoreKind.LOCAL_FILE,
                uri=str(output_path),
                size_bytes=output_path.stat().st_size,
                content_hash=content_hash(output_path),
            ),
            spatial=SpatialDescriptor(
                crs=meta.get("crs", "EPSG:4326"),
                extent=extent,
                resolution=meta.get("resolution"),
                band_count=len(extract),
            ),
            lineage=Lineage(
                operation="fof_stack_materialize",
                params={"source": str(path), "bands": extract},
            ),
            metadata={"source": "fof_stack", "bands": extract},
        )

"""PIXCConnector — materializes SWOT L2 HR PIXC (Pixel Cloud) data.

Lane: connector

Reads SWOT L2 HR PIXC NetCDF/HDF5 files and materializes them as raster artifacts.
PIXC data is sparse point cloud data (lat, lon, height, sig0, classification, water_frac)
that must be rasterized to a regular grid for analysis.

Key datasets in pixel_cloud/ group:
- latitude (float64): pixel latitude
- longitude (float64): pixel longitude
- height (float32): water surface elevation (EGM2008 geoid)
- sig0 (float32): radar backscatter
- classification (uint8): pixel classification (1-7)
- water_frac (float32): water fraction (0-1)
- azimuth_index (int32): azimuth grid index
- range_index (int32): range grid index

Classification values:
- 1=Land, 2=Land Near Water, 3=Water Near Land, 4=Open Water,
- 5=Dark Water, 6=Low Coh Water Edge, 7=Open Low Coh Water

Reference: JPL D-56410 SWOT Product Description L2 HR PIXC
"""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class PIXCMetadata:
    """SWOT PIXC file metadata extracted from HDF5 attributes."""

    tile_name: str
    cycle: int
    pass_number: int
    swath_side: str  # "L" or "R"
    time_start: str
    time_end: str
    num_pixels: int
    lat_bounds: tuple[float, float]  # (min, max)
    lon_bounds: tuple[float, float]  # (min, max)


class PIXCConnector:
    """Materializes SWOT L2 HR PIXC data from NetCDF/HDF5 files.

    The PIXC connector reads sparse pixel cloud data and rasterizes it to
    a regular grid at configurable resolution (default 30m). Output is a
    multi-band GeoTIFF with bands: sig0, height, water_frac, classification.
    """

    @property
    def name(self) -> str:
        return "pixc"

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
        resolution_m: float = 30.0,
        target_grid: tuple[float, float, float, float, float] | None = None,
    ) -> MaterializeResult:
        """Materialize a PIXC file as a raster artifact.

        Args:
            source_ref: Path to the PIXC NetCDF file.
            workspace: Where to write the rasterized output.
            lazy: If True, only extract metadata without reading pixel data.
            resolution_m: Target resolution in meters (ignored if target_grid set).
            target_grid: (west, south, east, north, resolution_deg) — rasterize
                directly onto this grid. Use to align with another raster (e.g. FOF stack).

        Returns:
            MaterializeResult with the artifact and provenance.
        """
        path = Path(source_ref).resolve()

        # Check extension first
        if path.suffix.lower() not in {".nc", ".h5", ".hdf5"}:
            raise MaterializeError(source_ref, f"Not a NetCDF/HDF5 file: {path.suffix}")

        if not path.exists():
            raise MaterializeError(source_ref, f"File not found: {path}")

        metadata = self._read_metadata(path)

        if lazy:
            artifact = self._materialize_lazy(path, metadata)
            strategy = "lazy_handle"
        else:
            artifact = self._materialize_rasterized(
                path, workspace, metadata, resolution_m, target_grid
            )
            strategy = "normalized"

        return MaterializeResult(
            artifact=artifact,
            strategy=strategy,
            source_ref=source_ref,
        )

    def discover(self, query: str | dict | None = None) -> list[CatalogEntry]:
        """List PIXC files in a directory.

        Args:
            query: Path to a directory to scan (str), or dict with 'path'
                   and optional 'recursive'.
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

        extensions = {".nc", ".h5", ".hdf5"}
        entries = []

        pattern = "**/*" if recursive else "*"
        for p in search_dir.glob(pattern):
            if p.suffix.lower() in extensions:
                try:
                    meta = self._read_metadata(p)
                    entries.append(
                        CatalogEntry(
                            source_ref=str(p),
                            name=meta.tile_name or p.stem,
                            spatial_hint={
                                "crs": "EPSG:4326",
                                "extent": (
                                    meta.lon_bounds[0],
                                    meta.lat_bounds[0],
                                    meta.lon_bounds[1],
                                    meta.lat_bounds[1],
                                ),
                            },
                            metadata={
                                "cycle": meta.cycle,
                                "pass": meta.pass_number,
                                "swath": meta.swath_side,
                                "pixels": meta.num_pixels,
                                "time_start": meta.time_start,
                                "time_end": meta.time_end,
                                "size_bytes": p.stat().st_size,
                            },
                        )
                    )
                except Exception:
                    # Skip files that can't be read as PIXC
                    pass

        return entries

    def metadata(self, source_ref: SourceRef | str) -> dict[str, Any]:
        """Get metadata without full materialization."""
        path = Path(source_ref).resolve()
        if not path.exists():
            raise MaterializeError(source_ref, f"File not found: {path}")

        meta = self._read_metadata(path)
        return {
            "tile_name": meta.tile_name,
            "cycle": meta.cycle,
            "pass": meta.pass_number,
            "swath_side": meta.swath_side,
            "time_start": meta.time_start,
            "time_end": meta.time_end,
            "num_pixels": meta.num_pixels,
            "lat_bounds": meta.lat_bounds,
            "lon_bounds": meta.lon_bounds,
            "crs": "EPSG:4326",
        }

    # -----------------------------------------------------------------------
    # Private
    # -----------------------------------------------------------------------

    def _read_metadata(self, path: Path) -> PIXCMetadata:
        """Read PIXC metadata from HDF5 file."""
        import h5py

        with h5py.File(str(path), "r") as ds:
            attrs = dict(ds.attrs)

            # Get pixel_cloud group for shape info
            if "pixel_cloud" not in ds:
                raise MaterializeError(path, "No pixel_cloud group found")

            pc = ds["pixel_cloud"]
            num_pixels = pc["latitude"].shape[0] if "latitude" in pc else 0

            return PIXCMetadata(
                tile_name=self._decode(attrs.get("tile_name", b"")),
                cycle=int(attrs.get("cycle_number", [0])[0]),
                pass_number=int(attrs.get("pass_number", [0])[0]),
                swath_side=self._decode(attrs.get("swath_side", b"")),
                time_start=self._decode(attrs.get("time_coverage_start", b"")),
                time_end=self._decode(attrs.get("time_coverage_end", b"")),
                num_pixels=num_pixels,
                lat_bounds=(
                    float(attrs.get("geospatial_lat_min", [0])[0]),
                    float(attrs.get("geospatial_lat_max", [0])[0]),
                ),
                lon_bounds=(
                    float(attrs.get("geospatial_lon_min", [0])[0]),
                    float(attrs.get("geospatial_lon_max", [0])[0]),
                ),
            )

    def _materialize_lazy(self, path: Path, metadata: PIXCMetadata) -> Artifact:
        """Create a lazy-handle artifact (metadata only)."""
        return Artifact(
            type=ArtifactType.RASTER,
            name=metadata.tile_name or path.stem,
            backing=BackingStore(
                kind=BackingStoreKind.LAZY_HANDLE,
                uri=str(path),
            ),
            spatial=SpatialDescriptor(
                crs="EPSG:4326",
                extent=(
                    metadata.lon_bounds[0],
                    metadata.lat_bounds[0],
                    metadata.lon_bounds[1],
                    metadata.lat_bounds[1],
                ),
                resolution=None,  # Unknown until rasterized
                band_count=4,  # sig0, height, water_frac, classification
            ),
            lineage=Lineage(operation="materialize", params={"lazy": True}),
            metadata={
                "source": "pixc",
                "cycle": metadata.cycle,
                "pass": metadata.pass_number,
                "swath": metadata.swath_side,
                "pixels": metadata.num_pixels,
            },
        )

    def _materialize_rasterized(
        self,
        path: Path,
        workspace: Path,
        metadata: PIXCMetadata,
        resolution_m: float,
        target_grid: tuple[float, float, float, float, float] | None = None,
    ) -> Artifact:
        """Rasterize PIXC pixel cloud to a regular grid and materialize as artifact."""
        import rasterio
        from rasterio.transform import from_bounds

        # Read PIXC data
        data = self._read_pixc_data(path)

        # Use target grid if provided, otherwise derive from PIXC extent
        if target_grid is not None:
            lon_min, lat_min, lon_max, lat_max, deg_resolution = target_grid
        else:
            lon_min, lon_max = metadata.lon_bounds
            lat_min, lat_max = metadata.lat_bounds
            deg_resolution = resolution_m / 111320.0

        width = max(1, int(np.ceil((lon_max - lon_min) / deg_resolution)))
        height = max(1, int(np.ceil((lat_max - lat_min) / deg_resolution)))

        # Rasterize to grid
        raster = self._rasterize_points(
            data["longitude"],
            data["latitude"],
            data["sig0"],
            data["height"],
            data["water_frac"],
            data["classification"],
            lon_min,
            lat_min,
            deg_resolution,
            width,
            height,
        )

        # Create output path
        out_dir = workspace / (metadata.tile_name or path.stem)
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / "pixc_rasterized.tif"

        # Write GeoTIFF
        transform = from_bounds(lon_min, lat_min, lon_max, lat_max, width, height)

        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 4,  # sig0, height, water_frac, classification
            "dtype": "float32",
            "crs": "EPSG:4326",
            "transform": transform,
            "compress": "deflate",
            "nodata": np.nan,
        }

        with rasterio.open(str(output_path), "w", **profile) as dst:
            dst.write(raster["sig0"], 1)
            dst.write(raster["height"], 2)
            dst.write(raster["water_frac"], 3)
            dst.write(raster["classification"], 4)

            # Set band descriptions
            dst.set_band_description(1, "sig0")
            dst.set_band_description(2, "height")
            dst.set_band_description(3, "water_frac")
            dst.set_band_description(4, "classification")

        return Artifact(
            type=ArtifactType.RASTER,
            name=metadata.tile_name or path.stem,
            backing=BackingStore(
                kind=BackingStoreKind.LOCAL_FILE,
                uri=str(output_path),
                size_bytes=output_path.stat().st_size,
                content_hash=content_hash(output_path),
            ),
            spatial=SpatialDescriptor(
                crs="EPSG:4326",
                extent=(lon_min, lat_min, lon_max, lat_max),
                resolution=(deg_resolution, deg_resolution),
                band_count=4,
            ),
            lineage=Lineage(
                operation="pixc_rasterize",
                params={
                    "source": str(path),
                    "resolution_m": resolution_m,
                    "pixels": metadata.num_pixels,
                },
            ),
            metadata={
                "source": "pixc",
                "format": "geotiff",
                "cycle": metadata.cycle,
                "pass": metadata.pass_number,
                "swath": metadata.swath_side,
                "bands": ["sig0", "height", "water_frac", "classification"],
            },
        )

    def _read_pixc_data(self, path: Path) -> dict[str, np.ndarray]:
        """Read pixel cloud data from PIXC file."""
        import h5py

        with h5py.File(str(path), "r") as ds:
            if "pixel_cloud" not in ds:
                raise MaterializeError(path, "No pixel_cloud group found")

            pc = ds["pixel_cloud"]

            # Read required datasets
            required = ["latitude", "longitude", "height", "sig0", "classification", "water_frac"]
            data = {}
            for key in required:
                if key not in pc:
                    raise MaterializeError(path, f"Missing required dataset: {key}")
                data[key] = pc[key][:]

            return data

    def _rasterize_points(
        self,
        longitude: np.ndarray,
        latitude: np.ndarray,
        sig0: np.ndarray,
        height: np.ndarray,
        water_frac: np.ndarray,
        classification: np.ndarray,
        lon_min: float,
        lat_min: float,
        resolution: float,
        width: int,
        height_dim: int,
    ) -> dict[str, np.ndarray]:
        """Rasterize sparse point cloud to regular grid using nearest-neighbor.

        Uses scatter-to-grid assignment with numpy advanced indexing.
        """
        # Calculate grid indices (north-up: row 0 = north)
        col = ((longitude - lon_min) / resolution).astype(np.int32)
        lat_max = lat_min + height_dim * resolution
        row = ((lat_max - latitude) / resolution).astype(np.int32)

        # Clip to grid bounds
        col = np.clip(col, 0, width - 1)
        row = np.clip(row, 0, height_dim - 1)

        # Initialize output arrays with NaN
        sig0_grid = np.full((height_dim, width), np.nan, dtype=np.float32)
        height_grid = np.full((height_dim, width), np.nan, dtype=np.float32)
        water_frac_grid = np.full((height_dim, width), np.nan, dtype=np.float32)
        classification_grid = np.full((height_dim, width), np.nan, dtype=np.float32)

        # Scatter values to grid (nearest-neighbor: last point wins at each cell)
        # For multiple points in same cell, we take the last one (simple approach)
        sig0_grid[row, col] = sig0
        height_grid[row, col] = height
        water_frac_grid[row, col] = water_frac
        classification_grid[row, col] = classification.astype(np.float32)

        return {
            "sig0": sig0_grid,
            "height": height_grid,
            "water_frac": water_frac_grid,
            "classification": classification_grid,
        }

    def _decode(self, val) -> str:
        """Decode bytes to string."""
        if isinstance(val, bytes):
            return val.decode()
        if isinstance(val, np.ndarray):
            return str(val.item()) if val.size == 1 else str(val)
        return str(val)

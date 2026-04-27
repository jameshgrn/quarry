"""SLCConnector — materializes SWOT L1B HR SLC (Single Look Complex) data.

Lane: connector

Reads HDF5 SLC files and materializes them as artifacts. Can optionally
process to calibrated sigma0 and interferometric products.

Key SWOT SLC formula:
    sigma0 = (|SLC|^2 - noise) / xfactor

Reference: JPL D-56410 SWOT Product Description L1B HR SLC
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
class SLCMetadata:
    """SWOT SLC file metadata extracted from HDF5 attributes."""

    tile_name: str
    cycle: int
    pass_number: int
    swath_side: str  # "L" or "R"
    transmit_antenna: str  # "plus_y" or "minus_y"
    wavelength: float  # meters
    near_range: float  # meters
    range_spacing: float  # meters
    azimuth_resolution: float  # meters
    lat_bounds: tuple[float, float]  # (min, max)
    lon_bounds: tuple[float, float]  # (min, max)
    num_lines: int
    num_pixels: int
    time_start: str
    time_end: str


@dataclass(frozen=True)
class SLCProducts:
    """Processed products from SLC data."""

    sigma0_plus: np.ndarray  # Calibrated backscatter, plus_y antenna
    sigma0_minus: np.ndarray  # Calibrated backscatter, minus_y antenna
    ifgram_power: np.ndarray  # Normalized interferogram magnitude
    ifgram_phase: np.ndarray  # Interferometric phase (radians)


class SLCConnector:
    """Materializes SWOT L1B HR SLC data from HDF5 files.

    The SLC connector reads raw SAR complex data and can optionally
    process it to calibrated sigma0 and interferometric products.

    Radar geometry uses pixel coordinates (no CRS) since the data
    is in native slant-range/azimuth space.
    """

    @property
    def name(self) -> str:
        return "slc"

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
        process: bool = False,
        az_looks: int = 4,
        rg_looks: int = 4,
    ) -> MaterializeResult:
        """Materialize an SLC file as an artifact.

        Args:
            source_ref: Path to the HDF5 SLC file.
            workspace: Where to write processed products if process=True.
            lazy: If True, only extract metadata without reading pixel data.
            process: If True, process to sigma0 and interferometric products.
            az_looks: Azimuth multi-look factor (only if process=True).
            rg_looks: Range multi-look factor (only if process=True).

        Returns:
            MaterializeResult with the artifact and provenance.
        """
        path = Path(source_ref).resolve()

        # Check extension first (even if file doesn't exist)
        if path.suffix.lower() not in {".h5", ".hdf5", ".he5"}:
            raise MaterializeError(source_ref, f"Not an HDF5 file: {path.suffix}")

        if not path.exists():
            raise MaterializeError(source_ref, f"File not found: {path}")

        metadata = self._read_metadata(path)

        if lazy:
            artifact = self._materialize_lazy(path, metadata)
            strategy = "lazy_handle"
        elif process:
            artifact = self._materialize_processed(path, workspace, metadata, az_looks, rg_looks)
            strategy = "normalized"
        else:
            artifact = self._materialize_raw(path, metadata)
            strategy = "wrapped_local"

        return MaterializeResult(
            artifact=artifact,
            strategy=strategy,
            source_ref=source_ref,
        )

    def discover(self, query: str | dict | None = None) -> list[CatalogEntry]:
        """List SLC files in a directory.

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

        extensions = {".h5", ".hdf5", ".he5"}
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
                                "transmit": meta.transmit_antenna,
                                "lines": meta.num_lines,
                                "pixels": meta.num_pixels,
                                "size_bytes": p.stat().st_size,
                            },
                        )
                    )
                except Exception:
                    # Skip files that can't be read as SLC
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
            "transmit_antenna": meta.transmit_antenna,
            "wavelength_m": meta.wavelength,
            "near_range_m": meta.near_range,
            "range_spacing_m": meta.range_spacing,
            "azimuth_resolution_m": meta.azimuth_resolution,
            "lat_bounds": meta.lat_bounds,
            "lon_bounds": meta.lon_bounds,
            "num_lines": meta.num_lines,
            "num_pixels": meta.num_pixels,
            "time_start": meta.time_start,
            "time_end": meta.time_end,
        }

    def process(
        self,
        source_ref: SourceRef | str,
        workspace: Path,
        az_looks: int = 4,
        rg_looks: int = 4,
    ) -> SLCProducts:
        """Process SLC file to calibrated products.

        This is a convenience method for direct processing without
        going through the full materialize flow.
        """
        path = Path(source_ref).resolve()
        return self._process_slc(path, az_looks, rg_looks)

    # -----------------------------------------------------------------------
    # Private
    # -----------------------------------------------------------------------

    def _read_metadata(self, path: Path) -> SLCMetadata:
        """Read SLC metadata from HDF5 file."""
        import h5py

        with h5py.File(str(path), "r") as ds:
            attrs = dict(ds.attrs)

            # Get shape from the actual data array
            slc_shape = ds["slc/slc_plus_y"].shape

            return SLCMetadata(
                tile_name=self._decode(attrs.get("tile_name", b"")),
                cycle=int(attrs.get("cycle_number", [0])[0]),
                pass_number=int(attrs.get("pass_number", [0])[0]),
                swath_side=self._decode(attrs.get("swath_side", b"")),
                transmit_antenna=self._decode(attrs.get("transmit_antenna", b"")),
                wavelength=float(attrs.get("wavelength", [0])[0]),
                near_range=float(attrs.get("near_range", [0])[0]),
                range_spacing=float(attrs.get("nominal_slant_range_spacing", [0])[0]),
                azimuth_resolution=float(attrs.get("slc_along_track_resolution", [0])[0]),
                lat_bounds=(
                    float(attrs.get("geospatial_lat_min", [0])[0]),
                    float(attrs.get("geospatial_lat_max", [0])[0]),
                ),
                lon_bounds=(
                    float(attrs.get("geospatial_lon_min", [0])[0]),
                    float(attrs.get("geospatial_lon_max", [0])[0]),
                ),
                num_lines=slc_shape[0],
                num_pixels=slc_shape[1],
                time_start=self._decode(attrs.get("time_coverage_start", b"")),
                time_end=self._decode(attrs.get("time_coverage_end", b"")),
            )

    def _materialize_lazy(self, path: Path, metadata: SLCMetadata) -> Artifact:
        """Create a lazy-handle artifact (metadata only)."""
        return Artifact(
            type=ArtifactType.RASTER,
            name=metadata.tile_name or path.stem,
            backing=BackingStore(
                kind=BackingStoreKind.LAZY_HANDLE,
                uri=str(path),
            ),
            spatial=SpatialDescriptor(
                crs=None,  # Radar geometry — slant-range/azimuth, not geographic
                extent=(0, 0, metadata.num_pixels, metadata.num_lines),
                resolution=(metadata.range_spacing, metadata.azimuth_resolution),
                band_count=2,  # plus_y and minus_y
            ),
            lineage=Lineage(operation="materialize", params={"lazy": True}),
            metadata={
                "source": "slc",
                "cycle": metadata.cycle,
                "pass": metadata.pass_number,
                "swath": metadata.swath_side,
                "transmit": metadata.transmit_antenna,
                "wavelength_m": metadata.wavelength,
                "geographic_bounds": {
                    "crs": "EPSG:4326",
                    "lat_bounds": metadata.lat_bounds,
                    "lon_bounds": metadata.lon_bounds,
                },
            },
        )

    def _materialize_raw(self, path: Path, metadata: SLCMetadata) -> Artifact:
        """Create artifact wrapping the raw HDF5 file."""
        return Artifact(
            type=ArtifactType.RASTER,
            name=metadata.tile_name or path.stem,
            backing=BackingStore(
                kind=BackingStoreKind.LOCAL_FILE,
                uri=str(path),
                size_bytes=path.stat().st_size,
                content_hash=content_hash(path),
            ),
            spatial=SpatialDescriptor(
                crs=None,  # Radar geometry — slant-range/azimuth, not geographic
                extent=(0, 0, metadata.num_pixels, metadata.num_lines),
                resolution=(metadata.range_spacing, metadata.azimuth_resolution),
                band_count=2,
            ),
            lineage=Lineage(operation="materialize"),
            metadata={
                "source": "slc",
                "format": "hdf5",
                "cycle": metadata.cycle,
                "pass": metadata.pass_number,
                "geographic_bounds": {
                    "crs": "EPSG:4326",
                    "lat_bounds": metadata.lat_bounds,
                    "lon_bounds": metadata.lon_bounds,
                },
            },
        )

    def _materialize_processed(
        self,
        path: Path,
        workspace: Path,
        metadata: SLCMetadata,
        az_looks: int,
        rg_looks: int,
    ) -> Artifact:
        """Process SLC and materialize products as artifacts."""
        products = self._process_slc(path, az_looks, rg_looks)

        # Create output directory for this SLC
        out_dir = workspace / (metadata.tile_name or path.stem)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Export products as GeoTIFFs
        from rasterio.transform import from_bounds

        shape = products.sigma0_plus.shape
        transform = from_bounds(0, 0, shape[1], shape[0], shape[1], shape[0])

        product_paths = {}
        for name, data in [
            ("sigma0_plus", products.sigma0_plus),
            ("sigma0_minus", products.sigma0_minus),
            ("ifgram_power", products.ifgram_power),
            ("ifgram_phase", products.ifgram_phase),
        ]:
            out_path = out_dir / f"{name}.tif"
            self._write_product(out_path, data, transform)
            product_paths[name] = str(out_path)

        # Create artifact pointing to the product directory
        # Use sigma0_plus as the primary backing
        primary_path = Path(product_paths["sigma0_plus"])

        return Artifact(
            type=ArtifactType.RASTER,
            name=metadata.tile_name or path.stem,
            backing=BackingStore(
                kind=BackingStoreKind.LOCAL_FILE,
                uri=str(primary_path),
                size_bytes=sum(Path(p).stat().st_size for p in product_paths.values()),
            ),
            spatial=SpatialDescriptor(
                crs=None,  # Radar geometry - pixel coordinates
                extent=(0, 0, shape[1], shape[0]),
                resolution=(
                    metadata.range_spacing * rg_looks,
                    metadata.azimuth_resolution * az_looks,
                ),
                band_count=1,  # Primary backing is sigma0_plus (1 band)
            ),
            lineage=Lineage(
                operation="slc_process",
                params={
                    "az_looks": az_looks,
                    "rg_looks": rg_looks,
                    "source": str(path),
                },
            ),
            metadata={
                "source": "slc",
                "processed": True,
                "products": product_paths,
                "cycle": metadata.cycle,
                "pass": metadata.pass_number,
            },
        )

    def _process_slc(
        self,
        path: Path,
        az_looks: int = 4,
        rg_looks: int = 4,
    ) -> SLCProducts:
        """Process SLC to calibrated products."""
        import h5py

        with h5py.File(str(path), "r") as ds:
            # Read complex SLC for both antennas
            slc_plus = self._read_complex(ds, "slc/slc_plus_y")
            slc_minus = self._read_complex(ds, "slc/slc_minus_y")

            # Read calibration data
            xfactor_plus = ds["xfactor/xfactor_plus_y"][:]
            xfactor_minus = ds["xfactor/xfactor_minus_y"][:]
            noise_plus = ds["noise/noise_plus_y"][:]
            noise_minus = ds["noise/noise_minus_y"][:]

        # Compute sigma0
        sigma0_plus = self._compute_sigma0(slc_plus, xfactor_plus, noise_plus)
        sigma0_minus = self._compute_sigma0(slc_minus, xfactor_minus, noise_minus)

        # Interferometric product
        interferogram = slc_plus * np.conj(slc_minus)

        # Multi-look
        sigma0_plus = self._multilook(sigma0_plus, az_looks, rg_looks)
        sigma0_minus = self._multilook(sigma0_minus, az_looks, rg_looks)
        interferogram = self._normalize_multilooked_interferogram(
            interferogram,
            slc_plus,
            slc_minus,
            az_looks,
            rg_looks,
        )

        ifgram_power = np.abs(interferogram).astype(np.float32)
        ifgram_phase = np.angle(interferogram).astype(np.float32)

        # Mask invalid
        invalid = ~np.isfinite(ifgram_power) | (ifgram_power <= 0)
        ifgram_power[invalid] = np.nan
        ifgram_phase[invalid] = np.nan

        return SLCProducts(
            sigma0_plus=sigma0_plus,
            sigma0_minus=sigma0_minus,
            ifgram_power=ifgram_power,
            ifgram_phase=ifgram_phase,
        )

    def _read_complex(self, ds, key: str) -> np.ndarray:
        """Read complex SLC stored as (lines, pixels, 2) [real, imag]."""

        raw = ds[key][:]  # (lines, pixels, 2)
        real = raw[:, :, 0].astype(np.float32)
        imag = raw[:, :, 1].astype(np.float32)
        invalid = (
            ~np.isfinite(real)
            | ~np.isfinite(imag)
            | (np.abs(real) >= 1e20)
            | (np.abs(imag) >= 1e20)
        )
        slc = real + 1j * imag
        slc[invalid] = np.nan + 1j * np.nan
        return slc

    def _compute_sigma0(
        self, slc: np.ndarray, xfactor: np.ndarray, noise: np.ndarray
    ) -> np.ndarray:
        """Compute calibrated sigma0: sigma0 = (|SLC|^2 - noise) / xfactor."""
        power = (np.abs(slc) ** 2).astype(np.float32)
        finite_power = np.isfinite(power)

        # Mask invalid: zero-padded lines + invalid xfactor
        has_signal = np.any(finite_power & (power > 0), axis=1)
        valid = (
            has_signal[:, np.newaxis]
            & finite_power
            & (xfactor != 0)
            & np.isfinite(xfactor)
            & (np.abs(xfactor) < 1e20)
            & np.isfinite(noise[:, np.newaxis])
            & (np.abs(noise[:, np.newaxis]) < 1e20)
        )

        sigma0 = np.full(power.shape, np.nan, dtype=np.float32)
        noise_2d = np.broadcast_to(noise[:, np.newaxis], power.shape)
        sigma0[valid] = (power[valid] - noise_2d[valid]) / xfactor[valid]

        return sigma0

    def _multilook(self, data: np.ndarray, az_looks: int, rg_looks: int) -> np.ndarray:
        """Reduce resolution by block-averaging."""
        lines, pixels = data.shape
        al = (lines // az_looks) * az_looks
        rl = (pixels // rg_looks) * rg_looks
        trimmed = data[:al, :rl]
        blocks = trimmed.reshape(al // az_looks, az_looks, rl // rg_looks, rg_looks)
        valid = np.isfinite(blocks)
        counts = valid.sum(axis=(1, 3))
        sums = np.where(valid, blocks, 0).sum(axis=(1, 3), dtype=np.float64)
        out = np.full(counts.shape, np.nan, dtype=np.float32)
        np.divide(sums, counts, out=out, where=counts > 0)
        return out

    def _multilook_complex(self, data: np.ndarray, az_looks: int, rg_looks: int) -> np.ndarray:
        """Reduce a complex raster by coherent block averaging."""
        lines, pixels = data.shape
        al = (lines // az_looks) * az_looks
        rl = (pixels // rg_looks) * rg_looks
        trimmed = data[:al, :rl]
        blocks = trimmed.reshape(al // az_looks, az_looks, rl // rg_looks, rg_looks)
        valid = np.isfinite(blocks.real) & np.isfinite(blocks.imag)
        counts = valid.sum(axis=(1, 3))
        sums = np.where(valid, blocks, 0).sum(axis=(1, 3), dtype=np.complex128)
        out = np.full(counts.shape, np.nan + 1j * np.nan, dtype=np.complex64)
        np.divide(sums, counts, out=out, where=counts > 0)
        return out

    def _normalize_multilooked_interferogram(
        self,
        interferogram: np.ndarray,
        slc_plus: np.ndarray,
        slc_minus: np.ndarray,
        az_looks: int,
        rg_looks: int,
    ) -> np.ndarray:
        """Compute multilooked normalized conjugate product."""
        numerator = self._multilook_complex(interferogram, az_looks, rg_looks)
        plus_power = self._multilook((np.abs(slc_plus) ** 2).astype(np.float32), az_looks, rg_looks)
        minus_power = self._multilook(
            (np.abs(slc_minus) ** 2).astype(np.float32), az_looks, rg_looks
        )

        denominator = np.sqrt(plus_power * minus_power).astype(np.float32)
        valid = np.isfinite(denominator) & (denominator > 0)

        normalized = np.full(numerator.shape, np.nan + 1j * np.nan, dtype=np.complex64)
        np.divide(numerator, denominator, out=normalized, where=valid)

        # Clip magnitude overshoots to 1.0
        magnitude = np.abs(normalized)
        overshoot = np.isfinite(magnitude) & (magnitude > 1.0)
        normalized[overshoot] /= magnitude[overshoot].astype(np.complex64)

        return normalized

    def _write_product(
        self, path: Path, data: np.ndarray, transform, nodata: float = np.nan
    ) -> None:
        """Write a product array as GeoTIFF with pixel coordinates."""
        import rasterio

        path.parent.mkdir(parents=True, exist_ok=True)

        profile = {
            "driver": "GTiff",
            "height": data.shape[0],
            "width": data.shape[1],
            "count": 1,
            "dtype": str(data.dtype),
            "crs": None,  # Pixel coordinates
            "transform": transform,
            "compress": "deflate",
            "nodata": nodata,
        }

        with rasterio.open(str(path), "w", **profile) as dst:
            dst.write(data, 1)

    def _decode(self, val) -> str:
        """Decode bytes to string."""
        if isinstance(val, bytes):
            return val.decode()
        if isinstance(val, np.ndarray):
            return str(val.item()) if val.size == 1 else str(val)
        return str(val)

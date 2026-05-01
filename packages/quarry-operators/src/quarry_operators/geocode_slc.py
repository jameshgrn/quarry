"""GeocodeSLCOperator — projects radar-space SLC products to EPSG:4326.

Lane: operator

Takes a radar-space raster artifact (σ0 from SLCConnector) and the source
HDF5 path, builds range-Doppler GCPs from orbit/geometry, then warps to a
regular geographic grid via GDAL.

Input: one RASTER artifact with crs=None (radar geometry)
Output: one RASTER artifact with crs=EPSG:4326

The geocoding uses terrain-aware GCPs when GRDEM is available in the SLC
file, falling back to ellipsoid intersection otherwise.

Reference geometry: JPL D-56410 SWOT Product Description L1B HR SLC
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rasterio
from quarry_core.artifact import (
    Artifact,
    ArtifactType,
    BackingStore,
    BackingStoreKind,
    CheckResult,
    Lineage,
    SpatialDescriptor,
    ValidationState,
    content_hash,
)
from quarry_core.operator import (
    OperatorError,
    OperatorParams,
    OperatorResult,
    OperatorSpec,
    ResourceScale,
)
from rasterio import CRS
from rasterio.control import GroundControlPoint
from rasterio.enums import Resampling
from rasterio.errors import NotGeoreferencedWarning
from rasterio.transform import from_bounds
from rasterio.warp import reproject

# WGS84 ellipsoid constants
_A = 6378137.0
_B = 6356752.314245
_E2 = 1.0 - (_B / _A) ** 2


@dataclass(frozen=True)
class GeocodeSLCParams(OperatorParams):
    """Parameters for SLC geocoding."""

    slc_path: str  # Path to source L1B HR SLC HDF5
    output_path: str  # Where to write the projected GeoTIFF
    resolution_m: float = 15.0  # Target ground resolution in meters
    az_looks: int = 4  # Azimuth look factor used to create input
    rg_looks: int = 4  # Range look factor used to create input
    resampling: str = "bilinear"  # GDAL resampling method
    pixc_path: str | None = None  # Path to L2 HR PIXC — uses JPL's geolocation directly
    n_az_gcps: int = 64  # GCP density along azimuth (fallback if no PIXC)
    n_rg_gcps: int = 32  # GCP density along range


class GeocodeSLCOperator:
    """Projects radar-space SLC products to EPSG:4326."""

    @property
    def name(self) -> str:
        return "geocode_slc"

    @property
    def spec(self) -> OperatorSpec:
        return OperatorSpec(
            input_types=(ArtifactType.RASTER,),
            output_type=ArtifactType.RASTER,
            min_inputs=1,
            max_inputs=1,
            resource_scale=ResourceScale.MEDIUM,
        )

    def validate_inputs(self, inputs: list[Artifact], params: OperatorParams) -> list[str]:
        errors = []

        if not inputs:
            errors.append("Exactly one input artifact required")
            return errors

        if len(inputs) > 1:
            errors.append(f"Expected 1 input, got {len(inputs)}")

        artifact = inputs[0]

        if artifact.type != ArtifactType.RASTER:
            errors.append(f"Input must be raster, got {artifact.type.value}")

        if not artifact.is_materialized:
            errors.append("Input artifact is not materialized")

        # Should have crs=None (radar geometry)
        if artifact.spatial.crs is not None:
            errors.append(
                f"Input already has CRS {artifact.spatial.crs} — "
                "geocode_slc is for radar-space data with crs=None"
            )

        if not isinstance(params, GeocodeSLCParams):
            errors.append("Params must be GeocodeSLCParams")
            return errors

        if not Path(params.slc_path).exists():
            errors.append(f"SLC file not found: {params.slc_path}")

        if not params.output_path:
            errors.append("output_path is required")

        if params.resolution_m <= 0:
            errors.append(f"resolution_m must be positive, got {params.resolution_m}")

        return errors

    def execute(self, inputs: list[Artifact], params: OperatorParams) -> OperatorResult:
        if not isinstance(params, GeocodeSLCParams):
            raise OperatorError(self.name, "Params must be GeocodeSLCParams")

        artifact = inputs[0]
        input_path = artifact.backing.uri
        output_path = Path(params.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Read the radar-space raster
        with rasterio.open(input_path) as src:
            data = src.read(1).astype(np.float32)

        # Build GCPs — prefer PIXC (JPL's geolocation) over range-Doppler
        if params.pixc_path and Path(params.pixc_path).exists():
            gcps = _build_pixc_gcps(
                params.pixc_path,
                params.slc_path,
                data_shape=data.shape,
                az_looks=params.az_looks,
                rg_looks=params.rg_looks,
                max_gcps=3000,
            )
        else:
            gcps = _build_slc_gcps(
                params.slc_path,
                data_shape=data.shape,
                az_looks=params.az_looks,
                rg_looks=params.rg_looks,
                n_az=params.n_az_gcps,
                n_rg=params.n_rg_gcps,
            )

        # Convert meters to degrees (approximate)
        resolution_deg = params.resolution_m / 111320.0

        # Map resampling string to enum
        resampling_map = {
            "nearest": Resampling.nearest,
            "bilinear": Resampling.bilinear,
            "cubic": Resampling.cubic,
            "lanczos": Resampling.lanczos,
            "average": Resampling.average,
        }
        resampling = resampling_map.get(params.resampling, Resampling.bilinear)

        # Warp to EPSG:4326 — polynomial handles edge extrapolation gracefully
        warped, transform, dst_width, dst_height = _warp_with_gcps(
            data, gcps, resolution=resolution_deg, resampling=resampling
        )

        # Write output GeoTIFF
        dst_crs = CRS.from_epsg(4326)
        profile = {
            "driver": "GTiff",
            "height": dst_height,
            "width": dst_width,
            "count": 1,
            "dtype": "float32",
            "crs": dst_crs,
            "transform": transform,
            "compress": "deflate",
            "nodata": np.nan,
        }
        with rasterio.open(str(output_path), "w", **profile) as dst:
            dst.write(warped, 1)

        # Build output artifact from actual file
        with rasterio.open(str(output_path)) as out:
            bounds = out.bounds
            output_artifact = Artifact(
                type=ArtifactType.RASTER,
                name=output_path.stem,
                backing=BackingStore(
                    kind=BackingStoreKind.LOCAL_FILE,
                    uri=str(output_path),
                    size_bytes=output_path.stat().st_size,
                    content_hash=content_hash(output_path),
                ),
                spatial=SpatialDescriptor(
                    crs="EPSG:4326",
                    extent=(bounds.left, bounds.bottom, bounds.right, bounds.top),
                    resolution=(out.res[0], out.res[1]),
                    band_count=1,
                ),
                lineage=Lineage(
                    operation=self.name,
                    inputs=(artifact.id,),
                    params={
                        "slc_path": params.slc_path,
                        "resolution_m": params.resolution_m,
                        "az_looks": params.az_looks,
                        "rg_looks": params.rg_looks,
                        "resampling": params.resampling,
                        "n_gcps": len(gcps),
                    },
                ),
                metadata={
                    "source": "slc",
                    "geocoded": True,
                    "n_gcps": len(gcps),
                },
            )

        checks = self._run_checks(output_artifact, artifact, params, len(gcps))
        return OperatorResult(artifact=output_artifact, checks=checks)

    def declared_checks(self) -> list[str]:
        return [
            "crs_valid",
            "extent_sane",
            "gcp_count_sufficient",
            "backing_accessible",
        ]

    def _run_checks(
        self,
        output: Artifact,
        input_artifact: Artifact,
        params: GeocodeSLCParams,
        n_gcps: int,
    ) -> list[CheckResult]:
        results = []

        # CRS valid
        results.append(
            CheckResult(
                check_name="crs_valid",
                state=ValidationState.VALID if output.spatial.crs else ValidationState.INVALID,
                message=f"CRS: {output.spatial.crs}",
            )
        )

        # Extent sane
        if output.spatial.extent:
            xmin, ymin, xmax, ymax = output.spatial.extent
            state = (
                ValidationState.VALID if (xmin < xmax and ymin < ymax) else ValidationState.INVALID
            )
            results.append(
                CheckResult(
                    check_name="extent_sane",
                    state=state,
                    message=f"Extent: ({xmin:.4f}, {ymin:.4f}, {xmax:.4f}, {ymax:.4f})",
                )
            )

        # GCP count
        gcp_state = ValidationState.VALID if n_gcps >= 16 else ValidationState.WARN
        results.append(
            CheckResult(
                check_name="gcp_count_sufficient",
                state=gcp_state,
                message=f"{n_gcps} GCPs used for warping",
            )
        )

        # Backing accessible
        exists = output.backing and Path(output.backing.uri).exists()
        results.append(
            CheckResult(
                check_name="backing_accessible",
                state=ValidationState.VALID if exists else ValidationState.INVALID,
                message=f"File exists: {output.backing.uri}" if exists else "Output file not found",
            )
        )

        return results


# ─────────────────────────────────────────────────────────────────────────────
# Geometry extraction from SLC HDF5
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _SLCGeometry:
    near_range: float
    range_spacing: float
    first_tvp: int
    num_lines: int
    num_pixels: int
    swath_side: str
    transmit_antenna: str
    pos_x: np.ndarray = field(repr=False)
    pos_y: np.ndarray = field(repr=False)
    pos_z: np.ndarray = field(repr=False)
    vel_x: np.ndarray = field(repr=False)
    vel_y: np.ndarray = field(repr=False)
    vel_z: np.ndarray = field(repr=False)
    tvp_latitude: np.ndarray = field(repr=False)
    tvp_longitude: np.ndarray = field(repr=False)
    tvp_time: np.ndarray = field(repr=False)
    grdem_height: np.ndarray = field(repr=False)
    grdem_time: np.ndarray = field(repr=False)
    grdem_cross_track_spacing: float
    grdem_min_cross_track: float
    grdem_lat: np.ndarray = field(repr=False)  # platform latitude per GRDEM line
    grdem_lon: np.ndarray = field(repr=False)  # platform longitude per GRDEM line
    grdem_alt: np.ndarray = field(repr=False)  # platform altitude per GRDEM line
    grdem_vx: np.ndarray = field(repr=False)  # platform velocity ECEF
    grdem_vy: np.ndarray = field(repr=False)
    grdem_vz: np.ndarray = field(repr=False)
    corners: dict[str, tuple[float, float]]


def _read_geometry(slc_path: str | Path) -> _SLCGeometry:
    import h5py

    with h5py.File(str(slc_path), "r") as ds:
        attrs = ds.attrs
        transmit_antenna = _decode(attrs.get("transmit_antenna", b""))
        pos_x_key, pos_y_key, pos_z_key = _position_keys(ds, transmit_antenna)
        grdem = ds.get("grdem")

        return _SLCGeometry(
            near_range=float(attrs["near_range"][0]),
            range_spacing=float(attrs["nominal_slant_range_spacing"][0]),
            first_tvp=int(attrs["slc_first_line_index_in_tvp"][0]),
            num_lines=ds["slc/slc_plus_y"].shape[0],
            num_pixels=ds["slc/slc_plus_y"].shape[1],
            swath_side=_decode(attrs.get("swath_side", b"")),
            transmit_antenna=transmit_antenna,
            pos_x=ds[pos_x_key][:],
            pos_y=ds[pos_y_key][:],
            pos_z=ds[pos_z_key][:],
            vel_x=ds["tvp/vx"][:],
            vel_y=ds["tvp/vy"][:],
            vel_z=ds["tvp/vz"][:],
            tvp_latitude=ds["tvp/latitude"][:],
            tvp_longitude=ds["tvp/longitude"][:],
            tvp_time=ds["tvp/time"][:],
            grdem_height=grdem["height"][:] if grdem is not None else np.empty((0, 0)),
            grdem_time=grdem["platform_time"][:] if grdem is not None else np.empty(0),
            grdem_cross_track_spacing=(
                float(grdem.attrs["grdem_cross_track_spacing"][0]) if grdem is not None else np.nan
            ),
            grdem_min_cross_track=(
                float(grdem.attrs["grdem_min_cross_track"][0]) if grdem is not None else np.nan
            ),
            grdem_lat=grdem["platform_latitude"][:] if grdem is not None else np.empty(0),
            grdem_lon=grdem["platform_longitude"][:] if grdem is not None else np.empty(0),
            grdem_alt=grdem["platform_altitude"][:] if grdem is not None else np.empty(0),
            grdem_vx=grdem["platform_velocity_x"][:] if grdem is not None else np.empty(0),
            grdem_vy=grdem["platform_velocity_y"][:] if grdem is not None else np.empty(0),
            grdem_vz=grdem["platform_velocity_z"][:] if grdem is not None else np.empty(0),
            corners={
                "inner_first": (
                    float(attrs["inner_first_latitude"][0]),
                    float(attrs["inner_first_longitude"][0]),
                ),
                "inner_last": (
                    float(attrs["inner_last_latitude"][0]),
                    float(attrs["inner_last_longitude"][0]),
                ),
                "outer_first": (
                    float(attrs["outer_first_latitude"][0]),
                    float(attrs["outer_first_longitude"][0]),
                ),
                "outer_last": (
                    float(attrs["outer_last_latitude"][0]),
                    float(attrs["outer_last_longitude"][0]),
                ),
            },
        )


# ─────────────────────────────────────────────────────────────────────────────
# GCP construction — PIXC (preferred)
# ─────────────────────────────────────────────────────────────────────────────


def _build_pixc_gcps(
    pixc_path: str | Path,
    slc_path: str | Path,
    data_shape: tuple[int, int],
    az_looks: int,
    rg_looks: int,
    max_gcps: int = 2000,
) -> list[GroundControlPoint]:
    """Build GCPs from L2 HR PIXC geolocation — JPL's operational pipeline.

    The PIXC contains per-pixel lat/lon computed with full corrections
    (terrain, troposphere, ionosphere, tides). We subsample these as GCPs
    for warping the multilooked SLC products.

    Mapping: PIXC azimuth_index → pixc_line_to_tvp → TVP index → SLC line
             PIXC range_index → SLC range pixel (direct)
    """
    import h5py

    height, width = data_shape

    # Get SLC first_tvp for the TVP→SLC mapping
    with h5py.File(str(slc_path), "r") as ds:
        first_tvp = int(ds.attrs["slc_first_line_index_in_tvp"][0])

    with h5py.File(str(pixc_path), "r") as ds:
        pc = ds["pixel_cloud"]
        az_idx = pc["azimuth_index"][:]
        rg_idx = pc["range_index"][:]
        lat = pc["latitude"][:]
        lon = pc["longitude"][:]
        l2tvp = pc["pixc_line_to_tvp"][:]

    # Map PIXC coords to SLC coords
    tvp_indices = l2tvp[az_idx]
    slc_lines = tvp_indices - first_tvp
    slc_cols = rg_idx.astype(np.float64)

    # Convert to multilooked data coordinates
    data_rows = (slc_lines + 0.5) / az_looks - 0.5
    data_cols = (slc_cols + 0.5) / rg_looks - 0.5

    # Filter to valid data coords
    valid = (
        np.isfinite(lat)
        & np.isfinite(lon)
        & (data_rows >= 0)
        & (data_rows < height)
        & (data_cols >= 0)
        & (data_cols < width)
    )
    data_rows = data_rows[valid]
    data_cols = data_cols[valid]
    lat = lat[valid]
    lon = lon[valid]

    n_valid = len(data_rows)
    if n_valid == 0:
        return []

    # Subsample uniformly
    if n_valid > max_gcps:
        step = n_valid // max_gcps
        indices = np.arange(0, n_valid, step)[:max_gcps]
    else:
        indices = np.arange(n_valid)

    gcps = [
        GroundControlPoint(
            row=float(data_rows[i]),
            col=float(data_cols[i]),
            x=float(lon[i]),
            y=float(lat[i]),
        )
        for i in indices
    ]

    return gcps


# ─────────────────────────────────────────────────────────────────────────────
# GCP construction — range-Doppler fallback
# ─────────────────────────────────────────────────────────────────────────────


def _build_slc_gcps(
    slc_path: str | Path,
    data_shape: tuple[int, int],
    az_looks: int,
    rg_looks: int,
    n_az: int,
    n_rg: int,
) -> list[GroundControlPoint]:
    """Build GCPs for a radar-space raster.

    Uses GRDEM terrain + platform geometry when available. The GRDEM approach
    steps outward from nadir in the cross-track direction using actual platform
    velocity for bearing, samples terrain height, and computes slant range back
    to the satellite to find the SLC column. This covers the full swath.

    Falls back to range-Doppler ellipsoid intersection if GRDEM not available.
    """
    geom = _read_geometry(slc_path)
    if geom.grdem_height.size and len(geom.grdem_lat) > 0:
        gcps = _build_grdem_gcps(geom, data_shape, az_looks, rg_looks, n_az, n_rg)
        if len(gcps) >= 10:
            return gcps
    return _build_ellipsoid_gcps(geom, data_shape, az_looks, rg_looks, n_az, n_rg)


def _build_ellipsoid_gcps(
    geom: _SLCGeometry,
    data_shape: tuple[int, int],
    az_looks: int,
    rg_looks: int,
    n_az: int,
    n_rg: int,
) -> list[GroundControlPoint]:
    """Fallback GCPs by intersecting SLC ranges with the WGS84 ellipsoid."""
    height, width = data_shape
    sign = _choose_cross_track_sign(geom)
    rows = _sample_axis(height, n_az)
    cols = _sample_axis(width, n_rg)
    gcps: list[GroundControlPoint] = []

    for row in rows:
        slc_row = (row + 0.5) * az_looks - 0.5
        for col in cols:
            slc_col = (col + 0.5) * rg_looks - 0.5
            lat, lon = _latlon_for_pixel(geom, slc_row, slc_col, sign)
            if np.isfinite(lat) and np.isfinite(lon):
                gcps.append(
                    GroundControlPoint(row=float(row), col=float(col), x=float(lon), y=float(lat))
                )

    if len(gcps) < 4:
        raise OperatorError(
            "geocode_slc",
            f"Only {len(gcps)} valid GCPs generated — check SLC orbit metadata",
        )
    return gcps


def _build_grdem_gcps(
    geom: _SLCGeometry,
    data_shape: tuple[int, int],
    az_looks: int,
    rg_looks: int,
    n_az: int,
    n_rg: int,
) -> list[GroundControlPoint]:
    """Terrain-aware GCPs using GRDEM platform positions + cross-track stepping.

    For each sampled GRDEM along-track line:
    1. Get platform position (ECEF) and velocity → compute cross-track bearing
    2. Step outward from nadir at GRDEM cross-track spacing
    3. Sample terrain height from GRDEM grid
    4. Compute slant range from satellite to ground point
    5. Map slant range to SLC pixel column → multilooked data column
    """
    height, width = data_shape
    n_grdem_lines = len(geom.grdem_lat)
    n_grdem_pixels = geom.grdem_height.shape[1] if geom.grdem_height.ndim == 2 else 0
    if n_grdem_lines == 0 or n_grdem_pixels == 0:
        return []

    # Map GRDEM lines to SLC/data rows via time interpolation
    tvp_sample = np.arange(len(geom.tvp_time), dtype=np.float64)

    # Sample subset of GRDEM lines
    az_step = max(1, n_grdem_lines // n_az)
    rg_step = max(1, n_grdem_pixels // n_rg)

    # Cross-track sign: left-looking → step left of velocity
    ct_sign = -1 if geom.swath_side.upper() == "L" else 1

    gcps: list[GroundControlPoint] = []

    for gi in range(0, n_grdem_lines, az_step):
        # Platform state from GRDEM (more aligned than TVP interpolation)
        plat_lat = float(geom.grdem_lat[gi])
        plat_lon = float(geom.grdem_lon[gi])
        plat_alt = float(geom.grdem_alt[gi])
        sat_ecef = _lla_to_ecef(plat_lat, plat_lon, plat_alt)

        # Velocity in ECEF → compute cross-track bearing on the ground
        vx = float(geom.grdem_vx[gi])
        vy = float(geom.grdem_vy[gi])
        vz = float(geom.grdem_vz[gi])
        bearing = _cross_track_bearing(plat_lat, plat_lon, vx, vy, vz, ct_sign)

        # Map GRDEM time to SLC row → data row
        grdem_time = float(geom.grdem_time[gi])
        tvp_idx = float(np.interp(grdem_time, geom.tvp_time, tvp_sample))
        slc_row = tvp_idx - geom.first_tvp
        if slc_row < 0 or slc_row >= geom.num_lines:
            continue
        data_row = (slc_row + 0.5) / az_looks - 0.5
        if data_row < 0 or data_row >= height:
            continue

        for gj in range(0, n_grdem_pixels, rg_step):
            ct_dist = geom.grdem_min_cross_track + gj * geom.grdem_cross_track_spacing
            terrain_h = float(geom.grdem_height[gi, gj])
            if not np.isfinite(terrain_h) or abs(terrain_h) >= 1e20:
                continue

            # Step from nadir in cross-track direction
            ground_lat, ground_lon = _destination_point(plat_lat, plat_lon, bearing, ct_dist)

            # Ground point in ECEF with terrain height
            ground_ecef = _lla_to_ecef(ground_lat, ground_lon, terrain_h)

            # Slant range → SLC column
            slant_range = float(np.linalg.norm(ground_ecef - sat_ecef))
            slc_col = (slant_range - geom.near_range) / geom.range_spacing
            data_col = (slc_col + 0.5) / rg_looks - 0.5

            if 0.0 <= data_col <= width - 1.0:
                gcps.append(
                    GroundControlPoint(
                        row=float(data_row),
                        col=float(data_col),
                        x=float(ground_lon),
                        y=float(ground_lat),
                    )
                )

    return gcps


# ─────────────────────────────────────────────────────────────────────────────
# GDAL warp
# ─────────────────────────────────────────────────────────────────────────────


def _warp_with_gcps(
    data: np.ndarray,
    gcps: list[GroundControlPoint],
    resolution: float,
    resampling: Resampling = Resampling.bilinear,
    tps: bool = False,
) -> tuple[np.ndarray, rasterio.Affine, int, int]:
    """Warp a 2D radar-space raster to a regular EPSG:4326 grid.

    When tps=True (PIXC GCPs with dense full-swath coverage), uses
    gdalwarp -tps for exact interpolation at every GCP.
    When tps=False (range-Doppler GCPs with near-range void), uses
    rasterio polynomial warp which extrapolates smoothly.
    """
    import shutil
    import subprocess
    import tempfile

    # Mask columns outside GCP coverage
    src_data = data.astype(np.float32)
    min_col = int(np.floor(min(g.col for g in gcps)))
    max_col = int(np.ceil(max(g.col for g in gcps)))
    if min_col > 0:
        src_data[:, :min_col] = np.nan
    if max_col < src_data.shape[1] - 1:
        src_data[:, max_col + 1 :] = np.nan

    dst_crs = CRS.from_epsg(4326)

    if tps:
        gdalwarp = shutil.which("gdalwarp")
        if gdalwarp is None:
            raise OperatorError("geocode_slc", "gdalwarp not found — needed for TPS warp")

        resampling_names = {
            Resampling.nearest: "near",
            Resampling.bilinear: "bilinear",
            Resampling.cubic: "cubic",
            Resampling.lanczos: "lanczos",
            Resampling.average: "average",
        }
        resample_str = resampling_names.get(resampling, "bilinear")

        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = Path(tmpdir) / "radar.tif"
            dst_path = Path(tmpdir) / "geo.tif"

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", NotGeoreferencedWarning)
                with rasterio.open(
                    str(src_path),
                    "w",
                    driver="GTiff",
                    height=data.shape[0],
                    width=data.shape[1],
                    count=1,
                    dtype="float32",
                    nodata=np.nan,
                ) as src_ds:
                    src_ds.write(src_data, 1)
                    src_ds.gcps = (gcps, dst_crs)

            cmd = [
                gdalwarp,
                "-tps",
                "-t_srs",
                "EPSG:4326",
                "-tr",
                str(resolution),
                str(resolution),
                "-r",
                resample_str,
                "-srcnodata",
                "nan",
                "-dstnodata",
                "nan",
                "-co",
                "COMPRESS=DEFLATE",
                "-overwrite",
                str(src_path),
                str(dst_path),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                raise OperatorError("geocode_slc", f"gdalwarp TPS failed: {result.stderr.strip()}")

            with rasterio.open(str(dst_path)) as dst_ds:
                return dst_ds.read(1), dst_ds.transform, dst_ds.width, dst_ds.height

    # Polynomial fallback (rasterio reproject)
    from rasterio.io import MemoryFile

    lons = np.array([g.x for g in gcps], dtype=np.float64)
    lats = np.array([g.y for g in gcps], dtype=np.float64)
    west, east = float(np.min(lons)), float(np.max(lons))
    south, north = float(np.min(lats)), float(np.max(lats))
    pad = resolution * 2
    west -= pad
    east += pad
    south -= pad
    north += pad

    dst_width = int(np.ceil((east - west) / resolution))
    dst_height = int(np.ceil((north - south) / resolution))
    if dst_width <= 0 or dst_height <= 0:
        raise OperatorError("geocode_slc", f"Invalid grid ({dst_height}x{dst_width})")

    dst_transform = from_bounds(west, south, east, north, dst_width, dst_height)
    dst_data = np.full((dst_height, dst_width), np.nan, dtype=np.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with (
            MemoryFile() as memfile,
            memfile.open(
                driver="GTiff",
                height=data.shape[0],
                width=data.shape[1],
                count=1,
                dtype="float32",
                nodata=np.nan,
            ) as src_ds,
        ):
            src_ds.write(src_data, 1)
            src_ds.gcps = (gcps, dst_crs)
            reproject(
                source=rasterio.band(src_ds, 1),
                destination=dst_data,
                gcps=gcps,
                src_crs=dst_crs,
                src_nodata=np.nan,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                dst_nodata=np.nan,
                resampling=resampling,
            )

    return dst_data, dst_transform, dst_width, dst_height


# ─────────────────────────────────────────────────────────────────────────────
# Range-Doppler geometry
# ─────────────────────────────────────────────────────────────────────────────


def _latlon_for_pixel(
    geom: _SLCGeometry, slc_row: float, slc_col: float, sign: int
) -> tuple[float, float]:
    sat_pos, sat_vel = _state_at(geom, _tvp_index(geom, slc_row))
    slant_range = geom.near_range + slc_col * geom.range_spacing
    return _range_doppler_to_latlon(sat_pos, sat_vel, slant_range, sign)


def _range_doppler_to_latlon(
    sat_pos: np.ndarray,
    sat_vel: np.ndarray,
    slant_range: float,
    sign: int,
) -> tuple[float, float]:
    """Solve range + zero-Doppler + WGS84 ellipsoid for one pixel."""
    if slant_range <= 0:
        return np.nan, np.nan

    speed = np.linalg.norm(sat_vel)
    sat_norm = np.linalg.norm(sat_pos)
    if speed < 1.0 or sat_norm < _A:
        return np.nan, np.nan

    along = sat_vel / speed
    nadir = -sat_pos / sat_norm
    down = nadir - np.dot(nadir, along) * along
    down_norm = np.linalg.norm(down)
    if down_norm < 1e-12:
        return np.nan, np.nan
    down /= down_norm

    cross = np.cross(along, down)
    cross_norm = np.linalg.norm(cross)
    if cross_norm < 1e-12:
        return np.nan, np.nan
    cross /= cross_norm

    # Bisection to find look angle alpha where the look vector hits the ellipsoid
    bracket = _find_bracket(sat_pos, down, cross, slant_range, sign)
    if bracket is None:
        return np.nan, np.nan

    lo, hi = bracket
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        f_lo = _ellipsoid_val(_look_pt(sat_pos, down, cross, slant_range, sign, lo))
        f_mid = _ellipsoid_val(_look_pt(sat_pos, down, cross, slant_range, sign, mid))
        if f_lo == 0 or f_lo * f_mid <= 0:
            hi = mid
        else:
            lo = mid

    alpha = 0.5 * (lo + hi)
    ground = _look_pt(sat_pos, down, cross, slant_range, sign, alpha)
    lla = _ecef_to_lla(ground)
    return float(np.degrees(lla[0])), float(np.degrees(lla[1]))


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────


def _position_keys(ds, transmit_antenna: str) -> tuple[str, str, str]:
    prefix = "plus_y" if transmit_antenna == "plus_y" else "minus_y"
    keys = (f"tvp/{prefix}_antenna_x", f"tvp/{prefix}_antenna_y", f"tvp/{prefix}_antenna_z")
    if all(key in ds for key in keys):
        return keys
    return "tvp/x", "tvp/y", "tvp/z"


def _tvp_index(geom: _SLCGeometry, slc_row: float) -> float:
    return float(np.clip(geom.first_tvp + slc_row, 0.0, len(geom.pos_x) - 1.0))


def _state_at(geom: _SLCGeometry, tvp_idx: float) -> tuple[np.ndarray, np.ndarray]:
    s = np.arange(len(geom.pos_x), dtype=np.float64)
    pos = np.array(
        [
            np.interp(tvp_idx, s, geom.pos_x),
            np.interp(tvp_idx, s, geom.pos_y),
            np.interp(tvp_idx, s, geom.pos_z),
        ],
        dtype=np.float64,
    )
    vel = np.array(
        [
            np.interp(tvp_idx, s, geom.vel_x),
            np.interp(tvp_idx, s, geom.vel_y),
            np.interp(tvp_idx, s, geom.vel_z),
        ],
        dtype=np.float64,
    )
    return pos, vel


def _nadir_at(geom: _SLCGeometry, tvp_idx: float) -> tuple[float, float]:
    s = np.arange(len(geom.tvp_latitude), dtype=np.float64)
    return (
        float(np.interp(tvp_idx, s, geom.tvp_latitude)),
        float(np.interp(tvp_idx, s, geom.tvp_longitude)),
    )


def _grdem_row(geom: _SLCGeometry, tvp_idx: float) -> float:
    s = np.arange(len(geom.tvp_time), dtype=np.float64)
    slc_time = float(np.interp(tvp_idx, s, geom.tvp_time))
    gs = np.arange(len(geom.grdem_time), dtype=np.float64)
    return float(np.interp(slc_time, geom.grdem_time, gs))


def _choose_cross_track_sign(geom: _SLCGeometry) -> int:
    errors = {sign: _corner_error(geom, sign) for sign in (-1, 1)}
    best = -1 if errors[-1] <= errors[1] else 1
    if np.isfinite(errors[best]):
        return best
    return -1 if geom.swath_side.upper() == "L" else 1


def _corner_error(geom: _SLCGeometry, sign: int) -> float:
    samples = {
        "inner_first": (0.0, 0.0),
        "inner_last": (geom.num_lines - 1.0, 0.0),
        "outer_first": (0.0, geom.num_pixels - 1.0),
        "outer_last": (geom.num_lines - 1.0, geom.num_pixels - 1.0),
    }
    error = 0.0
    used = 0
    for name, (row, col) in samples.items():
        lat, lon = _latlon_for_pixel(geom, row, col, sign)
        if not np.isfinite(lat) or not np.isfinite(lon):
            continue
        t_lat, t_lon = geom.corners[name]
        error += (lat - t_lat) ** 2 + (lon - t_lon) ** 2
        used += 1
    return error if used > 0 else float("inf")


def _find_bracket(sat_pos, down, cross, slant_range, sign):
    # SWOT near-range look angles can be < 0.01 rad (nearly nadir).
    # Dense sampling at small alpha + coarser at large alpha.
    alphas = np.concatenate(
        [
            np.linspace(0.0, 0.02, 64),  # Dense near-nadir (step ~0.0003)
            np.linspace(0.02, 0.8, 128),  # Normal range
        ]
    )
    vals = np.array(
        [_ellipsoid_val(_look_pt(sat_pos, down, cross, slant_range, sign, a)) for a in alphas]
    )
    for i in range(len(alphas) - 1):
        f0, f1 = vals[i], vals[i + 1]
        if not np.isfinite(f0) or not np.isfinite(f1):
            continue
        if f0 == 0:
            return (float(alphas[i]), float(alphas[i]))
        if f0 * f1 <= 0:
            return (float(alphas[i]), float(alphas[i + 1]))
    return None


def _look_pt(sat_pos, down, cross, slant_range, sign, alpha):
    look = np.cos(alpha) * down + sign * np.sin(alpha) * cross
    return sat_pos + slant_range * look


def _ellipsoid_val(pos):
    return float((pos[0] ** 2 + pos[1] ** 2) / (_A**2) + (pos[2] ** 2) / (_B**2) - 1.0)


def _ecef_to_lla(pos):
    x, y, z = pos
    lon = np.arctan2(y, x)
    p = np.sqrt(x**2 + y**2)
    lat = np.arctan2(z, p * (1 - _E2))
    for _ in range(5):
        sin_lat = np.sin(lat)
        n = _A / np.sqrt(1 - _E2 * sin_lat**2)
        lat = np.arctan2(z + _E2 * n * sin_lat, p)
    sin_lat = np.sin(lat)
    n = _A / np.sqrt(1 - _E2 * sin_lat**2)
    alt = p / np.cos(lat) - n
    return np.array([lat, lon, alt])


def _lla_to_ecef(lat_deg, lon_deg, height_m):
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    n = _A / np.sqrt(1.0 - _E2 * sin_lat**2)
    return np.array(
        [
            (n + height_m) * cos_lat * np.cos(lon),
            (n + height_m) * cos_lat * np.sin(lon),
            (n * (1.0 - _E2) + height_m) * sin_lat,
        ],
        dtype=np.float64,
    )


def _sample_axis(size: int, requested: int) -> np.ndarray:
    if size <= 0:
        raise ValueError(f"Raster dimension must be positive, got {size}")
    if size == 1:
        return np.array([0.0], dtype=np.float64)
    count = min(max(requested, 2), size)
    return np.linspace(0.0, size - 1.0, count, dtype=np.float64)


def _sample_cross_tracks(geom: _SLCGeometry, outer_ct: float, requested: int) -> np.ndarray:
    max_ct = (
        geom.grdem_min_cross_track
        + (geom.grdem_height.shape[1] - 1) * geom.grdem_cross_track_spacing
    )
    end = min(outer_ct, max_ct)
    start = max(0.0, geom.grdem_min_cross_track)
    if end <= start:
        return np.empty(0, dtype=np.float64)
    return np.linspace(start, end, max(requested, 2), dtype=np.float64)


def _sample_grdem(geom: _SLCGeometry, row: float, cross_track: float) -> float:
    if geom.grdem_height.size == 0:
        return np.nan
    col = (cross_track - geom.grdem_min_cross_track) / geom.grdem_cross_track_spacing
    if row < 0 or col < 0:
        return np.nan
    if row > geom.grdem_height.shape[0] - 1 or col > geom.grdem_height.shape[1] - 1:
        return np.nan
    r0, c0 = int(np.floor(row)), int(np.floor(col))
    r1 = min(r0 + 1, geom.grdem_height.shape[0] - 1)
    c1 = min(c0 + 1, geom.grdem_height.shape[1] - 1)
    rf, cf = row - r0, col - c0
    vals = np.array(
        [
            float(geom.grdem_height[r0, c0]),
            float(geom.grdem_height[r0, c1]),
            float(geom.grdem_height[r1, c0]),
            float(geom.grdem_height[r1, c1]),
        ]
    )
    if not np.all(np.isfinite(vals)) or np.any(np.abs(vals) >= 1e20):
        return np.nan
    top = vals[0] * (1 - cf) + vals[1] * cf
    bot = vals[2] * (1 - cf) + vals[3] * cf
    return float(top * (1 - rf) + bot * rf)


def _cross_track_bearing(
    lat_deg: float, lon_deg: float, vx: float, vy: float, vz: float, ct_sign: int
) -> float:
    """Compute cross-track bearing from ECEF velocity at a geodetic position.

    Projects ECEF velocity into local ENU, gets heading, rotates 90° for cross-track.
    ct_sign: -1 for left-looking, +1 for right-looking.
    """
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    sin_lon, cos_lon = np.sin(lon), np.cos(lon)

    # ECEF → ENU rotation
    v_east = -sin_lon * vx + cos_lon * vy
    v_north = -sin_lat * cos_lon * vx - sin_lat * sin_lon * vy + cos_lat * vz

    heading = np.arctan2(v_east, v_north)  # radians, CW from north
    # Cross-track: perpendicular to heading
    return float(heading + ct_sign * np.pi / 2)


def _destination_point(
    lat_deg: float, lon_deg: float, bearing: float, distance_m: float
) -> tuple[float, float]:
    """Compute destination point given start, bearing (rad), and distance (m)."""
    r = 6371008.8  # Earth mean radius
    lat1 = np.radians(lat_deg)
    lon1 = np.radians(lon_deg)
    d_r = distance_m / r

    lat2 = np.arcsin(np.sin(lat1) * np.cos(d_r) + np.cos(lat1) * np.sin(d_r) * np.cos(bearing))
    lon2 = lon1 + np.arctan2(
        np.sin(bearing) * np.sin(d_r) * np.cos(lat1),
        np.cos(d_r) - np.sin(lat1) * np.sin(lat2),
    )
    return float(np.degrees(lat2)), float(np.degrees(lon2))


def _interp_edge(
    start: tuple[float, float], end: tuple[float, float], frac: float
) -> tuple[float, float]:
    f = float(np.clip(frac, 0.0, 1.0))
    return (start[0] * (1 - f) + end[0] * f, start[1] * (1 - f) + end[1] * f)


def _fraction(value: float, maximum: float) -> float:
    if maximum <= 0 or not np.isfinite(maximum):
        return 0.0
    return float(np.clip(value / maximum, 0.0, 1.0))


def _haversine_m(start: tuple[float, float], end: tuple[float, float]) -> float:
    r = 6371008.8
    lat1, lon1 = np.radians(start[0]), np.radians(start[1])
    lat2, lon2 = np.radians(end[0]), np.radians(end[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return float(2 * r * np.arcsin(np.sqrt(a)))


def _decode(val) -> str:
    if isinstance(val, bytes):
        return val.decode()
    if isinstance(val, np.ndarray):
        return str(val.item()) if val.size == 1 else str(val)
    return str(val)

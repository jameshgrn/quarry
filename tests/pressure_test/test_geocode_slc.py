"""GeocodeSLCOperator pressure test.

Lane: operator

Stress points:
1. Protocol conformance (name, spec, declared_checks, Operator isinstance)
2. validate_inputs: no inputs, too many, wrong type, already has CRS, missing SLC,
   empty output_path, negative resolution
3. Execute with ellipsoid GCPs — warp radar raster to EPSG:4326
4. Execute with PIXC GCPs — uses JPL geolocation for warping
5. Output artifact: CRS, extent, backing, band count
6. Checks: crs_valid, extent_sane, gcp_count_sufficient, backing_accessible
7. Lineage records slc_path, resolution_m, n_gcps
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import rasterio
from quarry_core.artifact import (
    Artifact,
    ArtifactType,
    BackingStore,
    BackingStoreKind,
    SpatialDescriptor,
    ValidationState,
    content_hash,
)
from quarry_core.operator import Operator, OperatorError, ResourceScale
from quarry_operators.geocode_slc import GeocodeSLCOperator, GeocodeSLCParams
from rasterio.transform import from_origin

# ---------------------------------------------------------------------------
# Constants — SWOT-like orbit at ~891 km
# ---------------------------------------------------------------------------

SAT_ALT_M = 891_000.0
EARTH_R = 6371_000.0
SAT_R = EARTH_R + SAT_ALT_M
# Orbital velocity ~7.5 km/s
V_SAT = 7500.0

N_AZ = 20  # azimuth lines
N_RG = 30  # range pixels
AZ_LOOKS = 4
RG_LOOKS = 4
NEAR_RANGE = 895_000.0  # Must exceed SAT_ALT (891km) — off-nadir viewing
RANGE_SPACING = 0.75

# Reference ground location (Mississippi River area)
REF_LAT = 30.5
REF_LON = -89.5

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _lla_to_ecef(lat_deg, lon_deg, alt_m):
    a = 6378137.0
    e2 = 6.6943799901377997e-3
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    n = a / np.sqrt(1.0 - e2 * sin_lat**2)
    x = (n + alt_m) * cos_lat * np.cos(lon)
    y = (n + alt_m) * cos_lat * np.sin(lon)
    z = (n * (1.0 - e2) + alt_m) * sin_lat
    return x, y, z


def _build_tvp_arrays(n_tvp: int):
    """Build synthetic TVP arrays for a SWOT-like orbit arc."""
    # Satellite moves north along a roughly polar orbit
    lats = np.linspace(REF_LAT - 0.5, REF_LAT + 0.5, n_tvp)
    lons = np.full(n_tvp, REF_LON)
    alts = np.full(n_tvp, SAT_ALT_M)

    px, py, pz = [], [], []
    for lat, lon, alt in zip(lats, lons, alts):
        x, y, z = _lla_to_ecef(lat, lon, alt)
        px.append(x)
        py.append(y)
        pz.append(z)

    px, py, pz = np.array(px), np.array(py), np.array(pz)

    # Velocity: finite difference of position (roughly northward)
    vx = (
        np.gradient(px)
        * V_SAT
        / np.sqrt(np.gradient(px) ** 2 + np.gradient(py) ** 2 + np.gradient(pz) ** 2 + 1e-30)
    )
    vy = (
        np.gradient(py)
        * V_SAT
        / np.sqrt(np.gradient(px) ** 2 + np.gradient(py) ** 2 + np.gradient(pz) ** 2 + 1e-30)
    )
    vz = (
        np.gradient(pz)
        * V_SAT
        / np.sqrt(np.gradient(px) ** 2 + np.gradient(py) ** 2 + np.gradient(pz) ** 2 + 1e-30)
    )

    time = np.linspace(0.0, n_tvp * 0.0028, n_tvp)  # ~2.8 ms per line

    return {
        "x": px,
        "y": py,
        "z": pz,
        "vx": vx,
        "vy": vy,
        "vz": vz,
        "latitude": lats,
        "longitude": lons,
        "time": time,
    }


@pytest.fixture
def slc_h5(tmp_path: Path) -> Path:
    """Synthetic SWOT SLC HDF5 with orbit metadata for geocoding."""
    path = tmp_path / "SWOT_L1B_HR_SLC_016_089_133L.h5"
    n_tvp = N_AZ * AZ_LOOKS + 10  # TVP has more samples than SLC lines
    tvp = _build_tvp_arrays(n_tvp)

    with h5py.File(str(path), "w") as f:
        # Root attributes
        f.attrs["near_range"] = np.array([NEAR_RANGE])
        f.attrs["nominal_slant_range_spacing"] = np.array([RANGE_SPACING])
        f.attrs["slc_first_line_index_in_tvp"] = np.array([5])
        f.attrs["swath_side"] = b"L"
        f.attrs["transmit_antenna"] = b"plus_y"
        f.attrs["wavelength"] = np.array([0.00836])
        f.attrs["cycle_number"] = np.array([16])
        f.attrs["pass_number"] = np.array([89])
        f.attrs["tile_name"] = b"133L"
        f.attrs["time_coverage_start"] = b"2024-06-01T08:39:25Z"
        f.attrs["time_coverage_end"] = b"2024-06-01T08:39:36Z"
        f.attrs["geospatial_lat_min"] = np.array([30.0])
        f.attrs["geospatial_lat_max"] = np.array([31.0])
        f.attrs["geospatial_lon_min"] = np.array([-90.0])
        f.attrs["geospatial_lon_max"] = np.array([-89.0])
        f.attrs["slc_along_track_resolution"] = np.array([6.0])

        # Corner coordinates
        f.attrs["inner_first_latitude"] = np.array([30.1])
        f.attrs["inner_first_longitude"] = np.array([-89.6])
        f.attrs["inner_last_latitude"] = np.array([30.9])
        f.attrs["inner_last_longitude"] = np.array([-89.6])
        f.attrs["outer_first_latitude"] = np.array([30.1])
        f.attrs["outer_first_longitude"] = np.array([-89.1])
        f.attrs["outer_last_latitude"] = np.array([30.9])
        f.attrs["outer_last_longitude"] = np.array([-89.1])

        # SLC data — complex stored as (H, W, 2)
        slc_grp = f.create_group("slc")
        slc_data = np.stack(
            [
                np.random.rand(N_AZ * AZ_LOOKS, N_RG * RG_LOOKS).astype(np.float32),
                np.random.rand(N_AZ * AZ_LOOKS, N_RG * RG_LOOKS).astype(np.float32),
            ],
            axis=-1,
        )
        slc_grp.create_dataset("slc_plus_y", data=slc_data)
        slc_grp.create_dataset("slc_minus_y", data=slc_data)

        # TVP group
        tvp_grp = f.create_group("tvp")
        tvp_grp.create_dataset("x", data=tvp["x"])
        tvp_grp.create_dataset("y", data=tvp["y"])
        tvp_grp.create_dataset("z", data=tvp["z"])
        tvp_grp.create_dataset("vx", data=tvp["vx"])
        tvp_grp.create_dataset("vy", data=tvp["vy"])
        tvp_grp.create_dataset("vz", data=tvp["vz"])
        tvp_grp.create_dataset("latitude", data=tvp["latitude"])
        tvp_grp.create_dataset("longitude", data=tvp["longitude"])
        tvp_grp.create_dataset("time", data=tvp["time"])

        # Calibration (needed for SLC structure)
        xf_grp = f.create_group("xfactor")
        xf_grp.create_dataset(
            "xfactor_plus_y", data=np.ones((N_AZ * AZ_LOOKS, N_RG * RG_LOOKS), dtype=np.float32)
        )

        noise_grp = f.create_group("noise")
        noise_grp.create_dataset(
            "noise_plus_y", data=np.full(N_AZ * AZ_LOOKS, 0.1, dtype=np.float32)
        )

    return path


@pytest.fixture
def pixc_h5(tmp_path: Path, slc_h5: Path) -> Path:
    """Synthetic PIXC file with per-pixel geolocation for GCP building."""
    path = tmp_path / "SWOT_L2_HR_PIXC_016_089_133L.nc"
    n_points = 200
    rng = np.random.default_rng(42)

    # Read first_tvp from SLC
    with h5py.File(str(slc_h5), "r") as ds:
        first_tvp = int(ds.attrs["slc_first_line_index_in_tvp"][0])
        n_slc_lines = ds["slc/slc_plus_y"].shape[0]
        n_slc_cols = ds["slc/slc_plus_y"].shape[1]

    # Generate PIXC pixels mapped to SLC grid
    n_tvp_for_slc = n_slc_lines
    az_indices = rng.integers(0, n_tvp_for_slc, n_points).astype(np.int32)
    rg_indices = rng.integers(0, n_slc_cols, n_points).astype(np.int32)

    # pixc_line_to_tvp maps azimuth_index → TVP index
    l2tvp = np.arange(first_tvp, first_tvp + n_tvp_for_slc, dtype=np.int32)

    # Generate lat/lon in the expected area
    lats = rng.uniform(30.0, 31.0, n_points).astype(np.float64)
    lons = rng.uniform(-90.0, -89.0, n_points).astype(np.float64)

    with h5py.File(str(path), "w") as f:
        f.attrs["tile_name"] = b"133L"
        f.attrs["cycle_number"] = np.array([16])
        f.attrs["pass_number"] = np.array([89])
        f.attrs["swath_side"] = b"L"
        f.attrs["time_coverage_start"] = b"2024-06-01T08:39:25Z"
        f.attrs["time_coverage_end"] = b"2024-06-01T08:39:36Z"
        f.attrs["geospatial_lat_min"] = np.array([30.0])
        f.attrs["geospatial_lat_max"] = np.array([31.0])
        f.attrs["geospatial_lon_min"] = np.array([-90.0])
        f.attrs["geospatial_lon_max"] = np.array([-89.0])

        pc = f.create_group("pixel_cloud")
        pc.create_dataset("azimuth_index", data=az_indices)
        pc.create_dataset("range_index", data=rg_indices)
        pc.create_dataset("latitude", data=lats)
        pc.create_dataset("longitude", data=lons)
        pc.create_dataset("pixc_line_to_tvp", data=l2tvp)
        pc.create_dataset("height", data=rng.uniform(100, 110, n_points).astype(np.float32))
        pc.create_dataset("sig0", data=rng.uniform(-20, 0, n_points).astype(np.float32))
        pc.create_dataset("classification", data=rng.integers(1, 8, n_points).astype(np.uint8))
        pc.create_dataset("water_frac", data=rng.uniform(0, 1, n_points).astype(np.float32))

    return path


@pytest.fixture
def radar_raster(tmp_path: Path) -> tuple[Path, Artifact]:
    """Multilooked radar-space raster (crs=None) for geocoding input."""
    path = tmp_path / "sigma0_multilooked.tif"
    data = np.random.rand(N_AZ, N_RG).astype(np.float32) * 10.0

    # No CRS, no georeferencing — radar geometry
    import warnings

    from rasterio.errors import NotGeoreferencedWarning

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(
            str(path),
            "w",
            driver="GTiff",
            height=N_AZ,
            width=N_RG,
            count=1,
            dtype="float32",
            nodata=np.nan,
        ) as dst:
            dst.write(data, 1)

    artifact = Artifact(
        type=ArtifactType.RASTER,
        name="sigma0",
        backing=BackingStore(
            kind=BackingStoreKind.LOCAL_FILE,
            uri=str(path),
            size_bytes=path.stat().st_size,
            content_hash=content_hash(path),
        ),
        spatial=SpatialDescriptor(
            crs=None,
            extent=None,
            resolution=None,
            band_count=1,
        ),
        metadata={"source": "slc", "role": "data"},
    )
    return path, artifact


@pytest.fixture
def georef_raster(tmp_path: Path) -> Artifact:
    """Already-georeferenced raster — should fail validation."""
    path = tmp_path / "already_georef.tif"
    data = np.random.rand(10, 10).astype(np.float32)
    transform = from_origin(-90, 31, 0.1, 0.1)

    with rasterio.open(
        str(path),
        "w",
        driver="GTiff",
        height=10,
        width=10,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data, 1)

    return Artifact(
        type=ArtifactType.RASTER,
        name="georef",
        backing=BackingStore(
            kind=BackingStoreKind.LOCAL_FILE,
            uri=str(path),
            size_bytes=path.stat().st_size,
            content_hash=content_hash(path),
        ),
        spatial=SpatialDescriptor(
            crs="EPSG:4326",
            extent=(-90, 30, -89, 31),
            resolution=(0.1, 0.1),
            band_count=1,
        ),
        metadata={},
    )


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_name():
    assert GeocodeSLCOperator().name == "geocode_slc"


def test_spec():
    spec = GeocodeSLCOperator().spec
    assert ArtifactType.RASTER in spec.input_types
    assert spec.output_type == ArtifactType.RASTER
    assert spec.min_inputs == 1
    assert spec.max_inputs == 1
    assert spec.resource_scale == ResourceScale.MEDIUM


def test_declared_checks():
    checks = GeocodeSLCOperator().declared_checks()
    assert "crs_valid" in checks
    assert "extent_sane" in checks
    assert "gcp_count_sufficient" in checks
    assert "backing_accessible" in checks


def test_satisfies_operator_protocol():
    assert isinstance(GeocodeSLCOperator(), Operator)


# ---------------------------------------------------------------------------
# validate_inputs
# ---------------------------------------------------------------------------


def test_validate_no_inputs(slc_h5):
    op = GeocodeSLCOperator()
    params = GeocodeSLCParams(slc_path=str(slc_h5), output_path="/tmp/out.tif")
    errors = op.validate_inputs([], params)
    assert any("one input" in e.lower() or "exactly" in e.lower() for e in errors)


def test_validate_too_many_inputs(radar_raster, slc_h5, tmp_path):
    op = GeocodeSLCOperator()
    _, art = radar_raster
    params = GeocodeSLCParams(slc_path=str(slc_h5), output_path=str(tmp_path / "out.tif"))
    errors = op.validate_inputs([art, art], params)
    assert any("1 input" in e or "expected 1" in e.lower() for e in errors)


def test_validate_already_has_crs(georef_raster, slc_h5, tmp_path):
    op = GeocodeSLCOperator()
    params = GeocodeSLCParams(slc_path=str(slc_h5), output_path=str(tmp_path / "out.tif"))
    errors = op.validate_inputs([georef_raster], params)
    assert any("already has CRS" in e for e in errors)


def test_validate_missing_slc_file(radar_raster, tmp_path):
    op = GeocodeSLCOperator()
    _, art = radar_raster
    params = GeocodeSLCParams(slc_path="/nonexistent/slc.h5", output_path=str(tmp_path / "out.tif"))
    errors = op.validate_inputs([art], params)
    assert any("not found" in e.lower() for e in errors)


def test_validate_empty_output_path(radar_raster, slc_h5):
    op = GeocodeSLCOperator()
    _, art = radar_raster
    params = GeocodeSLCParams(slc_path=str(slc_h5), output_path="")
    errors = op.validate_inputs([art], params)
    assert any("output_path" in e for e in errors)


def test_validate_negative_resolution(radar_raster, slc_h5, tmp_path):
    op = GeocodeSLCOperator()
    _, art = radar_raster
    params = GeocodeSLCParams(
        slc_path=str(slc_h5), output_path=str(tmp_path / "out.tif"), resolution_m=-5.0
    )
    errors = op.validate_inputs([art], params)
    assert any("positive" in e.lower() for e in errors)


def test_validate_wrong_params_type(radar_raster):
    from quarry_core.operator import OperatorParams

    op = GeocodeSLCOperator()
    _, art = radar_raster
    errors = op.validate_inputs([art], OperatorParams())
    assert any("GeocodeSLCParams" in e for e in errors)


def test_validate_good_inputs(radar_raster, slc_h5, tmp_path):
    op = GeocodeSLCOperator()
    _, art = radar_raster
    params = GeocodeSLCParams(slc_path=str(slc_h5), output_path=str(tmp_path / "out.tif"))
    errors = op.validate_inputs([art], params)
    assert errors == []


# ---------------------------------------------------------------------------
# Execute — ellipsoid GCPs (no PIXC)
# ---------------------------------------------------------------------------


def test_execute_ellipsoid_produces_output(radar_raster, slc_h5, tmp_path):
    op = GeocodeSLCOperator()
    _, art = radar_raster
    output_path = tmp_path / "geocoded.tif"
    params = GeocodeSLCParams(slc_path=str(slc_h5), output_path=str(output_path))

    result = op.execute([art], params)

    assert output_path.exists()
    assert result.artifact.spatial.crs == "EPSG:4326"
    assert result.artifact.backing.size_bytes > 0


def test_execute_ellipsoid_extent_sane(radar_raster, slc_h5, tmp_path):
    op = GeocodeSLCOperator()
    _, art = radar_raster
    params = GeocodeSLCParams(slc_path=str(slc_h5), output_path=str(tmp_path / "geocoded.tif"))
    result = op.execute([art], params)

    xmin, ymin, xmax, ymax = result.artifact.spatial.extent
    assert xmin < xmax
    assert ymin < ymax


def test_execute_ellipsoid_checks_pass(radar_raster, slc_h5, tmp_path):
    op = GeocodeSLCOperator()
    _, art = radar_raster
    params = GeocodeSLCParams(slc_path=str(slc_h5), output_path=str(tmp_path / "geocoded.tif"))
    result = op.execute([art], params)

    check_names = {c.check_name for c in result.checks}
    assert "crs_valid" in check_names
    assert "extent_sane" in check_names
    assert "gcp_count_sufficient" in check_names
    assert "backing_accessible" in check_names

    for c in result.checks:
        assert c.state in {ValidationState.VALID, ValidationState.WARN}


def test_execute_ellipsoid_lineage(radar_raster, slc_h5, tmp_path):
    op = GeocodeSLCOperator()
    _, art = radar_raster
    params = GeocodeSLCParams(slc_path=str(slc_h5), output_path=str(tmp_path / "geocoded.tif"))
    result = op.execute([art], params)

    lineage = result.artifact.lineage
    assert lineage.operation == "geocode_slc"
    assert lineage.params["resolution_m"] == 15.0
    assert lineage.params["n_gcps"] > 0
    assert art.id in lineage.inputs


def test_execute_ellipsoid_output_is_raster(radar_raster, slc_h5, tmp_path):
    op = GeocodeSLCOperator()
    _, art = radar_raster
    params = GeocodeSLCParams(slc_path=str(slc_h5), output_path=str(tmp_path / "geocoded.tif"))
    result = op.execute([art], params)

    with rasterio.open(result.artifact.backing.uri) as src:
        assert src.count == 1
        assert str(src.crs) == "EPSG:4326"
        data = src.read(1)
        assert data.shape[0] > 0
        assert data.shape[1] > 0


# ---------------------------------------------------------------------------
# Execute — PIXC GCPs
# ---------------------------------------------------------------------------


def test_execute_pixc_gcps(radar_raster, slc_h5, pixc_h5, tmp_path):
    op = GeocodeSLCOperator()
    _, art = radar_raster
    params = GeocodeSLCParams(
        slc_path=str(slc_h5),
        output_path=str(tmp_path / "geocoded_pixc.tif"),
        pixc_path=str(pixc_h5),
    )
    result = op.execute([art], params)

    assert result.artifact.spatial.crs == "EPSG:4326"
    assert Path(result.artifact.backing.uri).exists()
    assert result.artifact.metadata["geocoded"] is True


def test_execute_pixc_lineage_records_gcps(radar_raster, slc_h5, pixc_h5, tmp_path):
    op = GeocodeSLCOperator()
    _, art = radar_raster
    params = GeocodeSLCParams(
        slc_path=str(slc_h5),
        output_path=str(tmp_path / "geocoded_pixc.tif"),
        pixc_path=str(pixc_h5),
    )
    result = op.execute([art], params)
    assert result.artifact.lineage.params["n_gcps"] > 0


# ---------------------------------------------------------------------------
# Error in execute
# ---------------------------------------------------------------------------


def test_execute_wrong_params_type(radar_raster):
    from quarry_core.operator import OperatorParams

    op = GeocodeSLCOperator()
    _, art = radar_raster
    with pytest.raises(OperatorError, match="GeocodeSLCParams"):
        op.execute([art], OperatorParams())

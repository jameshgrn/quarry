"""Tests for SLCConnector."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from quarry_connectors import SLCConnector


def test_slc_connector_name():
    conn = SLCConnector()
    assert conn.name == "slc"


def test_slc_connector_capabilities():
    from quarry_core.connector import ConnectorCapability

    conn = SLCConnector()
    caps = conn.capabilities
    assert ConnectorCapability.MATERIALIZE in caps
    assert ConnectorCapability.DISCOVER in caps
    assert ConnectorCapability.METADATA_ONLY in caps
    assert ConnectorCapability.MATERIALIZE_LAZY in caps


# --- Internal processing tests (no HDF5 file needed) ---


def test_compute_sigma0_masks_zero_padded_lines():
    conn = SLCConnector()

    slc = np.array(
        [
            [1.0 + 1.0j, 2.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j],
        ],
        dtype=np.complex64,
    )
    xfactor = np.ones((2, 2), dtype=np.float32)
    noise = np.array([0.5, 0.0], dtype=np.float32)

    sigma0 = conn._compute_sigma0(slc, xfactor, noise)

    np.testing.assert_allclose(sigma0[0], [1.5, 3.5])
    assert np.all(np.isnan(sigma0[1]))


def test_multilook_ignores_nan():
    conn = SLCConnector()

    data = np.array(
        [
            [1.0, np.nan],
            [3.0, np.nan],
        ],
        dtype=np.float32,
    )

    out = conn._multilook(data, 2, 2)
    np.testing.assert_allclose(out, [[2.0]])


def test_complex_multilook_preserves_wrapped_phase():
    conn = SLCConnector()

    phase = np.pi - 0.1
    data = np.array([[np.exp(1j * phase), np.exp(-1j * phase)]], dtype=np.complex64)

    averaged = conn._multilook_complex(data, 1, 2)
    averaged_phase = np.angle(averaged[0, 0])

    assert abs(abs(averaged_phase) - np.pi) < 0.01


def test_normalized_interferogram_magnitude_is_bounded():
    conn = SLCConnector()

    slc_plus = np.array([[1 + 0j, 1 + 0j]], dtype=np.complex64)
    slc_minus = np.array([[1 + 0j, -1 + 0j]], dtype=np.complex64)
    interferogram = slc_plus * np.conj(slc_minus)

    normalized = conn._normalize_multilooked_interferogram(
        interferogram,
        slc_plus,
        slc_minus,
        az_looks=1,
        rg_looks=2,
    )

    np.testing.assert_allclose(np.abs(normalized), [[0.0]], atol=1e-6)


# --- Discover tests ---


def test_discover_empty_directory(tmp_path: Path):
    conn = SLCConnector()
    result = conn.discover(str(tmp_path))
    assert result == []


def test_discover_skips_non_hdf5(tmp_path: Path):
    conn = SLCConnector()
    (tmp_path / "test.txt").touch()
    (tmp_path / "test.tif").touch()
    result = conn.discover(str(tmp_path))
    assert result == []


# --- Materialize error tests ---


def test_materialize_missing_file():
    from quarry_core.connector import MaterializeError

    conn = SLCConnector()
    with pytest.raises(MaterializeError):
        conn.materialize("/nonexistent/file.h5", Path("/tmp"))


def test_materialize_non_hdf5():
    from quarry_core.connector import MaterializeError

    conn = SLCConnector()
    with pytest.raises(MaterializeError, match="Not an HDF5 file"):
        conn.materialize("/some/path/file.txt", Path("/tmp"))


# --- Metadata error tests ---


def test_metadata_missing_file():
    from quarry_core.connector import MaterializeError

    conn = SLCConnector()
    with pytest.raises(MaterializeError):
        conn.metadata("/nonexistent/file.h5")

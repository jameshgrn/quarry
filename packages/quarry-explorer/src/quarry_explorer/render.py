"""Rendering pipelines for raster layers.

Each renderer takes a numpy array and returns an RGBA uint8 array of shape
(H, W, 4). Alpha=0 encodes nodata/transparent pixels.

All functions are pure numpy — no rasterio, no side effects.
Ported from reverse_eng_swot/render.py.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image


def render_sigma0(data: np.ndarray) -> np.ndarray:
    """Render radar power as percentile-stretched grayscale.

    Input: (H, W) float. Values > 1 get dB transform; values in [0, 1]
    are stretched directly (normalized interferogram magnitudes).
    """
    if data.ndim == 3:
        data = data[0]

    h, w = data.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    valid = (data > 0) & np.isfinite(data)
    if not np.any(valid):
        return rgba

    values = np.full_like(data, np.nan, dtype=np.float64)
    valid_values = data[valid].astype(np.float64)
    if float(np.nanmax(valid_values)) <= 1.0001:
        values[valid] = valid_values
    else:
        values[valid] = 10.0 * np.log10(valid_values)

    samples = values[valid]
    lo = float(np.percentile(samples, 2))
    hi = float(np.percentile(samples, 98))
    if hi <= lo:
        hi = lo + 1.0

    stretched = np.clip((values - lo) / (hi - lo), 0, 1)
    gray = (np.nan_to_num(stretched, nan=0.0) * 255).astype(np.uint8)

    rgba[valid, 0] = gray[valid]
    rgba[valid, 1] = gray[valid]
    rgba[valid, 2] = gray[valid]
    rgba[valid, 3] = 235

    return rgba


def render_phase(data: np.ndarray) -> np.ndarray:
    """Render interferometric phase as HSV colorwheel.

    Input: (H, W) float in [-pi, pi].
    """
    if data.ndim == 3:
        data = data[0]

    h, w = data.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    valid = np.isfinite(data)
    if not np.any(valid):
        return rgba

    hue = np.where(valid, np.mod((data + np.pi) / (2 * np.pi), 1.0), 0.0)
    sat = np.ones_like(hue, dtype=np.float32)
    val = np.ones_like(hue, dtype=np.float32)
    rgb = _hsv_to_rgb(hue, sat, val)

    rgba[..., :3] = np.nan_to_num(rgb * 255, nan=0).astype(np.uint8)
    rgba[..., 3] = np.where(valid, 220, 0).astype(np.uint8)

    return rgba


def render_rgb(data: np.ndarray) -> np.ndarray:
    """Render 3-band RGB with percentile stretch.

    Input: (3, H, W) or (H, W, 3) — uint8, uint16, or float.
    """
    if data.ndim == 3 and data.shape[0] in (3, 4):
        data = np.moveaxis(data, 0, -1)

    h, w = data.shape[:2]
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgb = data[:, :, :3].astype(np.float64)

    all_zero = np.all(data[:, :, :3] == 0, axis=-1)

    if rgb.max() > 255:
        valid = rgb[~all_zero]
        if len(valid) > 0:
            lo = float(np.percentile(valid, 2))
            hi = float(np.percentile(valid, 98))
            if hi > lo:
                rgb = (rgb - lo) / (hi - lo)
    elif rgb.max() <= 1.0:
        valid = rgb[~all_zero]
        if len(valid) > 0:
            lo = float(np.percentile(valid, 2))
            hi = float(np.percentile(valid, 98))
            if hi > lo:
                rgb = (rgb - lo) / (hi - lo)

    rgba[:, :, :3] = np.clip(np.nan_to_num(rgb, nan=0.0) * 255, 0, 255).astype(np.uint8)
    rgba[:, :, 3] = np.where(all_zero, 0, 255)

    return rgba


def render_wse(data: np.ndarray) -> np.ndarray:
    """Render water surface elevation as a blue→cyan→green→yellow color ramp.

    Input: (H, W) float — elevation in meters. NaN = no water.
    """
    if data.ndim == 3:
        data = data[0]

    h, w = data.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    valid = np.isfinite(data)
    if not np.any(valid):
        return rgba

    samples = data[valid]
    lo = float(np.percentile(samples, 2))
    hi = float(np.percentile(samples, 98))
    if hi <= lo:
        hi = lo + 1.0

    t = np.clip((data - lo) / (hi - lo), 0, 1)

    # Blue → cyan → green → yellow ramp
    r = np.clip(t * 3 - 1, 0, 1)
    g = np.clip(np.minimum(t * 2, 2 - t * 2 + 1), 0, 1)
    b = np.clip(1 - t * 2, 0, 1)

    rgba[valid, 0] = (np.nan_to_num(r, nan=0)[valid] * 255).astype(np.uint8)
    rgba[valid, 1] = (np.nan_to_num(g, nan=0)[valid] * 255).astype(np.uint8)
    rgba[valid, 2] = (np.nan_to_num(b, nan=0)[valid] * 255).astype(np.uint8)
    rgba[valid, 3] = 220

    return rgba


def render_height(data: np.ndarray) -> np.ndarray:
    """Alias for WSE rendering."""
    return render_wse(data)


RENDERERS: dict[str, callable] = {
    "sigma0_plus": render_sigma0,
    "sigma0_minus": render_sigma0,
    "sigma0": render_sigma0,
    "sig0": render_sigma0,
    "ifgram_power": render_sigma0,
    "ifgram_phase": render_phase,
    "slc_phase": render_phase,
    "rgb": render_rgb,
    "wse": render_wse,
    "height": render_height,
    "water_frequency": render_sigma0,
    "stack": render_sigma0,
}


def render_layer(data: np.ndarray, layer_type: str) -> np.ndarray:
    """Dispatch to the appropriate renderer. Returns RGBA uint8 (H, W, 4)."""
    renderer = RENDERERS.get(layer_type, render_sigma0)
    return renderer(data)


def rgba_to_png(rgba: np.ndarray) -> bytes:
    """Encode RGBA uint8 array to PNG bytes."""
    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def _hsv_to_rgb(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Vectorized HSV to RGB conversion."""
    i = np.floor(h * 6).astype(np.int32)
    f = h * 6 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    i_mod = i % 6

    r = np.choose(i_mod, [v, q, p, p, t, v])
    g = np.choose(i_mod, [t, v, v, q, p, p])
    b = np.choose(i_mod, [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=-1)

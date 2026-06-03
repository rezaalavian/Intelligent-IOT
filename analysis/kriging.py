"""Simple kriging wrapper for hourly spatial interpolation.

This module provides a small convenience wrapper around `pykrige` if
available; if not installed it falls back to inverse-distance weighting.
"""
from typing import Tuple
import numpy as np

try:
    from pykrige.ok import OrdinaryKriging
except Exception:
    OrdinaryKriging = None


def kriging_interpolate(lats: np.ndarray, lons: np.ndarray, values: np.ndarray, grid_lat: np.ndarray, grid_lon: np.ndarray) -> np.ndarray:
    """Interpolate values located at (lats, lons) to grid points (grid_lat, grid_lon).

    Returns an array matching grid_lat/grid_lon shape with interpolated values.
    """
    if OrdinaryKriging is not None and len(values) >= 3:
        try:
            ok = OrdinaryKriging(lons, lats, values, variogram_model='linear')
            z, ss = ok.execute('points', grid_lon.flatten(), grid_lat.flatten())
            return np.array(z).reshape(grid_lat.shape)
        except Exception:
            pass

    # fallback: simple inverse-distance weighting
    pts = np.column_stack([lats, lons])
    grid_pts = np.column_stack([grid_lat.flatten(), grid_lon.flatten()])
    out = np.zeros(len(grid_pts), dtype=float)
    for i, gp in enumerate(grid_pts):
        dists = np.linalg.norm(pts - gp, axis=1)
        # avoid zero division
        if np.any(dists == 0):
            out[i] = values[np.argmin(dists)]
        else:
            w = 1.0 / (dists ** 2)
            out[i] = np.sum(w * values) / np.sum(w)
    return out.reshape(grid_lat.shape)

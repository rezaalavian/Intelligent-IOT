import numpy as np
from analysis.kriging import kriging_interpolate


def test_kriging_basic():
    lats = np.array([0.0, 0.0, 1.0])
    lons = np.array([0.0, 1.0, 0.0])
    vals = np.array([10.0, 20.0, 30.0])
    grid_lat, grid_lon = np.meshgrid(np.linspace(0, 1, 3), np.linspace(0, 1, 3))
    out = kriging_interpolate(lats, lons, vals, grid_lat, grid_lon)
    assert out.shape == grid_lat.shape
    assert np.isfinite(out).all()

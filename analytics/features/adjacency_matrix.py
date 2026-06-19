"""Utilities to compute wind-conditioned dynamic adjacency matrix A(t)."""

import math

import numpy as np

from .geo import haversine_m, north_east_offsets_m

DECAY_KM = 10.0


def compute_adjacency(locations: np.ndarray, wind_vectors: np.ndarray, k: int = 8,
                      decay_km: float = DECAY_KM) -> np.ndarray:
    """Compute a direction-aware adjacency matrix.

    locations: (N,2) array of (latitude, longitude) coordinates
    wind_vectors: (N,2) array of unit wind vectors for each station
    returns: (N,N) adjacency matrix
    """
    del k
    N = locations.shape[0]
    A = np.zeros((N, N), dtype=float)
    for i in range(N):
        lat_i, lon_i = float(locations[i, 0]), float(locations[i, 1])
        for j in range(N):
            if i == j:
                continue
            lat_j, lon_j = float(locations[j, 0]), float(locations[j, 1])
            north, east = north_east_offsets_m(lat_i, lon_i, lat_j, lon_j)
            norm = math.hypot(north, east)
            if norm == 0.0:
                continue
            alignment = np.clip((north * wind_vectors[i, 0] + east * wind_vectors[i, 1]) / norm, -1, 1)
            dist_km = haversine_m(lat_i, lon_i, lat_j, lon_j) / 1000.0
            A[i, j] = math.exp(-dist_km / decay_km) * (0.5 + 0.5 * float(alignment))
    return A


if __name__ == "__main__":
    # small offline smoke test
    locs = np.array([[0.0,0.0],[1.0,0.0],[0.0,1.0]])
    winds = np.array([[1.0,0.0],[1.0,0.0],[0.0,1.0]])
    print(compute_adjacency(locs, winds))

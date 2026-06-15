"""Utilities to compute wind-conditioned dynamic adjacency matrix A(t)."""

import numpy as np


def compute_adjacency(locations: np.ndarray, wind_vectors: np.ndarray, k: int = 8) -> np.ndarray:
    """Compute a direction-aware adjacency matrix.

    locations: (N,2) array of coordinates (lat, lon or x,y)
    wind_vectors: (N,2) array of unit wind vectors for each station
    returns: (N,N) adjacency matrix
    """
    del k
    N = locations.shape[0]
    A = np.zeros((N, N), dtype=float)
    # simple distance-based weights modulated by wind alignment
    for i in range(N):
        disp = locations - locations[i:i+1]
        dists = np.linalg.norm(disp, axis=1) + 1e-6
        dir_unit = disp / dists[:, None]
        alignment = np.clip(np.sum(dir_unit * wind_vectors[i:i+1], axis=1), -1, 1)
        weights = np.exp(-dists) * (0.5 + 0.5 * alignment)
        weights[i] = 0.0
        A[i] = weights
    return A


if __name__ == "__main__":
    # small offline smoke test
    locs = np.array([[0.0,0.0],[1.0,0.0],[0.0,1.0]])
    winds = np.array([[1.0,0.0],[1.0,0.0],[0.0,1.0]])
    print(compute_adjacency(locs, winds))

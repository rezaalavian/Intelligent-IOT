import math

import pytest

from analytics.flink_jobs.geo import haversine_m, north_east_offsets_m
from analytics.flink_jobs.feature_engineering import compute_dynamic_graph_edges


def test_haversine_one_degree_latitude():
    assert haversine_m(0.0, 0.0, 1.0, 0.0) == pytest.approx(111195.0, abs=50.0)


def test_haversine_zero_for_same_point():
    assert haversine_m(43.6, -79.4, 43.6, -79.4) == 0.0


def test_haversine_symmetric():
    a = haversine_m(43.65, -79.38, 43.70, -79.40)
    b = haversine_m(43.70, -79.40, 43.65, -79.38)
    assert a == pytest.approx(b, abs=1e-6)


def test_one_degree_longitude_shrinks_with_latitude():
    at_equator = haversine_m(0.0, 0.0, 0.0, 1.0)
    at_60_north = haversine_m(60.0, 0.0, 60.0, 1.0)
    assert at_60_north == pytest.approx(at_equator * math.cos(math.radians(60)), rel=0.01)


def test_north_east_offsets_axes():
    north, east = north_east_offsets_m(0.0, 0.0, 1.0, 0.0)
    assert north == pytest.approx(111195.0, abs=50.0)
    assert east == pytest.approx(0.0, abs=1e-6)
    north, east = north_east_offsets_m(0.0, 0.0, 0.0, 1.0)
    assert east == pytest.approx(111195.0, abs=50.0)
    assert north == pytest.approx(0.0, abs=1e-6)


def test_edge_distance_accounts_for_curvature():
    # Same degree-delta in lon vs lat at 60N: lon edge is ~half the distance,
    # so its 1/distance weight must be larger. Euclidean would tie them.
    coords = {"c": (60.0, 0.0), "north": (61.0, 0.0), "east": (60.0, 1.0)}
    _edge_index, edge_attr = compute_dynamic_graph_edges(0.0, 0.0, coords)
    weight_c_to_north = float(edge_attr[0])
    weight_c_to_east = float(edge_attr[1])
    assert weight_c_to_east > weight_c_to_north


def test_adjacency_accounts_for_curvature():
    import numpy as np
    from analytics.flink_jobs.adjacency_matrix import compute_adjacency
    locations = np.array([[60.0, 0.0], [61.0, 0.0], [60.0, 1.0]])
    winds = np.zeros((3, 2))
    adjacency = compute_adjacency(locations, winds)
    assert adjacency[0, 2] > adjacency[0, 1]

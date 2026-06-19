"""Great-circle geography helpers for station distance and bearing."""

import math

EARTH_RADIUS_M = 6_371_000.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two lat/lon points."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    return 2.0 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def north_east_offsets_m(lat1: float, lon1: float, lat2: float, lon2: float) -> tuple[float, float]:
    """Local (north, east) displacement in metres from point 1 to point 2."""
    mean_lat = math.radians((lat1 + lat2) / 2.0)
    north = EARTH_RADIUS_M * math.radians(lat2 - lat1)
    east = EARTH_RADIUS_M * math.radians(lon2 - lon1) * math.cos(mean_lat)
    return north, east

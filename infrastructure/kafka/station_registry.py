from collections import namedtuple

Station = namedtuple("Station", "id name lat lon role")

STATIONS = {
    7570:    Station(7570,    "Toronto Downtown", 43.64543,  -79.38908,  "target"),
    1274950: Station(1274950, "Toronto West",     43.709444, -79.5435,   "neighbor"),
    1274949: Station(1274949, "Toronto North",    43.78043,  -79.467397, "neighbor"),
    1210341: Station(1210341, "Toronto East",     43.7453,   -79.2703,   "neighbor"),
}


def target_id() -> int:
    return next(s.id for s in STATIONS.values() if s.role == "target")


def neighbor_ids() -> list[int]:
    return [s.id for s in STATIONS.values() if s.role == "neighbor"]


def coords(station_id: int) -> tuple[float, float]:
    s = STATIONS[station_id]
    return (s.lat, s.lon)

import math

from .geo import haversine_m, north_east_offsets_m


def diffusion_features(t_lat: float, t_lon: float, wind_u: float, wind_v: float,
                       neighbors: list[dict]) -> dict:
    speed = math.hypot(wind_u, wind_v)
    weighted_sum = 0.0
    weight_total = 0.0
    potentials: list[float] = []
    alignments: list[float] = []
    for n in neighbors:
        pm = n.get("pm25")
        if pm is None:
            continue
        dist = haversine_m(n["lat"], n["lon"], t_lat, t_lon)
        north, east = north_east_offsets_m(n["lat"], n["lon"], t_lat, t_lon)
        norm = math.hypot(north, east)
        if dist == 0.0 or norm == 0.0:
            continue
        alignment = (north * wind_u + east * wind_v) / (norm * speed) if speed > 0 else 0.0
        base = 1.0 / (dist + 1e-5)
        weight = base * (1.0 + alignment) if alignment > 0 else base * math.exp(alignment)
        weighted_sum += weight * float(pm)
        weight_total += weight
        potentials.append(speed * alignment)
        alignments.append(alignment)
    if weight_total == 0.0:
        return {"upwind_pm25": 0.0, "transport_potential": 0.0, "wind_alignment": 0.0}
    return {
        "upwind_pm25": weighted_sum / weight_total,
        "transport_potential": max(potentials),
        "wind_alignment": sum(alignments) / len(alignments),
    }

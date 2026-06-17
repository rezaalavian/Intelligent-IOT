import math

from analytics.flink_jobs.geo import haversine_m, north_east_offsets_m


def wind_weighted_estimate(t_lat, t_lon, wind_u, wind_v, neighbors):
    speed = math.hypot(wind_u, wind_v)
    weighted_sum = 0.0
    weight_total = 0.0
    for n in neighbors:
        pm = n.get("pm25")
        if pm is None:
            continue
        dist = haversine_m(n["lat"], n["lon"], t_lat, t_lon)
        north, east = north_east_offsets_m(n["lat"], n["lon"], t_lat, t_lon)
        norm = math.hypot(north, east)
        if dist == 0.0 or norm == 0.0:
            continue
        align = (north * wind_u + east * wind_v) / (norm * speed) if speed > 0 else 0.0
        base = 1.0 / (dist + 1e-5)
        weight = base * (1.0 + align) if align > 0 else base * math.exp(align)
        weighted_sum += weight * float(pm)
        weight_total += weight
    if weight_total == 0.0:
        return None
    return weighted_sum / weight_total


def temporal_fallback(history_values):
    for value in reversed(list(history_values)):
        if value is not None:
            return float(value)
    return None


def recover(t_lat, t_lon, wind_u, wind_v, neighbors, history_values, gap_hours, threshold_hours=3):
    if gap_hours <= threshold_hours:
        spatial = wind_weighted_estimate(t_lat, t_lon, wind_u, wind_v, neighbors)
        if spatial is not None:
            return spatial, "spatial"
    temporal = temporal_fallback(history_values)
    if temporal is not None:
        return temporal, "temporal"
    return None, "none"

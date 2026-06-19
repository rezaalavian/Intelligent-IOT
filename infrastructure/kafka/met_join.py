from analytics.features.geo import haversine_m


def nearest_met(lat: float, lon: float, met_records: list[dict]) -> dict | None:
    best = None
    best_dist = float("inf")
    for rec in met_records:
        rlat = rec.get("latitude")
        rlon = rec.get("longitude")
        if rlat is None or rlon is None:
            continue
        d = haversine_m(lat, lon, float(rlat), float(rlon))
        if d < best_dist:
            best_dist = d
            best = rec
    return best

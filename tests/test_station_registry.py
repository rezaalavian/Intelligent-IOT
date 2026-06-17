from infrastructure.kafka import station_registry as reg


def test_target_is_downtown():
    assert reg.target_id() == 7570


def test_two_neighbors():
    assert sorted(reg.neighbor_ids()) == [1274949, 1274950]


def test_coords_lookup():
    lat, lon = reg.coords(1274950)
    assert (round(lat, 5), round(lon, 5)) == (43.70944, -79.5435)

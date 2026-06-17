from infrastructure.kafka.rolling_buffer import RollingBuffer


def test_append_returns_history_oldest_first():
    buf = RollingBuffer(lookback=3)
    buf.append("s", {"n": 1})
    buf.append("s", {"n": 2})
    hist = buf.append("s", {"n": 3})
    assert [h["n"] for h in hist] == [1, 2, 3]


def test_caps_at_lookback():
    buf = RollingBuffer(lookback=2)
    buf.append("s", {"n": 1})
    buf.append("s", {"n": 2})
    hist = buf.append("s", {"n": 3})
    assert [h["n"] for h in hist] == [2, 3]


def test_per_station_isolation():
    buf = RollingBuffer(lookback=3)
    buf.append("a", {"n": 1})
    buf.append("b", {"n": 9})
    assert [h["n"] for h in buf.history("a")] == [1]
    assert [h["n"] for h in buf.history("b")] == [9]

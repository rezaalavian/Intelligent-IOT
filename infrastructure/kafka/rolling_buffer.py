from collections import deque


class RollingBuffer:
    def __init__(self, lookback: int = 24):
        self._lookback = lookback
        self._buffers: dict[str, deque] = {}

    def append(self, station_id: str, item: dict) -> list[dict]:
        buf = self._buffers.get(station_id)
        if buf is None:
            buf = deque(maxlen=self._lookback)
            self._buffers[station_id] = buf
        buf.append(item)
        return list(buf)

    def history(self, station_id: str) -> list[dict]:
        return list(self._buffers.get(station_id, ()))

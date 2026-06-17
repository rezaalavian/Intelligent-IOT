from datetime import datetime, timezone

from infrastructure.kafka.producers import run_ingestion as ri
from infrastructure.kafka.data_sources import environment_canada as ec


def test_build_sources_skips_openaq_without_key(monkeypatch):
    monkeypatch.delenv("OPENAQ_API_KEY", raising=False)
    import importlib, infrastructure.kafka.config as c
    importlib.reload(c)
    cfg = c.load_config()
    enabled = ri.enabled_sources(cfg)
    assert "openaq" not in enabled
    assert "envcanada" in enabled


def test_run_once_polls_and_produces():
    produced = []

    class P:
        def produce(self, topic, record):
            produced.append((topic, record))
        def flush(self, *a):
            pass

    sources = {
        "envcanada": lambda: [{"station_id": "swob-1", "datetime_utc": "t"}],
    }
    ri.run_once(sources, producer=P(), raw_topic=lambda s: f"aq.{s}.raw")
    assert produced == [("aq.envcanada.raw", {"station_id": "swob-1", "datetime_utc": "t"})]


def test_recent_window_is_rfc3339_interval():
    w = ec.recent_window(datetime(2026, 6, 16, 22, 30, 0, tzinfo=timezone.utc), hours=2)
    assert w == "2026-06-16T20:30:00Z/2026-06-16T22:30:00Z"


def test_envcanada_source_uses_bounded_window(monkeypatch):
    captured = {}

    def fake_poll(bbox, datetime_window=None, **kw):
        captured["bbox"] = bbox
        captured["window"] = datetime_window
        return []

    monkeypatch.setattr(ec, "poll", fake_poll)
    import importlib, infrastructure.kafka.config as c
    importlib.reload(c)
    cfg = c.load_config()
    enabled = ri.enabled_sources(cfg)
    enabled["envcanada"]()
    assert captured["window"] is not None
    assert "/" in captured["window"]

from infrastructure.kafka.producers import run_ingestion as ri


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

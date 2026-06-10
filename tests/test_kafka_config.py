import importlib


def test_defaults(monkeypatch):
    for var in ["KAFKA_BOOTSTRAP", "SCHEMA_REGISTRY_URL", "OPENAQ_API_KEY"]:
        monkeypatch.delenv(var, raising=False)
    import infrastructure.kafka.config as cfg
    importlib.reload(cfg)
    c = cfg.load_config()
    assert c.bootstrap_servers == "localhost:9092"
    assert c.schema_registry_url == "http://localhost:8081"
    assert c.topics["measurements"] == "aq.measurements"
    assert c.topics["deadletter"] == "aq.deadletter"
    assert c.raw_topic("openaq") == "aq.openaq.raw"
    assert c.partitions == 3
    assert c.openaq_api_key is None


def test_env_override(monkeypatch):
    monkeypatch.setenv("KAFKA_BOOTSTRAP", "broker:1234")
    monkeypatch.setenv("OPENAQ_API_KEY", "abc")
    import infrastructure.kafka.config as cfg
    importlib.reload(cfg)
    c = cfg.load_config()
    assert c.bootstrap_servers == "broker:1234"
    assert c.openaq_api_key == "abc"


def test_partitions_env_override(monkeypatch):
    monkeypatch.setenv("KAFKA_PARTITIONS", "5")
    import infrastructure.kafka.config as cfg
    importlib.reload(cfg)
    assert cfg.load_config().partitions == 5


def test_location_ids_parsing(monkeypatch):
    monkeypatch.setenv("OPENAQ_LOCATION_IDS", "100,200,300")
    import infrastructure.kafka.config as cfg
    importlib.reload(cfg)
    assert cfg.load_config().openaq_location_ids == (100, 200, 300)

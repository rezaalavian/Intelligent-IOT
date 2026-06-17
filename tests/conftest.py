import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: requires a running Kafka/Schema Registry stack")


@pytest.fixture(autouse=True)
def _hermetic_dotenv(monkeypatch):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False, raising=False)

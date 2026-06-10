def pytest_configure(config):
    config.addinivalue_line("markers", "integration: requires a running Kafka/Schema Registry stack")

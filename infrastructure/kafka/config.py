import os
from dataclasses import dataclass
from types import MappingProxyType

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

SOURCES: tuple[str, ...] = ("openaq", "envcanada", "iqair")


@dataclass(frozen=True)
class KafkaConfig:
    bootstrap_servers: str
    schema_registry_url: str
    partitions: int
    topics: dict[str, str]
    poll_intervals: dict[str, int]
    group_ids: dict[str, str]
    openaq_api_key: str | None
    openaq_location_ids: tuple[int, ...]
    envcanada_bbox: str
    sink_dir: str

    def raw_topic(self, source: str) -> str:
        return self.topics[f"{source}_raw"]


def load_config() -> KafkaConfig:
    topics = {
        "measurements": os.getenv("TOPIC_MEASUREMENTS", "aq.measurements"),
        "deadletter": os.getenv("TOPIC_DEADLETTER", "aq.deadletter"),
        "features": os.getenv("TOPIC_FEATURES", "aq.features"),
        "predictions": os.getenv("TOPIC_PREDICTIONS", "aq.predictions"),
        "alerts": os.getenv("TOPIC_ALERTS", "aq.alerts"),
    }
    for s in SOURCES:
        topics[f"{s}_raw"] = os.getenv(f"TOPIC_{s.upper()}_RAW", f"aq.{s}.raw")
    return KafkaConfig(
        bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP", "localhost:9092"),
        schema_registry_url=os.getenv("SCHEMA_REGISTRY_URL", "http://localhost:8081"),
        partitions=int(os.getenv("KAFKA_PARTITIONS", "3")),
        topics=MappingProxyType(topics),
        poll_intervals=MappingProxyType({s: int(os.getenv(f"POLL_INTERVAL_{s.upper()}", "300")) for s in SOURCES}),
        group_ids=MappingProxyType({
            "normalizer": os.getenv("GROUP_NORMALIZER", "aq-normalizer"),
            "sink": os.getenv("GROUP_SINK", "aq-sink"),
            "features": os.getenv("GROUP_FEATURES", "aq-features"),
            "inference": os.getenv("GROUP_INFERENCE", "aq-inference"),
            "alerts": os.getenv("GROUP_ALERTS", "aq-alerts"),
            "livestate": os.getenv("GROUP_LIVESTATE", "aq-livestate"),
        }),
        openaq_api_key=os.getenv("OPENAQ_API_KEY"),
        openaq_location_ids=tuple(
            int(x) for x in os.getenv("OPENAQ_LOCATION_IDS", "7570,1274950,1274949").split(",") if x.strip()
        ),
        envcanada_bbox=os.getenv("ENVCANADA_BBOX", "-80.0,43.0,-78.0,44.5"),
        sink_dir=os.getenv("SINK_DIR", "data/stream/measurements"),
    )

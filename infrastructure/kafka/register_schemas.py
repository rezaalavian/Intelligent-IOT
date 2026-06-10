from __future__ import annotations

import logging

from confluent_kafka.schema_registry import SchemaRegistryClient, Schema

from .config import load_config
from .serialization import schema_str

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# subject (<topic>-value) -> schema file
SUBJECTS = {
    "openaq": ("openaq_raw.avsc",),
    "envcanada": ("envcanada_raw.avsc",),
    "iqair": ("iqair_raw.avsc",),
}


def main() -> None:
    cfg = load_config()
    client = SchemaRegistryClient({"url": cfg.schema_registry_url})

    def register(subject: str, schema_file: str):
        schema = Schema(schema_str(schema_file), schema_type="AVRO")
        sid = client.register_schema(subject, schema)
        client.set_compatibility(subject_name=subject, level="BACKWARD")
        log.info("registered %s -> id=%s", subject, sid)

    for source, (schema_file,) in SUBJECTS.items():
        register(f"{cfg.raw_topic(source)}-value", schema_file)
    register(f"{cfg.topics['measurements']}-value", "measurement.avsc")


if __name__ == "__main__":
    main()

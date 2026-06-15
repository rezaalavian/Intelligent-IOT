import logging
from confluent_kafka import KafkaException, KafkaError
from confluent_kafka.admin import AdminClient, NewTopic
from .config import load_config, SOURCES

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def main() -> None:
    cfg = load_config()
    admin = AdminClient({"bootstrap.servers": cfg.bootstrap_servers})
    names = [cfg.raw_topic(s) for s in SOURCES] + [
        cfg.topics["measurements"], cfg.topics["deadletter"]
    ]
    topics = [NewTopic(n, num_partitions=cfg.partitions, replication_factor=1) for n in names]
    for name, fut in admin.create_topics(topics).items():
        try:
            fut.result()
            log.info("created topic %s", name)
        except KafkaException as exc:
            if exc.args[0].code() == KafkaError.TOPIC_ALREADY_EXISTS:
                log.info("topic %s already exists, skipping", name)
            else:
                log.error("failed to create topic %s: %s", name, exc)
                raise


if __name__ == "__main__":
    main()

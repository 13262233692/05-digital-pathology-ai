import os
import yaml
from pathlib import Path
from celery import Celery
from loguru import logger


def load_config():
    config_path = Path(__file__).parent.parent.parent / "configs" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


config = load_config()
celery_config = config.get("celery", {})

app = Celery(
    "pathology_sr",
    broker=celery_config.get("broker_url", "redis://localhost:6379/0"),
    backend=celery_config.get("result_backend", "redis://localhost:6379/0"),
    include=[
        "src.celery_tasks.tasks",
    ]
)

app.conf.update(
    task_serializer=celery_config.get("task_serializer", "json"),
    result_serializer=celery_config.get("result_serializer", "json"),
    accept_content=celery_config.get("accept_content", ["json"]),
    timezone=celery_config.get("timezone", "Asia/Shanghai"),
    enable_utc=True,
    task_acks_late=celery_config.get("task_acks_late", True),
    worker_prefetch_multiplier=celery_config.get("worker_prefetch_multiplier", 1),
    task_soft_time_limit=celery_config.get("task_soft_time_limit", 3600),
    task_time_limit=celery_config.get("task_time_limit", 7200),
    worker_max_tasks_per_child=100,
    worker_max_memory_per_child=8000000,
)

logger.info("Celery app initialized")

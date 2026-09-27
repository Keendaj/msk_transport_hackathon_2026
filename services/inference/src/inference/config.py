"""Настройки inference из переменных окружения с префиксом ``INFERENCE_`` и файла ``.env``."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Настройки сервиса, имя переменной — ``INFERENCE_`` плюс имя поля в верхнем регистре.

    Attributes:
        model_path: Файл весов ``DelayPredictorGRU``. Без него работает заглушка с прогнозом 0.
        schedule_path: CSV планового расписания. Без него и ``units_path`` нет целевых
            остановок и прогнозов.
        units_path: CSV соответствия ``unit_id`` → ``tr_id``.
        predict_every_s: Прогноз по одному ТС строится не чаще раза в столько секунд
            по времени устройства.
        kafka_bootstrap_servers: Адреса брокеров Kafka через запятую.
        kafka_telemetry_topic: Топик с пакетами телеметрии.
        kafka_group_id: Группа потребителей Kafka.
        database_url: DSN PostgreSQL.
        log_level: Уровень логирования: ``DEBUG``, ``INFO``, ``WARNING``...
    """

    model_config = SettingsConfigDict(env_prefix="INFERENCE_", env_file=".env", extra="ignore")

    model_path: Path | None = None
    schedule_path: Path | None = None
    units_path: Path | None = None
    predict_every_s: float = 15.0
    kafka_bootstrap_servers: str = "localhost:9094"
    kafka_telemetry_topic: str = "telemetry"
    kafka_group_id: str = "inference"
    database_url: str = "postgresql://transport:transport@localhost:5433/transport"
    log_level: str = "INFO"


settings = Settings()

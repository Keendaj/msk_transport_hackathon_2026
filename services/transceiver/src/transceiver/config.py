"""Настройки transceiver из переменных окружения с префиксом ``TRANSCEIVER_`` и файла ``.env``."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Настройки сервиса, имя переменной — ``TRANSCEIVER_`` плюс имя поля в верхнем регистре.

    Attributes:
        ndtp_host: Адрес, на котором TCP-сервер NDTP ждёт трекеры.
        ndtp_port: Порт TCP-сервера NDTP.
        ndtp_send_ack: Отвечать ли трекеру ``NPH_RESULT`` на кадры с флагом запроса.
        kafka_bootstrap_servers: Адреса брокеров Kafka через запятую.
        kafka_telemetry_topic: Топик, куда публикуются пакеты телеметрии.
        log_level: Уровень логирования: ``DEBUG``, ``INFO``, ``WARNING``...
    """

    model_config = SettingsConfigDict(env_prefix="TRANSCEIVER_", env_file=".env", extra="ignore")

    ndtp_host: str = "localhost"
    ndtp_port: int = 9000
    ndtp_send_ack: bool = True
    kafka_bootstrap_servers: str = "localhost:9094"
    kafka_telemetry_topic: str = "telemetry"
    log_level: str = "INFO"


settings = Settings()

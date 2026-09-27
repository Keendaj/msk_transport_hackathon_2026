"""Публикация пакетов телеметрии в Kafka."""

import json
from typing import Any

from aiokafka import AIOKafkaProducer
from aiokafka.errors import KafkaError


class PublishError(Exception):
    """Kafka не приняла пакет."""


class TelemetryPublisher:
    """Отправляет пакеты телеметрии в топик Kafka в виде JSON.

    Ключ сообщения — ``unit_id``, поэтому пакеты одного трекера попадают в одну партицию
    и читаются в порядке отправки.

    Args:
        producer: Запущенный продюсер Kafka.
        topic: Топик для пакетов.
    """

    def __init__(self, producer: AIOKafkaProducer, topic: str) -> None:
        self._producer = producer
        self._topic = topic

    async def publish(self, telemetry: dict[str, Any]) -> None:
        """Публикует пакет и ждёт подтверждения брокера.

        Args:
            telemetry: Пакет из ``NdtpServer.build_telemetry``.

        Raises:
            PublishError: Kafka не приняла пакет.
        """
        key = str(telemetry["unit_id"]).encode()
        value = json.dumps(telemetry).encode()
        try:
            await self._producer.send_and_wait(self._topic, value, key=key)
        except KafkaError as e:
            raise PublishError(f"Kafka rejected telemetry: {e!r}") from e

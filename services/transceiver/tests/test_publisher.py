import json

import pytest
from aiokafka.errors import KafkaConnectionError

from transceiver.publisher import PublishError, TelemetryPublisher

TELEMETRY = {
    "unit_id": 42,
    "request_id": 1,
    "received_at": "2026-09-26T12:00:00+00:00",
    "cells": [],
}


class FakeProducer:
    def __init__(self, error: Exception | None = None) -> None:
        self.sent: list[tuple[str, bytes, bytes]] = []
        self._error = error

    async def send_and_wait(self, topic: str, value: bytes, key: bytes) -> None:
        if self._error:
            raise self._error
        self.sent.append((topic, value, key))


async def test_publish_uses_unit_id_as_key() -> None:
    producer = FakeProducer()
    await TelemetryPublisher(producer, "telemetry").publish(TELEMETRY)  # type: ignore[arg-type]

    [(topic, value, key)] = producer.sent
    assert topic == "telemetry"
    assert key == b"42"
    assert json.loads(value) == TELEMETRY


async def test_publish_raises_on_kafka_error() -> None:
    producer = FakeProducer(error=KafkaConnectionError())
    publisher = TelemetryPublisher(producer, "telemetry")  # type: ignore[arg-type]
    with pytest.raises(PublishError, match="KafkaConnectionError"):
        await publisher.publish(TELEMETRY)

from unittest.mock import NonCallableMagicMock, create_autospec

import pytest
from aiokafka import AIOKafkaProducer
from fastapi.testclient import TestClient

from transceiver import main


@pytest.fixture
def producer(monkeypatch: pytest.MonkeyPatch) -> NonCallableMagicMock:
    producer = create_autospec(AIOKafkaProducer, instance=True)
    monkeypatch.setattr(main, "AIOKafkaProducer", lambda **kwargs: producer)
    monkeypatch.setattr(main.settings, "ndtp_host", "127.0.0.1")
    monkeypatch.setattr(main.settings, "ndtp_port", 0)
    return producer


def test_lifespan_starts_and_stops_kafka_producer(producer: NonCallableMagicMock) -> None:
    with TestClient(main.app):
        producer.start.assert_awaited_once()
        producer.stop.assert_not_awaited()
    producer.stop.assert_awaited_once()


def test_health(producer: NonCallableMagicMock) -> None:
    with TestClient(main.app) as client:
        assert client.get("/health").json() == {"status": "ok"}


def test_units(producer: NonCallableMagicMock) -> None:
    with TestClient(main.app) as client:
        assert client.get("/units").json() == {}
        missing = client.get("/units/7")
        assert missing.status_code == 404
        assert missing.json() == {"detail": "No packets from unitId=7 yet"}

        client.app.state.ndtp.last[7] = {"unit_id": 7}  # type: ignore[attr-defined]
        assert client.get("/units").json() == {"7": {"unit_id": 7}}
        assert client.get("/units/7").json() == {"unit_id": 7}

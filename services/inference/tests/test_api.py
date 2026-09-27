import asyncio
import logging
import signal
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from inference import main
from inference.features import FEATURES, SEQ_LEN, TrackPoint
from inference.main import app

T0 = datetime(2026, 1, 6, 10, 0, tzinfo=UTC).timestamp()
SHUTDOWN = main._shutdown  # настоящая, в остальных тестах её подменяет фикстура


class FakeKafkaConsumer:
    stops_immediately = False
    error: Exception | None = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    def __aiter__(self) -> FakeKafkaConsumer:
        return self

    async def __anext__(self) -> Any:
        if self.error:
            raise self.error
        if self.stops_immediately:
            raise StopAsyncIteration
        await asyncio.Event().wait()


class FakePool:
    async def close(self) -> None:
        pass


async def fake_create_pool(*args: Any, **kwargs: Any) -> FakePool:
    return FakePool()


@pytest.fixture(autouse=True)
def fake_kafka(monkeypatch: pytest.MonkeyPatch) -> Iterator[type[FakeKafkaConsumer]]:
    FakeKafkaConsumer.stops_immediately = False
    FakeKafkaConsumer.error = None
    monkeypatch.setattr(main, "AIOKafkaConsumer", FakeKafkaConsumer)
    monkeypatch.setattr(main, "create_pool", fake_create_pool)
    yield FakeKafkaConsumer


@pytest.fixture(autouse=True)
def shutdowns(monkeypatch: pytest.MonkeyPatch) -> list[None]:
    """Вызовы остановки сервиса вместо настоящего SIGTERM, который завершил бы pytest."""
    calls: list[None] = []
    monkeypatch.setattr(main, "_shutdown", lambda: calls.append(None))
    return calls


def test_health(shutdowns: list[None]) -> None:
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.json() == {"status": "ok", "model_version": "stub"}
    assert shutdowns == []  # отмена чтения при остановке приложения — не сбой


def test_health_fails_when_consumer_stopped(
    fake_kafka: type[FakeKafkaConsumer], shutdowns: list[None]
) -> None:
    fake_kafka.stops_immediately = True
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 503
    assert len(shutdowns) == 1


def test_consumer_error_is_logged_and_stops_service(
    fake_kafka: type[FakeKafkaConsumer],
    shutdowns: list[None],
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake_kafka.error = RuntimeError("broker gone")
    with TestClient(app) as client:
        assert client.get("/health").status_code == 503
    assert len(shutdowns) == 1
    [record] = [r for r in caplog.records if r.levelno == logging.CRITICAL]
    assert record.exc_info and "broker gone" in str(record.exc_info[1])


def test_shutdown_sends_sigterm(monkeypatch: pytest.MonkeyPatch) -> None:
    raised: list[int] = []
    monkeypatch.setattr(signal, "raise_signal", raised.append)
    SHUTDOWN()
    assert raised == [signal.SIGTERM]


def add_track(client: TestClient, unit_id: int) -> None:
    fleet = client.app.state.predictor.fleet  # type: ignore[attr-defined]
    for i in range(SEQ_LEN):
        fleet.add(TrackPoint(unit_id, T0 - 15 * i, 37.6, 55.75, 20.0, 90.0))


@pytest.fixture
def schedule(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Юнит 1 — ТС 700, у которого остановка 77 запланирована на 10:12 06.01.2026."""
    schedule, units = tmp_path / "schedule.csv", tmp_path / "units.csv"
    schedule.write_text(
        "tt_action_item_id,time_begin,order_date,tr_id,geom\n"
        "77,2026-01-06 10:12:00,2026-01-06,700,POINT (37.6 55.76)\n"
    )
    units.write_text("unit_id,tr_id\n1,700\n")
    monkeypatch.setattr(main.settings, "schedule_path", schedule)
    monkeypatch.setattr(main.settings, "units_path", units)


@pytest.mark.parametrize("path", ["/predict/42", "/features/42"])
def test_without_telemetry(path: str) -> None:
    with TestClient(app) as client:
        response = client.get(path)
    assert response.status_code == 404
    assert response.json() == {"detail": "No valid telemetry from unit 42"}


@pytest.mark.parametrize("path", ["/predict/1", "/features/1"])
def test_explains_missing_data(path: str) -> None:
    with TestClient(app) as client:
        add_track(client, 1)
        response = client.get(path)
    assert response.status_code == 404
    assert response.json() == {"detail": "unit 1 is not in the units list"}


@pytest.mark.usefixtures("schedule")
def test_predict() -> None:
    with TestClient(app) as client:
        add_track(client, 1)
        response = client.get("/predict/1")
    assert response.status_code == 200
    assert response.json() == {
        "unit_id": 1,
        "at": "2026-01-06T10:00:00Z",
        "score": 0.0,
        "model_version": "stub",
        "target_stop_id": 77,
        "target_planned_at": "2026-01-06T10:12:00Z",
        "cur_dev_s": 0.0,
    }


@pytest.mark.usefixtures("schedule")
def test_features() -> None:
    with TestClient(app) as client:
        add_track(client, 1)
        response = client.get("/features/1")
    assert response.status_code == 200
    body = response.json()
    assert (body["tr_id"], body["target"]["stop_id"]) == (700, 77)
    assert body["target"]["planned_at"] == "2026-01-06T10:12:00Z"
    assert body["cur_dev_s"] == 0.0
    assert body["columns"] == list(FEATURES)
    assert len(body["steps"]) == SEQ_LEN and len(body["steps"][0]) == len(FEATURES)


def test_predict_rejects_invalid_unit() -> None:
    with TestClient(app) as client:
        response = client.get("/predict/not-a-number")
    assert response.status_code == 422

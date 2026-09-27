import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import asyncpg
import pytest

from inference import consumer as consumer_module
from inference.consumer import TelemetryConsumer
from inference.features import MissingData
from inference.model import Model, Predictor
from inference.schedule import Schedule
from inference.schemas import Prediction, Telemetry
from inference.storage import Storage

T0 = 1767693600
TELEMETRY = {
    "unit_id": 42,
    "request_id": 7,
    "received_at": "2026-09-26T12:00:00+00:00",
    "cells": [{"type": 0, "number": 0, "name": "G6CellNav00", "fields": {"speedAvg": 40}}],
}
MESSAGE = json.dumps(TELEMETRY).encode()
PREDICTION = Prediction(
    unit_id=42,
    at=datetime.fromtimestamp(T0, UTC),
    score=12.5,
    model_version="v1",
    target_stop_id=77,
    target_planned_at=datetime.fromtimestamp(T0 + 720, UTC),
    cur_dev_s=30.0,
)


def message(t: float = T0) -> bytes:
    """Пакет с достоверной навигацией на момент t."""
    nav = {"timestamp": t, "lon": 37.6, "lat": 55.75, "speedAvg": 20, "course": 90}
    cells = [
        {"type": 0, "number": 0, "name": "G6CellNav00", "fields": nav | {"extraDopBit7": True}}
    ]
    return json.dumps(TELEMETRY | {"cells": cells}).encode()


class FakePredictor(Predictor):
    def __init__(self, result: Prediction | Exception = PREDICTION) -> None:
        super().__init__(Model(), Schedule({}, {}))
        self.result = result
        self.calls: list[tuple[int, datetime]] = []

    def predict(self, unit_id: int, at: datetime) -> Prediction:
        self.calls.append((unit_id, at))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class FakeStorage:
    def __init__(self, errors: list[Exception] | None = None) -> None:
        self.saved: list[tuple[Telemetry, Prediction | None]] = []
        self._errors = errors or []

    async def save(self, telemetry: Telemetry, prediction: Prediction | None) -> None:
        if self._errors:
            raise self._errors.pop(0)
        self.saved.append((telemetry, prediction))


def make_consumer(
    predictor: Predictor | None = None, storage: FakeStorage | None = None
) -> TelemetryConsumer:
    return TelemetryConsumer(
        predictor or FakePredictor(),
        None,  # type: ignore[arg-type]
        storage or FakeStorage(),  # type: ignore[arg-type]
    )


async def test_prediction_is_saved_with_telemetry() -> None:
    storage, predictor = FakeStorage(), FakePredictor()
    assert await make_consumer(predictor, storage).handle(message()) == PREDICTION
    assert predictor.calls == [(42, datetime.fromtimestamp(T0, UTC))]
    assert predictor.fleet.last_time(42) == T0
    [(telemetry, saved)] = storage.saved
    assert telemetry.unit_id == 42 and saved == PREDICTION


async def test_telemetry_without_navigation_is_saved_without_prediction() -> None:
    storage, predictor = FakeStorage(), FakePredictor()
    assert await make_consumer(predictor, storage).handle(MESSAGE) is None
    assert predictor.calls == []
    [(telemetry, saved)] = storage.saved
    assert telemetry.unit_id == 42 and saved is None


@pytest.mark.parametrize("error", [MissingData("5 of 60 points"), RuntimeError("model crashed")])
async def test_telemetry_is_saved_when_prediction_fails(error: Exception) -> None:
    storage = FakeStorage()
    assert await make_consumer(FakePredictor(error), storage).handle(message()) is None
    [(_, saved)] = storage.saved
    assert saved is None


async def test_predicts_once_per_interval() -> None:
    predictor = FakePredictor()
    consumer = make_consumer(predictor)
    for t in (T0, T0 + 5, T0 + 14, T0 + 15, T0 + 20):
        await consumer.handle(message(t))
    assert [at.timestamp() for _, at in predictor.calls] == [T0, T0 + 15]


async def test_failed_prediction_is_retried_on_next_packet() -> None:
    predictor = FakePredictor(MissingData("59 of 60 points"))
    consumer = make_consumer(predictor)
    for t in (T0, T0 + 1):
        await consumer.handle(message(t))
    assert len(predictor.calls) == 2


async def test_handle_skips_invalid_message() -> None:
    storage = FakeStorage()
    consumer = make_consumer(storage=storage)
    assert await consumer.handle(b"not json") is None
    assert await consumer.handle(b'{"unit_id": 1}') is None
    assert await consumer.handle(None) is None
    assert storage.saved == []


async def test_waits_for_database(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(consumer_module, "RETRY_DELAY", 0)
    storage = FakeStorage(errors=[ConnectionRefusedError(), asyncpg.CannotConnectNowError()])
    await make_consumer(storage=storage).handle(MESSAGE)
    assert len(storage.saved) == 1


async def test_skips_message_on_data_error() -> None:
    storage = FakeStorage(errors=[asyncpg.DataError("bad value")])
    await make_consumer(storage=storage).handle(MESSAGE)
    assert storage.saved == []


async def messages(count: int) -> AsyncIterator[SimpleNamespace]:
    for offset in range(count):
        yield SimpleNamespace(value=MESSAGE, partition=0, offset=offset)


async def test_run_handles_every_message() -> None:
    storage = AsyncMock(spec=Storage)
    await TelemetryConsumer(FakePredictor(), messages(2), storage).run()  # type: ignore[arg-type]
    assert storage.save.await_count == 2


async def test_run_skips_message_that_fails(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    storage = AsyncMock(spec=Storage)
    consumer = TelemetryConsumer(FakePredictor(), messages(2), storage)  # type: ignore[arg-type]
    handled: list[bytes | None] = []

    async def handle(value: bytes | None) -> None:
        handled.append(value)
        if len(handled) == 1:
            raise OverflowError("timestamp out of range")

    monkeypatch.setattr(consumer, "handle", handle)
    await consumer.run()
    assert len(handled) == 2
    assert "Message 0:0 skipped" in caplog.text

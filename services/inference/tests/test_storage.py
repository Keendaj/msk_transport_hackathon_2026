import json
import re
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from inference.schemas import Prediction, Telemetry
from inference.storage import Storage, prediction_row, telemetry_row

INIT_SQL = Path(__file__).parents[3] / "db" / "init.sql"
TARGET_AT = datetime(2026, 9, 26, 12, 12, tzinfo=UTC)
PREDICTION = Prediction(
    unit_id=500,
    at=datetime(2026, 9, 26, 12, 0, tzinfo=UTC),
    score=0.7,
    model_version="v1",
    target_stop_id=77,
    target_planned_at=TARGET_AT,
    cur_dev_s=30.0,
)


def cell(type_: int, name: str, fields: dict[str, object], number: int = 0) -> dict[str, object]:
    return {"type": type_, "number": number, "name": name, "fields": fields}


TELEMETRY = Telemetry.model_validate(
    {
        "unit_id": 500,
        "request_id": 9,
        "received_at": "2026-09-26T12:00:00.123456+00:00",
        "cells": [
            cell(
                0,
                "G6CellNav00",
                {
                    "timestamp": 1790424000,
                    "lat": 55.75,
                    "lon": 37.6,
                    "speedAvg": 41,
                    "course": 90,
                    "altitude": 150,
                    "nsat": 12,
                },
            ),
            cell(8, "G6CellUsi08", {"level_l": 120}),
            cell(8, "G6CellUsi08", {"level_l": 99}, number=1),
            cell(10, "G6CellCan10", {"speedTurnEngine": 1500, "tEngine": -5}),
            cell(16, "G6CellTermo16", {"temp": 21}),
        ],
    }
)


def test_telemetry_row() -> None:
    row = telemetry_row(TELEMETRY)
    assert row["unit_id"] == 500
    assert row["received_at"] == datetime(2026, 9, 26, 12, 0, 0, 123456, tzinfo=UTC)
    assert row["device_time"] == datetime.fromtimestamp(1790424000, UTC)
    assert (row["lat"], row["lon"], row["speed"], row["satellites"]) == (55.75, 37.6, 41, 12)
    assert row["fuel_level_l"] == 120
    assert (row["engine_rpm"], row["engine_temp"], row["temperature"]) == (1500, -5, 21)
    assert len(json.loads(row["cells"])) == 5


def test_telemetry_row_without_optional_cells() -> None:
    telemetry = TELEMETRY.model_copy(update={"cells": []})
    row = telemetry_row(telemetry)
    assert row["lat"] is None and row["fuel_level_l"] is None and row["device_time"] is None


def test_prediction_row_uses_telemetry_key() -> None:
    row = prediction_row(TELEMETRY, PREDICTION)
    assert (row["unit_id"], row["received_at"]) == (500, TELEMETRY.received_at)
    assert (row["score"], row["model_version"]) == (0.7, "v1")
    assert (row["target_stop_id"], row["target_planned_at"], row["cur_dev_s"]) == (
        77,
        TARGET_AT,
        30.0,
    )


def table_columns(table: str) -> list[str]:
    body = re.search(rf"CREATE TABLE {table} \((.*?)\n\);", INIT_SQL.read_text(), re.S)
    assert body, f"table {table} not found in {INIT_SQL}"
    return re.findall(r"^\s+([a-z_]+)\s+[a-z]", body.group(1), re.M)


def test_rows_match_database_schema() -> None:
    prediction_columns = [c for c in table_columns("predictions") if c != "created_at"]
    assert list(telemetry_row(TELEMETRY)) == table_columns("telemetry")
    assert list(prediction_row(TELEMETRY, PREDICTION)) == prediction_columns


def fake_pool() -> tuple[MagicMock, MagicMock]:
    connection = MagicMock()
    connection.execute = AsyncMock()
    pool = MagicMock()
    pool.acquire.return_value.__aenter__.return_value = connection
    return pool, connection


async def test_save_writes_both_tables_in_one_transaction() -> None:
    pool, connection = fake_pool()

    await Storage(pool).save(TELEMETRY, PREDICTION)

    connection.transaction.return_value.__aenter__.assert_awaited_once()
    (telemetry_sql, *telemetry_args), (prediction_sql, *prediction_args) = (
        call.args for call in connection.execute.await_args_list
    )
    assert telemetry_sql.startswith("INSERT INTO telemetry (unit_id, received_at, request_id,")
    assert telemetry_sql.endswith(
        "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, "
        "$13, $14, $15) ON CONFLICT DO NOTHING"
    )
    assert telemetry_args == list(telemetry_row(TELEMETRY).values())
    assert prediction_sql == (
        "INSERT INTO predictions (unit_id, received_at, score, model_version, "
        "target_stop_id, target_planned_at, cur_dev_s) "
        "VALUES ($1, $2, $3, $4, $5, $6, $7) ON CONFLICT DO NOTHING"
    )
    assert prediction_args == [500, TELEMETRY.received_at, 0.7, "v1", 77, TARGET_AT, 30.0]


async def test_save_without_prediction_writes_only_telemetry() -> None:
    pool, connection = fake_pool()

    await Storage(pool).save(TELEMETRY, None)

    [call] = connection.execute.await_args_list
    assert call.args[0].startswith("INSERT INTO telemetry ")

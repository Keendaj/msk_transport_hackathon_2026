"""Запись пакетов телеметрии и прогнозов в PostgreSQL, схема — в ``db/init.sql``.

``TRANSIENT_ERRORS`` — ошибки недоступной БД, после которых запись стоит повторить.
"""

import json
from datetime import UTC, datetime
from typing import Any

import asyncpg

from inference.schemas import Prediction, Telemetry

TRANSIENT_ERRORS = (
    OSError,
    TimeoutError,
    asyncpg.PostgresConnectionError,
    asyncpg.CannotConnectNowError,
)


class Storage:
    """Пишет пакеты в таблицу ``telemetry``, прогнозы — в ``predictions``.

    Args:
        pool: Пул соединений asyncpg.
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def save(self, telemetry: Telemetry, prediction: Prediction | None) -> None:
        """Пишет пакет и прогноз по нему в одной транзакции, повторы пропускаются."""
        async with self._pool.acquire() as connection, connection.transaction():
            await _insert(connection, "telemetry", telemetry_row(telemetry))
            if prediction is not None:
                await _insert(connection, "predictions", prediction_row(telemetry, prediction))


def telemetry_row(telemetry: Telemetry) -> dict[str, Any]:
    """Строка таблицы ``telemetry``.

    Основные поля ячеек навигации, CAN, датчиков топлива и температуры лежат в отдельных
    колонках, все ячейки целиком — в ``cells`` (jsonb). Поля отсутствующих ячеек — ``None``.
    """
    nav = _fields(telemetry, "G6CellNav00")
    can = _fields(telemetry, "G6CellCan10")
    timestamp = nav.get("timestamp")
    return {
        "unit_id": telemetry.unit_id,
        "received_at": telemetry.received_at,
        "request_id": telemetry.request_id,
        "device_time": datetime.fromtimestamp(timestamp, UTC) if timestamp is not None else None,
        "lat": nav.get("lat"),
        "lon": nav.get("lon"),
        "speed": nav.get("speedAvg"),
        "course": nav.get("course"),
        "altitude": nav.get("altitude"),
        "satellites": nav.get("nsat"),
        "fuel_level_l": _fields(telemetry, "G6CellUsi08").get("level_l"),
        "engine_rpm": can.get("speedTurnEngine"),
        "engine_temp": can.get("tEngine"),
        "temperature": _fields(telemetry, "G6CellTermo16").get("temp"),
        "cells": json.dumps([cell.model_dump() for cell in telemetry.cells]),
    }


def prediction_row(telemetry: Telemetry, prediction: Prediction) -> dict[str, Any]:
    """Строка таблицы ``predictions``, ключ ``(unit_id, received_at)`` — как у пакета."""
    return {
        "unit_id": telemetry.unit_id,
        "received_at": telemetry.received_at,
        "score": prediction.score,
        "model_version": prediction.model_version,
        "target_stop_id": prediction.target_stop_id,
        "target_planned_at": prediction.target_planned_at,
        "cur_dev_s": prediction.cur_dev_s,
    }


def _fields(telemetry: Telemetry, name: str) -> dict[str, Any]:
    """Поля ячейки ``name`` с номером 0, или пустой словарь, если её нет."""
    cell = telemetry.cell(name)
    return cell.fields if cell else {}


async def _insert(connection: asyncpg.Connection, table: str, row: dict[str, Any]) -> None:
    """Вставляет строку, если строки с таким ключом ещё нет."""
    columns = ", ".join(row)
    placeholders = ", ".join(f"${i}" for i in range(1, len(row) + 1))
    await connection.execute(
        f"INSERT INTO {table} ({columns}) VALUES ({placeholders}) ON CONFLICT DO NOTHING",
        *row.values(),
    )

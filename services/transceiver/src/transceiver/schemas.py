"""Схемы ответов API transceiver для Swagger.

Пакеты телеметрии API отдаёт как есть, в том виде, в каком они уходят в Kafka, поэтому
``Telemetry`` и ``Cell`` только описывают ответ в документации и ответы по ним не проверяются.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

NAV_FIELDS_EXAMPLE: dict[str, Any] = {
    "timestamp": 1767693600,
    "longitude": 376173210,
    "latitude": 557551234,
    "batVoltage": 200,
    "speedAvg": 40,
    "speedMax": 60,
    "course": 90,
    "track": 0,
    "altitude": 150,
    "nsat": 12,
    "pdop": 1,
    **{f"extraDopBit{i}": i >= 5 for i in range(8)},
    "lat": 55.7551234,
    "lon": 37.617321,
}


class Health(BaseModel):
    """Состояние сервиса."""

    status: str = Field(description="`ok`, если сервис отвечает", examples=["ok"])


class Error(BaseModel):
    """Ошибка запроса."""

    detail: str = Field(description="Причина ошибки", examples=["No packets from unitId=7 yet"])


class Cell(BaseModel):
    """Ячейка телематики из пакета NDTP."""

    type: int = Field(description="Тип ячейки NDTP", examples=[0])
    number: int = Field(
        description="Номер среди ячеек того же типа в пакете, например второго датчика топлива",
        examples=[0],
    )
    name: str = Field(description="Имя ячейки", examples=["G6CellNav00"])
    fields: dict[str, Any] = Field(
        description=(
            "Значения полей по именам, набор зависит от типа ячейки. В `G6CellNav00` байт "
            "`extraDop` разложен на `extraDopBit0…7`, а координаты в градусах добавлены в "
            "`lat` и `lon`. В `G6CellCan10` давление на осях собрано в список `pressureAxis`"
        ),
        examples=[NAV_FIELDS_EXAMPLE],
    )


class Telemetry(BaseModel):
    """Пакет телеметрии, собранный из пакета NDTP `NPH_SND_REALTIME`."""

    unit_id: int = Field(description="`unitId` трекера", examples=[1099984])
    request_id: int = Field(
        description="Номер запроса NPH, эмулятор сбрасывает его при каждой загрузке конфига",
        examples=[42],
    )
    received_at: datetime = Field(
        description="Время приёма пакета сервером, UTC", examples=["2026-01-06T10:00:00.25+00:00"]
    )
    cells: list[Cell] = Field(description="Ячейки телематики в порядке пакета")

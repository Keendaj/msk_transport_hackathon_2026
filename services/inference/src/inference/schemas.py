"""Схемы пакетов телеметрии из Kafka и ответов API."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class Cell(BaseModel):
    """Ячейка телематики из пакета NDTP."""

    type: int = Field(description="Тип ячейки NDTP", examples=[0])
    number: int = Field(description="Номер среди ячеек того же типа в пакете", examples=[0])
    name: str = Field(description="Имя ячейки", examples=["G6CellNav00"])
    fields: dict[str, Any] = Field(description="Значения полей по именам")


class Telemetry(BaseModel):
    """Пакет телеметрии от transceiver."""

    unit_id: int = Field(description="`unitId` трекера", examples=[1012706])
    request_id: int = Field(description="Номер запроса NPH", examples=[42])
    received_at: datetime = Field(description="Время приёма пакета transceiver, UTC")
    cells: list[Cell] = Field(description="Ячейки телематики в порядке пакета")

    def cell(self, name: str, number: int = 0) -> Cell | None:
        """Ячейка по имени и номеру среди ячеек того же типа, или ``None``, если её нет."""
        return next((c for c in self.cells if c.name == name and c.number == number), None)


class Prediction(BaseModel):
    """Прогноз задержки ТС на целевой остановке."""

    unit_id: int = Field(description="`unitId` трекера", examples=[1012706])
    at: datetime = Field(
        description="Момент прогноза: время устройства в последней точке трека, UTC",
        examples=["2026-01-06T10:00:00Z"],
    )
    score: float = Field(
        description="Прогноз задержки на целевой остановке, секунды: плюс — опоздание",
        examples=[95.4],
    )
    model_version: str = Field(
        description="Версия модели: имя файла весов или `stub` для заглушки", examples=["v1"]
    )
    target_stop_id: int = Field(
        description="`tt_action_item_id` целевой остановки из расписания", examples=[77]
    )
    target_planned_at: datetime = Field(
        description="Плановое время целевой остановки, в интервале (at + 10 мин, at + 15 мин]",
        examples=["2026-01-06T10:12:00Z"],
    )
    cur_dev_s: float = Field(
        description="Текущее отклонение от расписания на момент `at`, секунды: плюс — опоздание",
        examples=[60.0],
    )


class TargetStop(BaseModel):
    """Целевая остановка прогноза."""

    stop_id: int = Field(description="`tt_action_item_id` из расписания", examples=[77])
    planned_at: datetime = Field(
        description="Плановое время прибытия, UTC", examples=["2026-01-06T10:12:00Z"]
    )
    lon: float = Field(description="Долгота, градусы", examples=[37.6])
    lat: float = Field(description="Широта, градусы", examples=[55.76])


class Features(BaseModel):
    """Вход модели: последовательность признаков на момент прогноза."""

    unit_id: int = Field(description="`unitId` трекера", examples=[1012706])
    tr_id: int = Field(description="Номер ТС в расписании", examples=[132430])
    at: datetime = Field(
        description="Момент прогноза: время устройства в последней точке трека, UTC",
        examples=["2026-01-06T10:00:00Z"],
    )
    target: TargetStop = Field(description="Целевая остановка")
    cur_dev_s: float = Field(
        description="Текущее отклонение от расписания на момент `at`, секунды: плюс — опоздание",
        examples=[60.0],
    )
    columns: list[str] = Field(description="Имена признаков в порядке столбцов `steps`")
    steps: list[list[float]] = Field(
        description=(
            "60 последних точек трека за 20 минут от старых к новым, по 7 признаков в каждой. "
            "Признаки без нормализации: скорость в км/ч, расстояние до целевой остановки "
            "в метрах, курс в градусах, средняя скорость, число и стандартное отклонение "
            "скорости соседей"
        ),
        examples=[[[60.0, 23.0, 1520.4, 90.0, 21.5, 2.0, 3.5]]],
    )


class Health(BaseModel):
    """Состояние сервиса."""

    status: str = Field(description="`ok`, если чтение Kafka работает", examples=["ok"])
    model_version: str = Field(
        description="Версия загруженной модели: имя файла весов или `stub`", examples=["v1"]
    )


class Error(BaseModel):
    """Ошибка запроса."""

    detail: str = Field(description="Причина ошибки", examples=["No valid telemetry from unit 42"])

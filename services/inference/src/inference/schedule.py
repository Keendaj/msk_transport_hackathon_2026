"""Плановое расписание ТС и выбор целевой остановки.

Расписание читается из CSV с колонками ``tt_action_item_id``, ``tr_id``, ``time_begin``,
``order_date`` и ``geom`` (``POINT (lon lat)``), соответствие трекеров и ТС — из CSV
с колонками ``unit_id`` и ``tr_id``. В CSV расписания должен быть план на один
``order_date``: план за несколько дней склеится в один суточный шаблон с повторами остановок.
"""

import csv
import logging
import re
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

log = logging.getLogger(__name__)

# Целевая остановка — первая, чьё плановое время попадает в (T + 10 мин, T + 15 мин]
TARGET_FROM = timedelta(minutes=10)
TARGET_TO = timedelta(minutes=15)

POINT = re.compile(r"POINT\s*\(\s*(\S+)\s+(\S+)\s*\)", re.IGNORECASE)


@dataclass(frozen=True)
class StopVisit:
    """Остановка в наряде ТС без привязки к дате.

    Attributes:
        stop_id: ``tt_action_item_id`` из расписания.
        offset: Плановое время от полуночи дня наряда, после полуночи бывает больше суток.
        lon: Долгота остановки, градусы.
        lat: Широта остановки, градусы.
    """

    stop_id: int
    offset: timedelta
    lon: float
    lat: float


@dataclass(frozen=True)
class PlannedStop:
    """Остановка с плановым временем в конкретный день.

    Attributes:
        stop_id: ``tt_action_item_id`` из расписания.
        planned_at: Плановое время прибытия, UTC.
        lon: Долгота остановки, градусы.
        lat: Широта остановки, градусы.
    """

    stop_id: int
    planned_at: datetime
    lon: float
    lat: float


class Schedule:
    """Плановое расписание (без факта), повторяется каждый день.

    Наряды заходят за полночь, поэтому сутки расписания начинаются не в полночь, а за
    TARGET_TO до самой ранней остановки: момент T до этой границы относится к вчерашнему наряду.

    Args:
        visits: Остановки нарядов по ``tr_id``.
        units: ``tr_id`` по ``unit_id`` трекера.
    """

    def __init__(self, visits: dict[int, list[StopVisit]], units: dict[int, int]) -> None:
        self._visits = {tr_id: sorted(v, key=lambda s: s.offset) for tr_id, v in visits.items()}
        self._offsets = {tr_id: [s.offset for s in v] for tr_id, v in self._visits.items()}
        self._units = units
        first = min((o[0] for o in self._offsets.values() if o), default=TARGET_TO)
        self._day_start = first - TARGET_TO

    def tr_id(self, unit_id: int) -> int | None:
        """Номер ТС в расписании по ``unit_id`` трекера, или ``None``, если трекера нет."""
        return self._units.get(unit_id)

    def target(self, tr_id: int, at: datetime) -> PlannedStop | None:
        """Первая остановка, чьё плановое время попадает в (at + 10 мин, at + 15 мин].

        Returns:
            ``None``, если в этот интервал у ТС нет остановок, например в перерыв.
        """
        visits = self._visits.get(tr_id)
        if not visits:
            return None
        offsets = self._offsets[tr_id]
        midnight = self._midnight(at)
        # При равном плановом времени берётся остановка, что раньше в файле
        i = bisect_right(offsets, at + TARGET_FROM - midnight)
        if i == len(visits) or offsets[i] > at + TARGET_TO - midnight:
            return None
        return _planned(visits[i], midnight)

    def planned_stops(
        self, tr_id: int, at: datetime, before: timedelta, after: timedelta
    ) -> list[PlannedStop]:
        """Остановки наряда, к которому относится at, по плану в [at - before, at + after]."""
        visits = self._visits.get(tr_id, [])
        offsets = self._offsets.get(tr_id, [])
        midnight = self._midnight(at)
        start = bisect_left(offsets, at - before - midnight)
        end = bisect_right(offsets, at + after - midnight)
        return [_planned(v, midnight) for v in visits[start:end]]

    def _midnight(self, at: datetime) -> datetime:
        """Полночь UTC дня наряда, к которому относится ``at``."""
        day = (at.astimezone(UTC) - self._day_start).date()
        return datetime.combine(day, time(), UTC)


def _planned(visit: StopVisit, midnight: datetime) -> PlannedStop:
    """Остановка наряда с плановым временем в день с полуночью ``midnight``."""
    return PlannedStop(visit.stop_id, midnight + visit.offset, visit.lon, visit.lat)


def load_schedule(schedule_path: Path | None, units_path: Path | None) -> Schedule:
    """Загружает расписание и соответствие трекеров и ТС.

    Если хотя бы один путь не задан, расписание пустое и прогнозов не будет.
    """
    if schedule_path is None or units_path is None:
        log.warning("INFERENCE_SCHEDULE_PATH or INFERENCE_UNITS_PATH is not set, no target stops")
        return Schedule({}, {})
    visits = _read_visits(schedule_path)
    units = _read_units(units_path)
    log.info(f"Schedule: {sum(map(len, visits.values()))} stops of {len(visits)} vehicles")
    return Schedule(visits, units)


def _read_visits(path: Path) -> dict[int, list[StopVisit]]:
    """Остановки нарядов по ``tr_id``, строки без точки в ``geom`` пропускаются."""
    visits: dict[int, list[StopVisit]] = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            point = POINT.fullmatch(row["geom"].strip())
            if point is None:
                continue
            day = datetime.combine(date.fromisoformat(row["order_date"]), time(), UTC)
            planned = datetime.fromisoformat(row["time_begin"]).replace(tzinfo=UTC)
            visit = StopVisit(
                stop_id=int(row["tt_action_item_id"]),
                offset=planned - day,
                lon=float(point[1]),
                lat=float(point[2]),
            )
            visits.setdefault(int(row["tr_id"]), []).append(visit)
    return visits


def _read_units(path: Path) -> dict[int, int]:
    """``tr_id`` по ``unit_id`` трекера."""
    with path.open(encoding="utf-8", newline="") as f:
        return {int(row["unit_id"]): int(row["tr_id"]) for row in csv.DictReader(f)}

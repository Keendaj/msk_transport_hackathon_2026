"""Треки ТС и вход модели, как в ``extract_rnn_sequences`` из ноутбука обучения.

Вход модели — матрица ``SEQ_LEN`` × ``len(FEATURES)``: последние ``SEQ_LEN`` точек трека
за ``HISTORY_SECONDS`` до момента прогноза, от старых к новым, без нормализации.

Константы:
    SEQ_LEN: Число шагов последовательности.
    HISTORY_SECONDS: Точки для последовательности берутся за столько секунд до прогноза.
    BUCKET_SECONDS: Длина окна, в котором ищутся соседи, секунды.
    NEIGHBOR_RADIUS_M: Сосед — ТС не дальше этого расстояния, метры.
    NEIGHBOR_HEADING_DEG: Курс соседа отличается не больше чем на столько градусов.
    FEATURES: Имена признаков в порядке столбцов матрицы.
    KEEP_SECONDS: Сколько секунд точек хранит ``Fleet``: хватает и на последовательность,
        и на расчёт ``cur_dev_s``.
    MAX_AHEAD_SECONDS: Насколько время устройства может опережать время приёма пакета.
"""

import math
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import numpy.typing as npt

from inference.delay import LOOKBACK, PASS_EARLY, current_delay
from inference.geo import haversine_m
from inference.schedule import PlannedStop, Schedule
from inference.schemas import Telemetry

SEQ_LEN = 60
HISTORY_SECONDS = 20 * 60
BUCKET_SECONDS = 15
NEIGHBOR_RADIUS_M = 700.0
NEIGHBOR_HEADING_DEG = 30.0

FEATURES = (
    "cur_dev_s",
    "speed",
    "dist_to_stop",
    "heading",
    "neighbor_speed_mean",
    "neighbor_count",
    "neighbor_speed_std",
)

KEEP_SECONDS = max(HISTORY_SECONDS, (LOOKBACK + PASS_EARLY).total_seconds()) + 60
MAX_AHEAD_SECONDS = 60


@dataclass(frozen=True)
class TrackPoint:
    """Точка трека ТС из ячейки ``G6CellNav00``.

    Attributes:
        unit_id: ``unitId`` трекера.
        time: Время устройства, Unix-секунды.
        lon: Долгота, градусы.
        lat: Широта, градусы.
        speed: Средняя скорость, км/ч.
        heading: Курс, градусы в [0, 360).
    """

    unit_id: int
    time: float
    lon: float
    lat: float
    speed: float
    heading: float


@dataclass(frozen=True)
class Sequence:
    """Вход модели для одного ТС на момент прогноза.

    Attributes:
        unit_id: ``unitId`` трекера.
        tr_id: Номер ТС в расписании.
        at: Момент прогноза.
        target: Целевая остановка.
        cur_dev_s: Текущее отклонение от расписания, секунды.
        steps: Матрица ``SEQ_LEN`` × ``len(FEATURES)``, столбцы в порядке ``FEATURES``.
    """

    unit_id: int
    tr_id: int
    at: datetime
    target: PlannedStop
    cur_dev_s: float
    steps: npt.NDArray[np.float32]


class MissingData(Exception):
    """Данных для прогноза не хватает, текст объясняет, каких именно."""


def track_point(telemetry: Telemetry) -> TrackPoint | None:
    """Точка трека из пакета.

    Returns:
        ``None``, если в пакете нет ``G6CellNav00``, координаты недостоверны
        (``extraDopBit7`` сброшен), полей не хватает или время устройства опережает время
        приёма больше чем на ``MAX_AHEAD_SECONDS``: ``Fleet`` отбрасывает точки старше
        самой свежей, и одна точка из будущего вытеснила бы треки всех ТС.
    """
    nav = telemetry.cell("G6CellNav00")
    if nav is None or not nav.fields.get("extraDopBit7"):
        return None
    fields = nav.fields
    try:
        point = TrackPoint(
            unit_id=telemetry.unit_id,
            time=float(fields["timestamp"]),
            lon=float(fields["lon"]),
            lat=float(fields["lat"]),
            speed=float(fields["speedAvg"]),
            heading=float(fields["course"]) % 360.0,
        )
    except KeyError, TypeError, ValueError:
        return None
    if point.time > telemetry.received_at.timestamp() + MAX_AHEAD_SECONDS:
        return None
    return point


class Fleet:
    """Точки всех ТС за последние ``keep_seconds`` секунд.

    Точки хранятся по ТС в порядке времени, для истории, и по окнам ``BUCKET_SECONDS``,
    для поиска соседей. Точки старше ``keep_seconds`` от самой свежей отбрасываются.

    Args:
        keep_seconds: Сколько секунд точек хранить.
    """

    def __init__(self, keep_seconds: float = KEEP_SECONDS) -> None:
        self._keep = keep_seconds
        self._tracks: dict[int, list[TrackPoint]] = {}
        self._buckets: dict[int, list[TrackPoint]] = {}
        self._latest = -math.inf

    def add(self, point: TrackPoint) -> None:
        """Добавляет точку, опоздавшую вставляет по времени, слишком старую пропускает."""
        if point.time < self._latest - self._keep:
            return
        track = self._tracks.setdefault(point.unit_id, [])
        track.insert(bisect_right(track, point.time, key=_time), point)
        self._buckets.setdefault(_bucket(point.time), []).append(point)
        if point.time > self._latest:
            self._latest = point.time
            self._prune()

    def last_time(self, unit_id: int) -> float | None:
        """Время последней точки ТС, Unix-секунды, или ``None``, если точек нет."""
        track = self._tracks.get(unit_id)
        return track[-1].time if track else None

    def track(self, unit_id: int, start: float, end: float) -> list[TrackPoint]:
        """Точки ТС со временем в [start, end] по возрастанию времени."""
        track = self._tracks.get(unit_id, [])
        return track[bisect_left(track, start, key=_time) : bisect_right(track, end, key=_time)]

    def history(self, unit_id: int, at: float) -> list[TrackPoint]:
        """Последние ``SEQ_LEN`` точек ТС за ``HISTORY_SECONDS`` до ``at``.

        Raises:
            MissingData: Точек за это время меньше ``SEQ_LEN``.
        """
        history = self.track(unit_id, at - HISTORY_SECONDS, at)
        if len(history) < SEQ_LEN:
            raise MissingData(f"{len(history)} of {SEQ_LEN} points in the last 20 minutes")
        return history[-SEQ_LEN:]

    def neighbor_stats(self, point: TrackPoint) -> tuple[float, float, float]:
        """Скорость соседей: других ТС в том же окне ``BUCKET_SECONDS`` рядом и в ту же сторону.

        Сосед — точка не дальше ``NEIGHBOR_RADIUS_M`` с курсом, отличающимся не больше
        чем на ``NEIGHBOR_HEADING_DEG``.

        Returns:
            Средняя скорость соседей, их число и выборочное стандартное отклонение скорости.
            Нули, если соседей нет, отклонение 0 при одном соседе.
        """
        others = [
            p for p in self._buckets.get(_bucket(point.time), []) if p.unit_id != point.unit_id
        ]
        if not others:
            return 0.0, 0.0, 0.0
        lon, lat, speed, heading = np.array([(p.lon, p.lat, p.speed, p.heading) for p in others]).T
        diff = np.abs(heading - point.heading)
        diff = np.minimum(diff, 360.0 - diff)
        near = (haversine_m(point.lon, point.lat, lon, lat) <= NEIGHBOR_RADIUS_M) & (
            diff <= NEIGHBOR_HEADING_DEG
        )
        speeds = speed[near]
        if len(speeds) == 0:
            return 0.0, 0.0, 0.0
        std = float(np.std(speeds, ddof=1)) if len(speeds) > 1 else 0.0
        return float(speeds.mean()), float(len(speeds)), std

    def _prune(self) -> None:
        """Отбрасывает точки и окна старше ``keep_seconds`` от самой свежей точки."""
        cutoff = self._latest - self._keep
        for unit_id, track in list(self._tracks.items()):
            del track[: bisect_left(track, cutoff, key=_time)]
            if not track:
                del self._tracks[unit_id]
        for key in [k for k in self._buckets if (k + 1) * BUCKET_SECONDS <= cutoff]:
            del self._buckets[key]


def build_steps(
    fleet: Fleet, history: list[TrackPoint], stop_lon: float, stop_lat: float
) -> npt.NDArray[np.float32]:
    """Признаки по точкам истории, кроме ``cur_dev_s``.

    Args:
        fleet: Точки всех ТС, по ним считаются соседи.
        history: Точки ТС по возрастанию времени.
        stop_lon: Долгота целевой остановки, градусы.
        stop_lat: Широта целевой остановки, градусы.

    Returns:
        Матрица ``len(history)`` × ``len(FEATURES) − 1``, столбцы в порядке ``FEATURES[1:]``.
    """
    rows = [
        (
            p.speed,
            float(haversine_m(p.lon, p.lat, stop_lon, stop_lat)),
            p.heading,
            *fleet.neighbor_stats(p),
        )
        for p in history
    ]
    return np.array(rows, dtype=np.float32)


def build_sequence(fleet: Fleet, schedule: Schedule, unit_id: int, at: datetime) -> Sequence:
    """Вход модели для ТС на момент ``at``.

    Raises:
        MissingData: ТС нет в списке ``units``, у него нет целевой остановки по расписанию
            или в треке меньше ``SEQ_LEN`` точек за ``HISTORY_SECONDS``.
    """
    tr_id = schedule.tr_id(unit_id)
    if tr_id is None:
        raise MissingData(f"unit {unit_id} is not in the units list")
    target = schedule.target(tr_id, at)
    if target is None:
        raise MissingData(f"no stop of vehicle {tr_id} is planned 10-15 minutes after {at}")
    history = fleet.history(unit_id, at.timestamp())
    cur_dev_s = delay_at(fleet, schedule, unit_id, tr_id, at)
    steps = np.column_stack(
        [
            np.full(SEQ_LEN, cur_dev_s, dtype=np.float32),
            build_steps(fleet, history, target.lon, target.lat),
        ]
    )
    return Sequence(unit_id, tr_id, at, target, cur_dev_s, steps)


def delay_at(fleet: Fleet, schedule: Schedule, unit_id: int, tr_id: int, at: datetime) -> float:
    """``cur_dev_s`` ТС на момент ``at`` по треку и расписанию за ``LOOKBACK`` до него."""
    stops = schedule.planned_stops(tr_id, at, before=LOOKBACK, after=PASS_EARLY)
    track = fleet.track(unit_id, (at - LOOKBACK - PASS_EARLY).timestamp(), at.timestamp())
    times, lon, lat = (
        np.array([(p.time, p.lon, p.lat) for p in track], dtype=np.float64).reshape(-1, 3).T
    )
    return current_delay(times, lon, lat, stops, at)


def _bucket(t: float) -> int:
    """Номер окна ``BUCKET_SECONDS``, в которое попадает момент ``t``."""
    return math.floor(t / BUCKET_SECONDS)


def _time(point: TrackPoint) -> float:
    """Ключ сортировки точек по времени."""
    return point.time

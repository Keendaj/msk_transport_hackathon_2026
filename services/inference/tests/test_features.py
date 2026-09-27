import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from inference.features import (
    FEATURES,
    KEEP_SECONDS,
    MAX_AHEAD_SECONDS,
    SEQ_LEN,
    Fleet,
    MissingData,
    TrackPoint,
    build_sequence,
    track_point,
)
from inference.geo import haversine_m
from inference.schedule import load_schedule
from inference.schemas import Telemetry

T0 = datetime(2026, 1, 6, 10, 0, tzinfo=UTC).timestamp()
LON, LAT = 37.6, 55.75
METERS_PER_DEGREE_LAT = 6367000.0 * math.pi / 180


def point(unit_id: int = 1, t: float = T0, **fields: float) -> TrackPoint:
    values = {"lon": LON, "lat": LAT, "speed": 20.0, "heading": 90.0} | fields
    return TrackPoint(unit_id=unit_id, time=t, **values)


def north(meters: float) -> float:
    return LAT + meters / METERS_PER_DEGREE_LAT


def test_haversine() -> None:
    assert haversine_m(LON, LAT, LON, LAT + 1) == pytest.approx(METERS_PER_DEGREE_LAT)
    assert haversine_m(LON, LAT, np.array([LON, LON]), np.array([LAT, LAT + 1])) == pytest.approx(
        [0.0, METERS_PER_DEGREE_LAT]
    )


def telemetry(**nav: object) -> Telemetry:
    fields = {
        "timestamp": T0,
        "lon": LON,
        "lat": LAT,
        "speedAvg": 30,
        "course": 360,
        "extraDopBit7": True,
    } | nav
    return Telemetry.model_validate(
        {
            "unit_id": 5,
            "request_id": 1,
            "received_at": "2026-01-06T10:00:00+00:00",
            "cells": [{"type": 0, "number": 0, "name": "G6CellNav00", "fields": fields}],
        }
    )


def test_track_point_from_telemetry() -> None:
    assert track_point(telemetry()) == TrackPoint(5, T0, LON, LAT, 30.0, 0.0)


@pytest.mark.parametrize("nav", [{"extraDopBit7": False}, {"lat": None}, {"course": "x"}])
def test_track_point_skips_invalid_navigation(nav: dict[str, object]) -> None:
    assert track_point(telemetry(**nav)) is None


def test_track_point_without_navigation() -> None:
    empty = telemetry().model_copy(update={"cells": []})
    assert track_point(empty) is None


def test_track_point_rejects_time_ahead_of_reception() -> None:
    # Пакет принят в T0: точка из будущего вытеснила бы из Fleet треки всех ТС
    assert track_point(telemetry(timestamp=T0 + MAX_AHEAD_SECONDS)) is not None
    assert track_point(telemetry(timestamp=T0 + MAX_AHEAD_SECONDS + 1)) is None


def fill(fleet: Fleet, count: int, unit_id: int = 1, step: float = 15.0) -> None:
    for i in range(count):
        fleet.add(point(unit_id, T0 - step * i, speed=float(i)))


def test_history_takes_last_points_in_order() -> None:
    fleet = Fleet()
    fill(fleet, SEQ_LEN + 5, step=10)  # точки приходят от новых к старым
    history = fleet.history(1, T0)
    assert len(history) == SEQ_LEN
    assert [p.time for p in history] == sorted(p.time for p in history)
    assert history[-1].time == T0


def test_history_needs_enough_points_in_20_minutes() -> None:
    fleet = Fleet()
    fill(fleet, SEQ_LEN, step=20.5)  # 60 точек, но самая старая — раньше чем за 20 минут
    with pytest.raises(MissingData, match="59 of 60"):
        fleet.history(1, T0)


def test_history_ignores_points_after_moment() -> None:
    fleet = Fleet()
    fill(fleet, SEQ_LEN)
    fleet.add(point(1, T0 + 5))
    assert fleet.history(1, T0)[-1].time == T0


def test_points_with_same_time_are_kept() -> None:
    fleet = Fleet()
    fleet.add(point(1, T0 + 1, speed=1))
    fleet.add(point(1, T0 + 1, speed=2))
    fill(fleet, SEQ_LEN - 2, step=5)
    history = fleet.history(1, T0 + 1)
    assert [p.speed for p in history[-2:]] == [1.0, 2.0]


def test_old_points_are_dropped() -> None:
    fleet = Fleet()
    fleet.add(point(1, T0))
    fleet.add(point(2, T0 + 2 * KEEP_SECONDS))
    assert fleet.last_time(1) is None
    fleet.add(point(1, T0))  # опоздавшая точка старше окна
    assert fleet.last_time(1) is None
    assert fleet.neighbor_stats(point(3, T0)) == (0.0, 0.0, 0.0)


def test_neighbor_stats() -> None:
    fleet = Fleet()
    me = point(1, T0 + 1, heading=10)
    for p in [
        me,
        point(1, T0 + 2, speed=99),  # тот же автобус
        point(2, T0, lat=north(600), heading=350, speed=10),  # курс через 0°: разница 20°
        point(3, T0 + 14, speed=30, heading=40),  # разница ровно 30°
        point(4, T0 + 3, lat=north(800), heading=10, speed=99),  # дальше 700 м
        point(5, T0 + 3, heading=190, speed=99),  # встречный
        point(6, T0 + 15, heading=10, speed=99),  # следующее 15-секундное окно
    ]:
        fleet.add(p)
    mean, count, std = fleet.neighbor_stats(me)
    assert (mean, count) == (20.0, 2.0)
    assert std == pytest.approx(np.std([10, 30], ddof=1))


def test_single_neighbor_has_zero_std() -> None:
    fleet = Fleet()
    fleet.add(point(2, T0, speed=12))
    assert fleet.neighbor_stats(point(1, T0)) == (12.0, 1.0, 0.0)


def test_no_neighbors_among_others_in_bucket() -> None:
    fleet = Fleet()
    fleet.add(point(2, T0, lat=north(800)))  # то же окно, но дальше 700 м
    fleet.add(point(3, T0, heading=270))  # рядом, но встречный
    assert fleet.neighbor_stats(point(1, T0)) == (0.0, 0.0, 0.0)


def schedule_files(tmp_path: Path) -> tuple[Path, Path]:
    schedule = tmp_path / "schedule.csv"
    schedule.write_text(
        "tt_action_item_id,time_begin,order_date,tr_id,geom\n"
        f"77,2026-01-06 10:12:00,2026-01-06,700,POINT ({LON} {north(1000)})\n",
        encoding="utf-8",
    )
    units = tmp_path / "units.csv"
    units.write_text("unit_id,tr_id\n1,700\n", encoding="utf-8")
    return schedule, units


def test_build_sequence(tmp_path: Path) -> None:
    schedule = load_schedule(*schedule_files(tmp_path))
    fleet = Fleet()
    fill(fleet, SEQ_LEN)
    fleet.add(point(2, T0, speed=40))
    sequence = build_sequence(fleet, schedule, 1, datetime.fromtimestamp(T0, UTC))
    assert (sequence.tr_id, sequence.target.stop_id) == (700, 77)
    assert sequence.steps.shape == (SEQ_LEN, len(FEATURES))
    assert sequence.steps.dtype == np.float32
    # До целевой остановки плановых нет, поэтому cur_dev_s = 0
    assert sequence.cur_dev_s == 0.0
    assert sequence.steps[-1].tolist() == pytest.approx([0.0, 0.0, 1000.0, 90.0, 40.0, 1.0, 0.0])
    assert sequence.steps[0, 1] == SEQ_LEN - 1  # скорость самой старой точки


def test_build_sequence_uses_passed_stops(tmp_path: Path) -> None:
    schedule_path, units = schedule_files(tmp_path)
    with schedule_path.open("a", encoding="utf-8") as f:
        # Остановка по плану в 09:50 там, где автобус стоит с 09:45:15 до 09:52
        f.write(f"76,2026-01-06 09:50:00,2026-01-06,700,POINT ({LON} {LAT})\n")
    fleet = Fleet()
    for i in range(SEQ_LEN):  # стоит на остановке, потом отъезжает на 100 м к северу
        t = T0 - 15 * (SEQ_LEN - 1 - i)
        fleet.add(point(1, t, lat=LAT if t <= T0 - 480 else north(100)))
    sequence = build_sequence(
        fleet, load_schedule(schedule_path, units), 1, datetime.fromtimestamp(T0, UTC)
    )
    # Первая точка у остановки — T0 - 885 с, то есть 09:45:15, на 285 с раньше плана
    assert sequence.cur_dev_s == -285.0
    assert (sequence.steps[:, 0] == -285.0).all()


def test_build_sequence_explains_missing_data(tmp_path: Path) -> None:
    schedule = load_schedule(*schedule_files(tmp_path))
    fleet = Fleet()
    fill(fleet, SEQ_LEN)
    with pytest.raises(MissingData, match="not in the units list"):
        build_sequence(fleet, schedule, 2, datetime.fromtimestamp(T0, UTC))
    with pytest.raises(MissingData, match="no stop"):
        build_sequence(fleet, schedule, 1, datetime.fromtimestamp(T0 + 3600, UTC))

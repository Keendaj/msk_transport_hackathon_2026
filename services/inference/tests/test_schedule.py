from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from inference.schedule import Schedule, load_schedule

HEADER = "tt_action_item_id,time_begin,order_date,manual_fill,tr_id,geom,building_address\n"


def stop(stop_id: int, time_begin: str, tr_id: int = 7, geom: str = "POINT (37.5 55.8)") -> str:
    return f'{stop_id},{time_begin},2026-01-06,True,{tr_id},{geom},"ул. Тестовая, д.1"\n'


def make_schedule(tmp_path: Path, *rows: str) -> Schedule:
    schedule = tmp_path / "schedule.csv"
    schedule.write_text(HEADER + "".join(rows), encoding="utf-8")
    units = tmp_path / "units.csv"
    units.write_text("unit_id,tr_id\n100,7\n", encoding="utf-8")
    return load_schedule(schedule, units)


def at(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=UTC)


def test_units(tmp_path: Path) -> None:
    schedule = make_schedule(tmp_path, stop(1, "2026-01-06 10:12:00"))
    assert schedule.tr_id(100) == 7
    assert schedule.tr_id(200) is None


def test_first_stop_in_window(tmp_path: Path) -> None:
    schedule = make_schedule(
        tmp_path,
        stop(3, "2026-01-06 10:15:00"),
        stop(1, "2026-01-06 10:10:00"),
        stop(2, "2026-01-06 10:12:00.000000000", geom="POINT (37.61 55.75)"),
    )
    target = schedule.target(7, at("2026-01-06 10:00"))
    assert target is not None
    assert (target.stop_id, target.lon, target.lat) == (2, 37.61, 55.75)
    assert target.planned_at == at("2026-01-06 10:12")


@pytest.mark.parametrize(
    ("moment", "expected"),
    [
        ("2026-01-06 10:00", 1),  # 10:15 — верхняя граница входит
        ("2026-01-06 10:05", None),  # 10:15 — нижняя граница не входит
        ("2026-01-06 09:59", None),
    ],
)
def test_window_bounds(tmp_path: Path, moment: str, expected: int | None) -> None:
    schedule = make_schedule(tmp_path, stop(1, "2026-01-06 10:15:00"))
    target = schedule.target(7, at(moment))
    assert (target.stop_id if target else None) == expected


def test_tie_takes_first_in_file(tmp_path: Path) -> None:
    schedule = make_schedule(
        tmp_path, stop(9, "2026-01-06 10:12:00"), stop(4, "2026-01-06 10:12:00")
    )
    target = schedule.target(7, at("2026-01-06 10:00"))
    assert target is not None and target.stop_id == 9


def test_repeats_every_day(tmp_path: Path) -> None:
    schedule = make_schedule(tmp_path, stop(1, "2026-01-06 10:12:00"))
    target = schedule.target(7, at("2026-09-26 10:00"))
    assert target is not None and target.planned_at == at("2026-09-26 10:12")


def test_order_after_midnight_belongs_to_previous_day(tmp_path: Path) -> None:
    schedule = make_schedule(
        tmp_path,
        stop(1, "2026-01-06 06:00:00"),
        stop(2, "2026-01-07 00:30:00"),
        stop(3, "2026-01-06 06:00:00", tr_id=8),
    )
    target = schedule.target(7, at("2026-09-27 00:18"))
    assert target is not None
    assert (target.stop_id, target.planned_at) == (2, at("2026-09-27 00:30"))
    # Утренний наряд начинается как обычно
    target = schedule.target(8, at("2026-09-27 05:48"))
    assert target is not None and target.planned_at == at("2026-09-27 06:00")


def test_planned_stops(tmp_path: Path) -> None:
    schedule = make_schedule(
        tmp_path,
        stop(3, "2026-01-06 10:20:00"),
        stop(1, "2026-01-06 08:00:00"),
        stop(2, "2026-01-06 09:30:00"),
        stop(4, "2026-01-06 09:40:00", tr_id=8),
    )
    stops = schedule.planned_stops(
        7, at("2026-09-26 10:00"), timedelta(hours=1), timedelta(minutes=20)
    )
    assert [(s.stop_id, s.planned_at) for s in stops] == [
        (2, at("2026-09-26 09:30")),
        (3, at("2026-09-26 10:20")),
    ]
    assert schedule.planned_stops(99, at("2026-09-26 10:00"), timedelta(hours=1), timedelta()) == []


def test_skips_stops_without_point(tmp_path: Path) -> None:
    schedule = make_schedule(tmp_path, stop(1, "2026-01-06 10:12:00", geom="EMPTY"))
    assert schedule.target(7, at("2026-01-06 10:00")) is None


def test_without_files() -> None:
    schedule = load_schedule(None, None)
    assert schedule.tr_id(100) is None
    assert schedule.target(7, at("2026-01-06 10:00")) is None

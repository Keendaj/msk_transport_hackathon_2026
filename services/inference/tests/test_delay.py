import math
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from inference.delay import current_delay, passages
from inference.schedule import PlannedStop

T0 = datetime(2026, 1, 6, 10, 0, tzinfo=UTC)
LON, LAT = 37.6, 55.75
METERS_PER_DEGREE_LAT = 6367000.0 * math.pi / 180


def north(meters: float) -> float:
    return LAT + meters / METERS_PER_DEGREE_LAT


def stop(minutes: float, meters_north: float = 0.0, stop_id: int = 1) -> PlannedStop:
    return PlannedStop(stop_id, T0 + timedelta(minutes=minutes), LON, north(meters_north))


def track(*points: tuple[float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Точки (минуты от T0, метры к северу) в массивы времени и координат."""
    times = np.array([(T0 + timedelta(minutes=m)).timestamp() for m, _ in points])
    lat = np.array([north(meters) for _, meters in points])
    return times, np.full(len(points), LON), lat


def at(minutes: float) -> datetime:
    return T0 + timedelta(minutes=minutes)


def test_passage_is_closest_point() -> None:
    times, lon, lat = track((0, -300), (1, -40), (2, 10), (3, 200))
    assert passages(times, lon, lat, [stop(1)]) == {0: times[2]}


def test_passage_while_standing_is_arrival() -> None:
    times, lon, lat = track((0, -300), (1, 0), (2, 0), (3, 0))
    assert passages(times, lon, lat, [stop(1)]) == {0: times[1]}


@pytest.mark.parametrize(
    "points",
    [
        [(0, -300), (1, -100), (2, 70), (3, 300)],  # не подъезжал ближе 60 м
        [(0, -300), (1, -100), (2, -10)],  # ещё подъезжает: после ближайшей точки ничего нет
        [(-12, 0), (-11, 100)],  # раньше окна: за 10 минут до плана
        [(16, 0), (17, 100)],  # позже окна: через 15 минут после плана
    ],
)
def test_no_passage(points: list[tuple[float, float]]) -> None:
    assert passages(*track(*points), [stop(0)]) == {}


def test_stops_are_matched_in_order() -> None:
    # Кольцо: автобус дважды проезжает одну и ту же точку, по плану в 0 и в 8 минут
    times, lon, lat = track((0, 0), (1, 300), (8, 0), (9, 300))
    assert passages(times, lon, lat, [stop(0), stop(8, stop_id=2)]) == {0: times[0], 1: times[2]}


def test_delay_at_passed_stop() -> None:
    times, lon, lat = track((0, 0), (1, 300))
    assert current_delay(times, lon, lat, [stop(-1)], at(2)) == 60.0


def test_delay_is_negative_when_early() -> None:
    times, lon, lat = track((0, 0), (1, 300))
    assert current_delay(times, lon, lat, [stop(2), stop(5, 5000, 2)], at(3)) == -120.0


def test_lower_bound_when_stop_not_reached() -> None:
    # Проехал остановку с опозданием 30 с, следующую по плану 4 минуты назад — ещё нет
    times, lon, lat = track((0, 0), (0.5, 0), (1, 300))
    stops = [stop(-0.5), stop(2, 2000, 2)]
    assert current_delay(times, lon, lat, stops, at(6)) == 240.0
    # Пока до следующей по плану не дошло, остаётся задержка на проеханной
    assert current_delay(times, lon, lat, stops, at(1.5)) == 30.0


def test_long_missed_stop_is_not_a_delay() -> None:
    # Остановка по плану 40 минут назад не найдена: перерыв на конечной или промах трекера
    times, lon, lat = track((-50, 0), (-49, 300), (-5, 5000), (0, 5000))
    stops = [stop(-50), stop(-40, 2000, 2)]
    assert current_delay(times, lon, lat, stops, at(0)) == 0.0


def test_missed_stop_before_passed_one_takes_later_delay() -> None:
    # Остановку по плану в 3 мин трекер не поймал, а следующую проехал на минуту раньше плана
    times, lon, lat = track((0, 0), (1, 300), (5, 1000), (5.25, 1500))
    stops = [stop(-1), stop(3, 500, 2), stop(6, 1000, 3)]
    assert current_delay(times, lon, lat, stops, at(5.5)) == -60.0


def test_no_data() -> None:
    empty = np.array([], dtype=np.float64)
    assert current_delay(empty, empty, empty, [], at(0)) == 0.0
    assert current_delay(empty, empty, empty, [stop(-3)], at(0)) == 180.0

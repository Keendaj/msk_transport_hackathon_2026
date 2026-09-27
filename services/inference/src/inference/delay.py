"""Текущее отклонение от расписания ``cur_dev_s`` по треку ТС.

Организаторы берут задержку на последней остановке с планом не позже T по факту прибытия,
которого у опаздывающего автобуса к T ещё нет. Поэтому проезд остановки определяется по
треку, а для остановки, до которой автобус ещё не доехал, берётся оценка снизу ``T − план``.

Константы:
    LOOKBACK: Насколько назад от T смотреть расписание и трек.
    PASS_RADIUS_M: Остановка проехана, если трек подходил к ней ближе, метры.
    PASS_EARLY, PASS_LATE: Проезд ищется в окне [план − PASS_EARLY, план + PASS_LATE].
    RECENT: Оценка снизу ``T − план`` берётся, только пока она не больше этого. Иначе автобус,
        скорее всего, стоит в перерыве на конечной.
"""

import math
from datetime import datetime, timedelta

import numpy as np
import numpy.typing as npt

from inference.geo import haversine_m
from inference.schedule import PlannedStop

LOOKBACK = timedelta(hours=2)
PASS_RADIUS_M = 60.0
PASS_EARLY = timedelta(minutes=10)
PASS_LATE = timedelta(minutes=15)
RECENT = timedelta(minutes=15)


def passages(
    times: npt.NDArray[np.float64],
    lon: npt.NDArray[np.float64],
    lat: npt.NDArray[np.float64],
    stops: list[PlannedStop],
) -> dict[int, float]:
    """Моменты проезда остановок по треку.

    Момент проезда — время ближайшей к остановке точки в окне
    [план − ``PASS_EARLY``, план + ``PASS_LATE``], если она ближе ``PASS_RADIUS_M``
    и после неё в окне есть ещё точки, то есть автобус уехал. Остановки проезжаются
    по порядку: следующая ищется только после проезда предыдущей.

    Args:
        times: Время точек трека, Unix-секунды, по возрастанию.
        lon: Долгота точек, градусы.
        lat: Широта точек, градусы.
        stops: Плановые остановки по возрастанию планового времени.

    Returns:
        Индекс проеханной остановки в ``stops`` → момент проезда, Unix-секунды.
    """
    passed: dict[int, float] = {}
    previous = -math.inf
    for i, stop in enumerate(stops):
        planned = stop.planned_at.timestamp()
        window = np.flatnonzero(
            (times > previous)
            & (times >= planned - PASS_EARLY.total_seconds())
            & (times <= planned + PASS_LATE.total_seconds())
        )
        if len(window) == 0:
            continue
        distance = haversine_m(lon[window], lat[window], stop.lon, stop.lat)
        closest = int(distance.argmin())
        if distance[closest] < PASS_RADIUS_M and closest < len(window) - 1:
            previous = passed[i] = float(times[window[closest]])
    return passed


def current_delay(
    times: npt.NDArray[np.float64],
    lon: npt.NDArray[np.float64],
    lat: npt.NDArray[np.float64],
    stops: list[PlannedStop],
    at: datetime,
) -> float:
    """Отклонение от расписания на момент ``at``, секунды, плюс — опоздание.

    Ниже «последняя задержка» — задержка на последней проеханной остановке, или 0, если
    проеханных нет. Берётся последняя остановка с планом не позже ``at``:

    - если она проехана, это задержка на ней;
    - если она пропущена, но проехана более поздняя, это последняя задержка;
    - если автобус до неё ещё не доехал и ``at − план`` не больше ``RECENT``, это наибольшее
      из ``at − план`` и последней задержки;
    - иначе, как и без остановок с планом не позже ``at``, это последняя задержка.

    Args:
        times: Время точек трека, Unix-секунды, по возрастанию.
        lon: Долгота точек, градусы.
        lat: Широта точек, градусы.
        stops: Плановые остановки по возрастанию планового времени.
        at: Момент, на который считается отклонение.
    """
    passed = passages(times, lon, lat, stops)
    delays = {i: t - stops[i].planned_at.timestamp() for i, t in passed.items()}
    last = max(passed, default=None)
    last_delay = delays[last] if last is not None else 0.0
    due = [i for i, stop in enumerate(stops) if stop.planned_at <= at]
    if not due:
        return last_delay
    k = due[-1]
    if last is not None and last >= k:
        return delays.get(k, last_delay)
    behind = at - stops[k].planned_at
    return max(last_delay, behind.total_seconds()) if behind <= RECENT else last_delay

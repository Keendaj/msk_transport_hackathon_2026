"""Расстояния между точками на поверхности Земли."""

import numpy as np
import numpy.typing as npt

EARTH_RADIUS_M = 6367000.0


def haversine_m(
    lon1: npt.ArrayLike, lat1: npt.ArrayLike, lon2: npt.ArrayLike, lat2: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    """Расстояние по формуле гаверсинусов, метры.

    Координаты в градусах: числа или массивы, которые numpy приводит к общей форме.
    """
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = (
        np.sin((lat2 - lat1) / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2.0) ** 2
    )
    return 2.0 * np.arcsin(np.sqrt(a)) * EARTH_RADIUS_M

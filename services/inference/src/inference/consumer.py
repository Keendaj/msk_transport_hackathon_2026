"""Чтение телеметрии из Kafka: трек ТС, прогноз и запись в БД."""

import asyncio
import logging
import math
from datetime import UTC, datetime

from aiokafka import AIOKafkaConsumer
from pydantic import ValidationError

from inference.features import MissingData, TrackPoint, track_point
from inference.model import Predictor
from inference.schemas import Prediction, Telemetry
from inference.storage import TRANSIENT_ERRORS, Storage

log = logging.getLogger(__name__)

RETRY_DELAY = 2.0


class TelemetryConsumer:
    """Читает пакеты из Kafka, копит точки, строит прогнозы и пишет всё в БД.

    Args:
        predictor: Модель с расписанием и треками всех ТС.
        consumer: Запущенный потребитель Kafka, подписанный на топик телеметрии.
        storage: Запись пакетов и прогнозов в БД.
        predict_every_s: Прогноз по одному ТС строится не чаще раза в столько секунд
            по времени устройства.
    """

    def __init__(
        self,
        predictor: Predictor,
        consumer: AIOKafkaConsumer,
        storage: Storage,
        predict_every_s: float = 15.0,
    ) -> None:
        self._predictor = predictor
        self._consumer = consumer
        self._storage = storage
        self._predict_every = predict_every_s
        self._predicted_at: dict[int, float] = {}

    async def run(self) -> None:
        """Обрабатывает сообщения по одному, пока задачу не отменят.

        Сообщение, на котором обработка упала, пишется в лог и пропускается: иначе чтение
        остановилось бы, а после перезапуска сервиса упало бы на нём же.
        """
        async for message in self._consumer:
            try:
                await self.handle(message.value)
            except Exception:
                log.exception(f"Message {message.partition}:{message.offset} skipped")

    async def handle(self, value: bytes | None) -> Prediction | None:
        """Сохраняет пакет и возвращает прогноз, если он построен по этому пакету.

        Невалидный пакет пропускается. Прогноз не строится, если в пакете нет достоверных
        координат, с прошлого прогноза по ТС прошло меньше ``predict_every_s`` или данных
        для прогноза не хватает.

        Args:
            value: Тело сообщения Kafka: JSON пакета телеметрии от transceiver.
        """
        telemetry = self._parse(value)
        if telemetry is None:
            return None
        prediction = None
        point = track_point(telemetry)
        if point is not None:
            # Fleet меняется только здесь, в цикле событий, до ухода прогноза в поток
            self._predictor.fleet.add(point)
            if point.time - self._predicted_at.get(point.unit_id, -math.inf) >= self._predict_every:
                prediction = await self._predict(point)
        await self._save(telemetry, prediction)
        return prediction

    @staticmethod
    def _parse(value: bytes | None) -> Telemetry | None:
        """Разбирает пакет, невалидный пишет в лог и отбрасывает."""
        try:
            return Telemetry.model_validate_json(value or b"")
        except ValidationError as e:
            log.warning(f"Invalid telemetry skipped: {e}")
            return None

    async def _predict(self, point: TrackPoint) -> Prediction | None:
        """Прогноз на момент точки в отдельном потоке, или ``None``, если его не построить."""
        at = datetime.fromtimestamp(point.time, UTC)
        try:
            prediction = await asyncio.to_thread(self._predictor.predict, point.unit_id, at)
        except MissingData:
            return None
        except Exception:
            log.exception(f"Prediction failed for unitId={point.unit_id}")
            return None
        self._predicted_at[point.unit_id] = point.time
        return prediction

    async def _save(self, telemetry: Telemetry, prediction: Prediction | None) -> None:
        """Пишет пакет с прогнозом в БД.

        Пока БД недоступна, повторяет запись каждые ``RETRY_DELAY`` секунд: чтение Kafka
        встаёт, и сообщения копятся в топике. При другой ошибке пакет пропускается.
        """
        while True:
            try:
                await self._storage.save(telemetry, prediction)
                return
            except TRANSIENT_ERRORS as e:
                log.warning(f"Database unavailable, retrying in {RETRY_DELAY}s: {e!r}")
                await asyncio.sleep(RETRY_DELAY)
            except Exception:
                log.exception(f"Telemetry of unitId={telemetry.unit_id} not saved")
                return

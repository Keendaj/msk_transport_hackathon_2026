"""FastAPI-приложение inference.

На время работы приложения загружает модель и расписание и запускает чтение телеметрии
из Kafka. API отдаёт прогноз и вход модели на момент последнего пакета ТС, Swagger — на
``/docs``.
"""

import asyncio
import logging
import signal
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from typing import Annotated, Any

from aiokafka import AIOKafkaConsumer
from asyncpg import create_pool
from fastapi import FastAPI, HTTPException, Path, Request

from inference.config import settings
from inference.consumer import TelemetryConsumer
from inference.features import FEATURES, MissingData, build_sequence
from inference.model import Predictor, load_model
from inference.schedule import load_schedule
from inference.schemas import Error, Features, Health, Prediction, TargetStop
from inference.storage import Storage

logging.basicConfig(
    level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger(__name__)

DESCRIPTION = """
Читает пакеты телеметрии из Kafka (топик `telemetry`), копит треки всех ТС и прогнозирует
задержку на **целевой остановке** — первой, чьё плановое время попадает в интервал
(T + 10 мин, T + 15 мин] после момента прогноза T. Пакеты и прогнозы пишутся в PostgreSQL.

Прогноз строится, если ТС есть в `units.csv`, у него есть целевая остановка по расписанию
и хотя бы 60 точек с достоверными координатами за последние 20 минут. Иначе эндпоинты
прогноза отвечают 404 с причиной в `detail`.

Треки хранятся только в памяти и после перезапуска копятся заново.
"""

TAGS = [
    {"name": "service", "description": "Состояние сервиса"},
    {"name": "prediction", "description": "Прогноз задержки и вход модели по ТС"},
]

# ТС из emulator_config.json с самым полным расписанием: unit_id → tr_id
EMULATOR_UNITS = {1012706: 132430, 893159: 122048, 1099984: 133957}

UnitId = Annotated[
    int,
    Path(
        description="`unitId` трекера",
        openapi_examples={
            str(unit_id): {"summary": f"{unit_id}, автобус tr_id {tr_id}", "value": unit_id}
            for unit_id, tr_id in EMULATOR_UNITS.items()
        },
    ),
]

NOT_FOUND: dict[int | str, dict[str, Any]] = {
    404: {
        "model": Error,
        "description": "От ТС нет пакетов с достоверными координатами и временем или данных "
        "для прогноза не хватает, причина в `detail`",
    }
}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Загружает модель и расписание, запускает чтение Kafka и останавливает его в конце."""
    predictor = Predictor(
        load_model(settings.model_path),
        load_schedule(settings.schedule_path, settings.units_path),
    )
    pool = await create_pool(settings.database_url, min_size=1, max_size=5)
    kafka_consumer = AIOKafkaConsumer(
        settings.kafka_telemetry_topic,
        bootstrap_servers=settings.kafka_bootstrap_servers,
        group_id=settings.kafka_group_id,
        auto_offset_reset="earliest",
    )
    await kafka_consumer.start()
    consumer = TelemetryConsumer(predictor, kafka_consumer, Storage(pool), settings.predict_every_s)
    consumer_task = asyncio.create_task(consumer.run())
    consumer_task.add_done_callback(_on_consumer_done)

    log.info(
        f"Consuming {settings.kafka_telemetry_topic} from {settings.kafka_bootstrap_servers} "
        f"as group {settings.kafka_group_id}"
    )

    app.state.predictor = predictor
    app.state.consumer_task = consumer_task
    try:
        yield
    finally:
        consumer_task.cancel()
        await asyncio.gather(consumer_task, return_exceptions=True)
        await kafka_consumer.stop()
        await pool.close()


def _on_consumer_done(task: asyncio.Task[None]) -> None:
    """Пишет в лог, почему остановилось чтение Kafka, и завершает сервис.

    Само чтение останавливается только с ошибкой, а контейнер со статусом unhealthy
    ``restart: unless-stopped`` не перезапускает. Поэтому сервис выходит, и Docker
    запускает его заново. Отмена при остановке приложения пропускается.
    """
    if task.cancelled():
        return
    log.critical("Kafka consumer stopped, shutting down", exc_info=task.exception())
    _shutdown()


def _shutdown() -> None:
    """Штатно останавливает uvicorn, как это делает ``docker stop``."""
    signal.raise_signal(signal.SIGTERM)


app = FastAPI(
    title="inference",
    summary="Прогноз задержки автобусов по телеметрии NDTP",
    description=DESCRIPTION,
    version=version("inference"),
    openapi_tags=TAGS,
    lifespan=lifespan,
)


@app.get(
    "/health",
    tags=["service"],
    summary="Проверка работоспособности",
    response_description="Сервис работает",
    responses={503: {"model": Error, "description": "Чтение Kafka остановилось"}},
)
async def health(request: Request) -> Health:
    """Отвечает `ok` и версию модели, пока работает чтение Kafka.

    Используется в `HEALTHCHECK` образа.
    """
    if request.app.state.consumer_task.done():
        raise HTTPException(503, "Kafka consumer stopped")
    return Health(status="ok", model_version=request.app.state.predictor.model.version)


# Обработчики ниже async: читают Fleet в цикле событий, где его меняет консьюмер


@app.get(
    "/predict/{unit_id}",
    tags=["prediction"],
    summary="Прогноз задержки ТС",
    response_description="Прогноз на момент последнего пакета ТС",
    responses=NOT_FOUND,
)
async def predict(unit_id: UnitId, request: Request) -> Prediction:
    """Прогноз задержки на целевой остановке на момент последнего пакета ТС.

    Прогноз строится заново при каждом запросе и в БД не пишется.
    """
    predictor: Predictor = request.app.state.predictor
    try:
        return predictor.predict(unit_id, _last_moment(predictor, unit_id))
    except MissingData as e:
        raise HTTPException(404, str(e)) from e


@app.get(
    "/features/{unit_id}",
    tags=["prediction"],
    summary="Вход модели для ТС",
    response_description="Признаки на момент последнего пакета ТС",
    responses=NOT_FOUND,
)
async def features(unit_id: UnitId, request: Request) -> Features:
    """Последовательность признаков, по которой `/predict/{unit_id}` строит прогноз.

    Нужна, чтобы проверить вход модели и сравнить его с ноутбуком обучения.
    """
    predictor: Predictor = request.app.state.predictor
    try:
        sequence = build_sequence(
            predictor.fleet, predictor.schedule, unit_id, _last_moment(predictor, unit_id)
        )
    except MissingData as e:
        raise HTTPException(404, str(e)) from e
    return Features(
        unit_id=unit_id,
        tr_id=sequence.tr_id,
        at=sequence.at,
        target=TargetStop(**asdict(sequence.target)),
        cur_dev_s=sequence.cur_dev_s,
        columns=list(FEATURES),
        steps=sequence.steps.tolist(),
    )


def _last_moment(predictor: Predictor, unit_id: int) -> datetime:
    """Время последней точки ТС.

    Raises:
        HTTPException: 404, если от ТС нет пакетов с достоверными координатами и временем.
    """
    last = predictor.fleet.last_time(unit_id)
    if last is None:
        raise HTTPException(404, f"No valid telemetry from unit {unit_id}")
    return datetime.fromtimestamp(last, UTC)

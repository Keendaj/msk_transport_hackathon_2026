"""FastAPI-приложение transceiver.

На время работы приложения запускает TCP-сервер NDTP и продюсер Kafka. API отдаёт последний
пакет телеметрии от каждого трекера, Swagger — на ``/docs``.
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib.metadata import version
from typing import Annotated, Any

from aiokafka import AIOKafkaProducer
from fastapi import FastAPI, HTTPException, Path, Request

from transceiver.config import settings
from transceiver.publisher import TelemetryPublisher
from transceiver.schemas import Error, Health, Telemetry
from transceiver.server import NdtpServer

logging.basicConfig(
    level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger(__name__)

DESCRIPTION = """
Принимает пакеты NDTP от трекеров по TCP (порт `9000`) и публикует их в Kafka
(топик `telemetry`, ключ — `unit_id`), откуда их читает сервис inference.

API отдаёт последний пакет от каждого трекера. Пакеты хранятся только в памяти
и после перезапуска копятся заново.
"""

TAGS = [
    {"name": "service", "description": "Состояние сервиса"},
    {"name": "units", "description": "Последние пакеты телеметрии от трекеров"},
]

UnitId = Annotated[
    int,
    Path(
        description="`unitId` трекера",
        openapi_examples={
            str(unit_id): {"summary": f"{unit_id} из emulator_config.json", "value": unit_id}
            for unit_id in (1012706, 893159, 1099984)
        },
    ),
]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Запускает продюсер Kafka и TCP-сервер NDTP и останавливает их вместе с приложением."""
    producer = AIOKafkaProducer(bootstrap_servers=settings.kafka_bootstrap_servers)
    await producer.start()
    publisher = TelemetryPublisher(producer, settings.kafka_telemetry_topic)
    ndtp_server = NdtpServer(publisher, send_ack=settings.ndtp_send_ack)
    tcp_server = await asyncio.start_server(
        ndtp_server.handle, settings.ndtp_host, settings.ndtp_port
    )

    log.info(
        f"NDTP listening {settings.ndtp_host}:{settings.ndtp_port}, "
        f"Kafka {settings.kafka_bootstrap_servers} topic {settings.kafka_telemetry_topic}"
    )

    app.state.ndtp = ndtp_server
    try:
        yield
    finally:
        tcp_server.close()
        await ndtp_server.close_connections()
        await tcp_server.wait_closed()
        await producer.stop()


app = FastAPI(
    title="transceiver",
    summary="Приём телематики NDTP и публикация пакетов в Kafka",
    description=DESCRIPTION,
    version=version("transceiver"),
    openapi_tags=TAGS,
    lifespan=lifespan,
)


@app.get(
    "/health",
    tags=["service"],
    summary="Проверка работоспособности",
    response_description="Сервис работает",
)
async def health() -> Health:
    """Отвечает `ok`, пока сервис работает. Используется в `HEALTHCHECK` образа."""
    return Health(status="ok")

@app.get(
    "/units",
    tags=["units"],
    summary="Последние пакеты всех трекеров",
    response_model=None,
    responses={200: {"model": dict[int, Telemetry], "description": "Пакеты по `unit_id`"}},
)
async def units(request: Request) -> dict[int, dict[str, Any]]:
    """Последний пакет от каждого трекера, который присылал телеметрию с запуска сервиса."""
    return request.app.state.ndtp.last


@app.get(
    "/units/{unit_id}",
    tags=["units"],
    summary="Последний пакет трекера",
    response_model=None,
    responses={
        200: {"model": Telemetry, "description": "Последний пакет трекера"},
        404: {"model": Error, "description": "От трекера ещё не было пакетов"},
    },
)
async def unit(unit_id: UnitId, request: Request) -> dict[str, Any]:
    """Последний пакет телеметрии от трекера `unit_id`."""
    last = request.app.state.ndtp.last.get(unit_id)
    if last is None:
        raise HTTPException(404, f"No packets from unitId={unit_id} yet")
    return last

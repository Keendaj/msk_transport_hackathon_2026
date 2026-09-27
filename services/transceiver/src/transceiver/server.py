"""TCP-сервер NDTP: читает кадры трекеров, отвечает на запросы и публикует телеметрию."""

import asyncio
import contextlib
import logging
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

from transceiver import ndtp
from transceiver.publisher import PublishError, TelemetryPublisher

log = logging.getLogger(__name__)


class NdtpServer:
    """Обрабатывает соединения трекеров, по одной задаче asyncio на соединение.

    Handshake только пишется в лог, пакет ``NPH_SND_REALTIME`` превращается в пакет
    телеметрии и публикуется, остальные пакеты пропускаются. ``NPH_RESULT`` уходит только
    после обработки кадра: если Kafka не приняла пакет, ответа нет, и трекер считает пакет
    недоставленным.

    Args:
        publisher: Куда отправлять пакеты телеметрии.
        send_ack: Отвечать ли ``NPH_RESULT`` на кадры с флагом запроса.

    Attributes:
        last: Последний пакет телеметрии по каждому ``unit_id``, его отдаёт API.
    """

    def __init__(self, publisher: TelemetryPublisher, *, send_ack: bool) -> None:
        self._publisher = publisher
        self._send_ack = send_ack
        self.last: dict[int, dict[str, Any]] = {}
        self._connections: dict[asyncio.StreamWriter, asyncio.Task[Any]] = {}

    async def close_connections(self) -> None:
        """Закрывает все соединения и ждёт, пока их обработчики завершатся."""
        tasks = list(self._connections.values())
        for writer in self._connections:
            writer.close()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Обработчик соединения для ``asyncio.start_server``.

        Читает кадры, пока трекер не отключится. Испорченный кадр пропускается, а если поток
        не похож на NDTP, соединение закрывается.
        """
        peer = str(writer.get_extra_info("peername"))
        self._register(writer)
        log.info(f"Tracker connected: {peer}")
        try:
            while True:
                await self._process_next_frame(reader, writer, peer)
        except asyncio.IncompleteReadError, ConnectionError:
            pass
        except ndtp.NdtpError as e:
            log.warning(f"Connection with {peer} closed: {e}")
        finally:
            await self._disconnect(writer, peer)

    def _register(self, writer: asyncio.StreamWriter) -> None:
        """Запоминает соединение и его задачу, чтобы закрыть их при остановке."""
        task = asyncio.current_task()
        assert task is not None
        self._connections[writer] = task

    async def _disconnect(self, writer: asyncio.StreamWriter, peer: str) -> None:
        """Закрывает соединение и забывает его."""
        self._connections.pop(writer, None)
        writer.close()
        with contextlib.suppress(ConnectionError):
            await writer.wait_closed()
        log.info(f"Tracker disconnected: {peer}")

    async def _process_next_frame(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, peer: str
    ) -> None:
        """Читает кадр, обрабатывает его и подтверждает.

        Испорченный кадр и кадр, который не удалось опубликовать, не подтверждаются.
        """
        try:
            frame = await ndtp.read_frame(reader)
            await self._dispatch(frame)
            await self._acknowledge(writer, frame)
        except ndtp.BadFrameError as e:
            log.warning(f"Frame from {peer} skipped: {e}")
        except PublishError as e:
            log.warning(f"Frame from {peer} not acknowledged: {e}")

    async def _acknowledge(self, writer: asyncio.StreamWriter, frame: ndtp.Frame) -> None:
        """Отвечает ``NPH_RESULT``, если ответы включены и трекер его ждёт."""
        if not self._send_ack or not frame.is_request:
            return
        writer.write(ndtp.encode_result(frame))
        await writer.drain()

    async def _dispatch(self, frame: ndtp.Frame) -> None:
        """Передаёт кадр обработчику по сервису и типу пакета."""
        match frame.service_id, frame.type:
            case ndtp.SERVICE_GENERIC_CONTROLS, ndtp.NPH_SGC_CONN_REQUEST:
                self._on_handshake(frame)
            case ndtp.SERVICE_NAVDATA, ndtp.NPH_SND_REALTIME:
                await self._on_realtime(frame)
            case _:
                log.debug(f"Packet skipped: service={frame.service_id} type={frame.type}")

    def _on_handshake(self, frame: ndtp.Frame) -> None:
        """Пишет в лог handshake трекера."""
        request = ndtp.parse_conn_request(frame.body)
        high, low = request.proto_version
        log.info(f"Handshake from unitId={request.peer_address}: NDTP {high}.{low}")

    async def _on_realtime(self, frame: ndtp.Frame) -> None:
        """Запоминает пакет телеметрии как последний от трекера и публикует его."""
        telemetry = self.build_telemetry(frame)
        self.last[frame.peer_address] = telemetry
        await self._publisher.publish(telemetry)

    @staticmethod
    def build_telemetry(frame: ndtp.Frame) -> dict[str, Any]:
        """Пакет телеметрии из кадра ``NPH_SND_REALTIME`` в том виде, в каком он уходит в Kafka.

        Returns:
            Словарь с ключами ``unit_id``, ``request_id``, ``received_at`` (время приёма
            сервером, ISO 8601 в UTC) и ``cells`` (ячейки как словари полей ``ndtp.Cell``).
        """
        return {
            "unit_id": frame.peer_address,
            "request_id": frame.request_id,
            "received_at": datetime.now(UTC).isoformat(),
            "cells": [asdict(cell) for cell in ndtp.parse_cells(frame.body)],
        }

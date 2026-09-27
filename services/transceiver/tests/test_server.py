import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock

from transceiver import ndtp
from transceiver.publisher import PublishError, TelemetryPublisher
from transceiver.server import NdtpServer

NAV = ndtp.encode_cell(
    0, 0, 1_700_000_000, 376173210, 557551234, 0b1110_0000, 200, 40, 60, 90, 100, 150, 12, 1
)


class FakePublisher:
    def __init__(self) -> None:
        self.published: list[dict[str, Any]] = []

    async def publish(self, telemetry: dict[str, Any]) -> None:
        self.published.append(telemetry)


async def test_handshake_and_realtime() -> None:
    publisher = FakePublisher()
    ndtp_server = NdtpServer(publisher, send_ack=True)  # type: ignore[arg-type]
    server = await asyncio.start_server(ndtp_server.handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]

    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    handshake = ndtp.CONN_REQUEST.pack(6, 2, 0, 42, 65535, 0)
    writer.write(
        ndtp.encode_frame(
            42,
            ndtp.SERVICE_GENERIC_CONTROLS,
            ndtp.NPH_SGC_CONN_REQUEST,
            1,
            handshake,
            flags=ndtp.NPH_FLAG_REQUEST,
        )
    )
    writer.write(
        ndtp.encode_frame(
            42, ndtp.SERVICE_NAVDATA, ndtp.NPH_SND_REALTIME, 2, NAV, flags=ndtp.NPH_FLAG_REQUEST
        )
    )
    await writer.drain()

    acks = [await ndtp.read_frame(reader) for _ in range(2)]
    assert [(a.service_id, a.type, a.request_id) for a in acks] == [
        (ndtp.SERVICE_GENERIC_CONTROLS, ndtp.NPH_RESULT, 1),
        (ndtp.SERVICE_NAVDATA, ndtp.NPH_RESULT, 2),
    ]
    assert all(ndtp.RESULT.unpack(a.body) == (0,) for a in acks)

    writer.close()
    await writer.wait_closed()
    server.close()
    await asyncio.wait_for(server.wait_closed(), timeout=5)

    [telemetry] = publisher.published
    assert telemetry["unit_id"] == 42
    assert telemetry["cells"][0]["name"] == "G6CellNav00"
    assert ndtp_server.last[42] == telemetry


async def test_close_connections_disconnects_trackers() -> None:
    ndtp_server = NdtpServer(FakePublisher(), send_ack=True)  # type: ignore[arg-type]
    server = await asyncio.start_server(ndtp_server.handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    await asyncio.sleep(0.1)  # дать серверу принять подключение

    server.close()
    await asyncio.wait_for(ndtp_server.close_connections(), timeout=5)
    await asyncio.wait_for(server.wait_closed(), timeout=5)

    assert await reader.read() == b""  # клиент получил EOF
    writer.close()


async def test_no_ack_when_disabled() -> None:
    publisher = FakePublisher()
    ndtp_server = NdtpServer(publisher, send_ack=False)  # type: ignore[arg-type]
    server = await asyncio.start_server(ndtp_server.handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]

    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(
        ndtp.encode_frame(
            42, ndtp.SERVICE_NAVDATA, ndtp.NPH_SND_REALTIME, 1, NAV, flags=ndtp.NPH_FLAG_REQUEST
        )
    )
    writer.write_eof()

    assert await asyncio.wait_for(reader.read(), timeout=5) == b""
    writer.close()
    server.close()
    await asyncio.wait_for(server.wait_closed(), timeout=5)
    assert len(publisher.published) == 1


@asynccontextmanager
async def connected(
    ndtp_server: NdtpServer,
) -> AsyncIterator[tuple[asyncio.StreamReader, asyncio.StreamWriter]]:
    server = await asyncio.start_server(ndtp_server.handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    try:
        yield reader, writer
    finally:
        writer.close()
        server.close()
        await asyncio.wait_for(server.wait_closed(), timeout=5)


def realtime(request_id: int) -> bytes:
    return ndtp.encode_frame(
        42,
        ndtp.SERVICE_NAVDATA,
        ndtp.NPH_SND_REALTIME,
        request_id,
        NAV,
        flags=ndtp.NPH_FLAG_REQUEST,
    )


async def test_bad_signature_closes_connection() -> None:
    publisher = AsyncMock(spec=TelemetryPublisher)
    async with connected(NdtpServer(publisher, send_ack=True)) as (reader, writer):
        writer.write(b"\x00" * ndtp.NPL.size)
        assert await asyncio.wait_for(reader.read(), timeout=5) == b""
    publisher.publish.assert_not_awaited()


async def test_bad_frame_is_skipped_and_next_one_processed() -> None:
    publisher = AsyncMock(spec=TelemetryPublisher)
    broken = bytearray(realtime(1))
    broken[-1] ^= 0xFF
    async with connected(NdtpServer(publisher, send_ack=True)) as (reader, writer):
        writer.write(bytes(broken) + realtime(2))
        ack = await asyncio.wait_for(ndtp.read_frame(reader), timeout=5)
    assert ack.request_id == 2
    publisher.publish.assert_awaited_once()
    assert publisher.publish.await_args.args[0]["request_id"] == 2


async def test_frame_is_not_acked_until_published() -> None:
    publisher = AsyncMock(spec=TelemetryPublisher)
    publisher.publish.side_effect = [PublishError("Kafka is down"), None]
    async with connected(NdtpServer(publisher, send_ack=True)) as (reader, writer):
        writer.write(realtime(1) + realtime(2))
        ack = await asyncio.wait_for(ndtp.read_frame(reader), timeout=5)
    # Первый пакет Kafka не приняла: ответа на него нет, трекер считает его недоставленным
    assert ack.request_id == 2
    assert publisher.publish.await_count == 2


async def test_unknown_packet_is_acked_but_not_published() -> None:
    publisher = AsyncMock(spec=TelemetryPublisher)
    unknown = ndtp.encode_frame(42, ndtp.SERVICE_NAVDATA, 999, 5, flags=ndtp.NPH_FLAG_REQUEST)
    async with connected(NdtpServer(publisher, send_ack=True)) as (reader, writer):
        writer.write(unknown)
        ack = await asyncio.wait_for(ndtp.read_frame(reader), timeout=5)
    assert (ack.type, ack.request_id) == (ndtp.NPH_RESULT, 5)
    publisher.publish.assert_not_awaited()

import asyncio

import pytest

from transceiver import ndtp

NAV = ndtp.encode_cell(
    0, 0, 1_700_000_000, 376173210, 557551234, 0b1110_0000, 200, 40, 60, 90, 100, 150, 12, 1
)


async def read(data: bytes) -> ndtp.Frame:
    reader = asyncio.StreamReader()
    reader.feed_data(data)
    reader.feed_eof()
    return await ndtp.read_frame(reader)


def test_crc16_modbus_check_value() -> None:
    assert ndtp.crc16_modbus(b"123456789") == 0x4B37


@pytest.mark.parametrize(
    ("cell_type", "size"), [(0, 26), (2, 26), (8, 6), (10, 37), (15, 50), (16, 8)]
)
def test_cell_sizes_match_spec(cell_type: int, size: int) -> None:
    assert ndtp.CELLS[cell_type].struct.size == size


def test_crc_is_stored_byte_swapped() -> None:
    raw = ndtp.encode_frame(1, ndtp.SERVICE_NAVDATA, ndtp.NPH_SND_REALTIME, 1, NAV)
    crc = ndtp.crc16_modbus(raw[ndtp.NPL.size :])
    assert raw[6:8] == crc.to_bytes(2, "big")


async def test_frame_roundtrip() -> None:
    raw = ndtp.encode_frame(42, 1, 101, 7, b"body", flags=ndtp.NPH_FLAG_REQUEST)
    frame = await read(raw)
    assert frame == ndtp.Frame(42, 1, 101, ndtp.NPH_FLAG_REQUEST, 7, b"body")
    assert frame.is_request


async def test_bad_crc() -> None:
    raw = bytearray(ndtp.encode_frame(1, 1, 101, 1, NAV))
    raw[-1] ^= 0xFF
    with pytest.raises(ndtp.BadFrameError):
        await read(bytes(raw))


async def test_bad_signature() -> None:
    raw = bytearray(ndtp.encode_frame(1, 1, 101, 1, NAV))
    raw[0] = 0
    with pytest.raises(ndtp.NdtpError):
        await read(bytes(raw))


def test_conn_request() -> None:
    request = ndtp.parse_conn_request(ndtp.CONN_REQUEST.pack(6, 2, 0, 42, 65535, 0))
    assert request == ndtp.ConnRequest((6, 2), 0, 42, 65535)


def test_parse_realtime_cells() -> None:
    body = NAV + ndtp.encode_cell(8, 0, 1, 350, 120, 20) + ndtp.encode_cell(8, 1, 1, 400, 140, 21)
    cells = ndtp.parse_cells(body)

    assert [(c.name, c.number) for c in cells] == [
        ("G6CellNav00", 0),
        ("G6CellUsi08", 0),
        ("G6CellUsi08", 1),
    ]
    nav = cells[0].fields
    assert nav["lat"] == pytest.approx(55.7551234)
    assert nav["lon"] == pytest.approx(37.6173210)
    assert nav["extraDopBit7"] and not nav["extraDopBit0"]
    assert nav["speedMax"] == 60
    assert cells[2].fields["level_l"] == 140


def test_southern_western_hemisphere() -> None:
    nav = ndtp.encode_cell(0, 0, 0, 376173210, 557551234, 0b1000_0000, *[0] * 8)
    fields = ndtp.parse_cells(nav)[0].fields
    assert fields["lat"] < 0 and fields["lon"] < 0


def test_can_pressure_axis() -> None:
    can = ndtp.encode_cell(10, 0, 0, 125, 1000, 50, 0x8032, 1500, -5, 60, 1, 2, 3, 4, 5, 0)
    fields = ndtp.parse_cells(can)[0].fields
    assert fields["pressureAxis"] == [1, 2, 3, 4, 5]
    assert fields["tEngine"] == -5


def test_parse_stops_on_unknown_cell() -> None:
    body = NAV + bytes([99, 0, 1, 2, 3]) + ndtp.encode_cell(16, 0, 0, 21)
    assert [c.name for c in ndtp.parse_cells(body)] == ["G6CellNav00"]


def crc16_bitwise(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return crc


def test_crc_table_matches_bitwise() -> None:
    data = bytes(range(256)) * 3
    for size in (0, 1, 2, 17, 255, len(data)):
        assert ndtp.crc16_modbus(data[:size]) == crc16_bitwise(data[:size])


def test_parse_stops_on_truncated_cell() -> None:
    body = NAV + ndtp.encode_cell(8, 0, 1, 350, 120, 20)[:-1]
    assert [c.name for c in ndtp.parse_cells(body)] == ["G6CellNav00"]


async def test_too_small_data_size() -> None:
    raw = bytearray(ndtp.encode_frame(1, 1, 101, 1, NAV))
    raw[2:4] = (5).to_bytes(2, "little")
    with pytest.raises(ndtp.NdtpError, match="dataSize") as error:
        await read(bytes(raw))
    assert error.type is ndtp.NdtpError


async def test_unknown_npl_type() -> None:
    raw = bytearray(ndtp.encode_frame(1, 1, 101, 1, NAV))
    raw[8] = 0x03
    with pytest.raises(ndtp.BadFrameError, match="NPL type"):
        await read(bytes(raw))


def test_short_conn_request() -> None:
    with pytest.raises(ndtp.BadFrameError, match="too short"):
        ndtp.parse_conn_request(b"\x06\x00\x02\x00")

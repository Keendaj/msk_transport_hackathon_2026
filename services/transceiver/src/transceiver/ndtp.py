"""Протокол NDTP: разбор и сборка кадров и ячеек телематики.

Кадр состоит из заголовка транспортного уровня NPL (15 байт), заголовка сетевого уровня
NPH (10 байт) и тела. Все поля little-endian, без выравнивания. CRC-16/Modbus считается по
NPH и телу и лежит в NPL с переставленными байтами.

Трекер сначала шлёт handshake ``NPH_SGC_CONN_REQUEST`` (сервис ``SERVICE_GENERIC_CONTROLS``),
затем пакеты ``NPH_SND_REALTIME`` (сервис ``SERVICE_NAVDATA``). Тело realtime-пакета — подряд
идущие ячейки ``[type: u8][number: u8][payload]``. Поддерживаемые ячейки перечислены в
``CELLS``, новый тип добавляется записью туда.

Подробная спецификация: ``dataset/docs/Emulator-and-Telematic-Packets-Specification.md``.
"""

import asyncio
import logging
import struct
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple

log = logging.getLogger(__name__)

NPL = struct.Struct("<HHHHBIH")
NPH = struct.Struct("<HHHI")
CONN_REQUEST = struct.Struct("<HHHIII")
RESULT = struct.Struct("<I")
SIGNATURE = 0x7E7E
NPL_TYPE_NPH = 0x02
NPH_FLAG_REQUEST = 0x0001
SERVICE_GENERIC_CONTROLS = 0
SERVICE_NAVDATA = 1
NPH_RESULT = 0
NPH_SGC_CONN_REQUEST = 100
NPH_SND_REALTIME = 101
CELL_HEADER_SIZE = 2


class NdtpError(Exception):
    """Поток не похож на NDTP: читать его дальше нельзя, соединение закрывается."""


class BadFrameError(NdtpError):
    """Кадр прочитан целиком, но испорчен: его можно пропустить и читать следующий."""


def _crc16_modbus_byte(value: int) -> int:
    """Элемент таблицы CRC-16/Modbus для одного байта."""
    for _ in range(8):
        value = (value >> 1) ^ 0xA001 if value & 1 else value >> 1
    return value


_CRC16_TABLE = [_crc16_modbus_byte(i) for i in range(256)]


def crc16_modbus(data: bytes) -> int:
    """CRC-16/Modbus: полином ``0xA001``, начальное значение ``0xFFFF``."""
    crc = 0xFFFF
    for byte in data:
        crc = (crc >> 8) ^ _CRC16_TABLE[(crc ^ byte) & 0xFF]
    return crc


def swap16(value: int) -> int:
    """Меняет местами байты 16-битного числа: в таком виде CRC лежит в заголовке NPL."""
    return ((value & 0xFF) << 8) | (value >> 8)


class NplHeader(NamedTuple):
    """Заголовок транспортного уровня NPL, 15 байт.

    Attributes:
        signature: Сигнатура кадра, всегда ``SIGNATURE``.
        data_size: Длина NPH-заголовка и тела, байт.
        flags: Флаги NPL: шифрование, CRC, задержка.
        crc: CRC-16/Modbus по NPH и телу с переставленными байтами.
        type: Тип данных, ``NPL_TYPE_NPH`` для пакета NPH.
        peer_address: ``unitId`` трекера.
        request_id: Номер запроса NPL, эмулятор всегда шлёт 0.
    """

    signature: int
    data_size: int
    flags: int
    crc: int
    type: int
    peer_address: int
    request_id: int


@dataclass(frozen=True, slots=True)
class Frame:
    """Кадр NDTP: поля NPH, тело и адрес трекера из NPL.

    Attributes:
        peer_address: ``unitId`` трекера.
        service_id: Сервис NPH: ``SERVICE_GENERIC_CONTROLS`` или ``SERVICE_NAVDATA``.
        type: Тип пакета внутри сервиса, например ``NPH_SND_REALTIME``.
        flags: Флаги NPH, бит ``NPH_FLAG_REQUEST`` — трекер ждёт ответ.
        request_id: Номер запроса NPH, по нему трекер сопоставляет ответ.
        body: Тело пакета после заголовка NPH.
    """

    peer_address: int
    service_id: int
    type: int
    flags: int
    request_id: int
    body: bytes

    @property
    def is_request(self) -> bool:
        """Трекер ждёт на этот кадр ответ ``NPH_RESULT``."""
        return bool(self.flags & NPH_FLAG_REQUEST)


async def read_frame(reader: asyncio.StreamReader) -> Frame:
    """Читает из потока один кадр и проверяет его.

    Raises:
        asyncio.IncompleteReadError: Поток закончился посреди кадра.
        BadFrameError: Не сошлась CRC или тип NPL неизвестен. Кадр прочитан целиком,
            следующий можно читать.
        NdtpError: Неверная сигнатура или длина в NPL, граница следующего кадра неизвестна.
    """
    npl = _parse_npl(await reader.readexactly(NPL.size))
    data = await reader.readexactly(npl.data_size)
    _verify_data(npl, data)
    return _parse_nph(npl.peer_address, data)


def _parse_npl(raw: bytes) -> NplHeader:
    """Разбирает заголовок NPL и проверяет сигнатуру и длину данных."""
    npl = NplHeader(*NPL.unpack(raw))
    if npl.signature != SIGNATURE:
        raise NdtpError(f"Incorrect signature 0x{npl.signature:04X}")
    if npl.data_size < NPH.size:
        raise NdtpError(f"Invalid dataSize: {npl.data_size} is less than NPH")
    return npl


def _verify_data(npl: NplHeader, data: bytes) -> None:
    """Проверяет CRC данных и тип NPL."""
    expected = swap16(npl.crc)
    actual = crc16_modbus(data)
    if actual != expected:
        raise BadFrameError(f"CRC mismatch: 0x{expected:04X} in frame, 0x{actual:04X} actual")
    if npl.type != NPL_TYPE_NPH:
        raise BadFrameError(f"Unknown NPL type 0x{npl.type:02X}")


def _parse_nph(peer_address: int, data: bytes) -> Frame:
    """Разбирает заголовок NPH и отделяет тело."""
    service_id, nph_type, flags, request_id = NPH.unpack_from(data)
    return Frame(peer_address, service_id, nph_type, flags, request_id, data[NPH.size :])


def encode_frame(
    peer_address: int,
    service_id: int,
    nph_type: int,
    request_id: int,
    body: bytes = b"",
    *,
    flags: int = 0,
) -> bytes:
    """Собирает кадр NDTP с заголовками NPL и NPH и CRC.

    Args:
        peer_address: ``unitId`` трекера.
        service_id: Сервис NPH.
        nph_type: Тип пакета внутри сервиса.
        request_id: Номер запроса NPH.
        body: Тело пакета.
        flags: Флаги NPH, например ``NPH_FLAG_REQUEST``.

    Returns:
        Кадр целиком, готовый к отправке в сокет.
    """
    data = NPH.pack(service_id, nph_type, flags, request_id) + body
    crc = swap16(crc16_modbus(data))
    return NPL.pack(SIGNATURE, len(data), 0, crc, NPL_TYPE_NPH, peer_address, 0) + data


def encode_result(request: Frame, error: int = 0) -> bytes:
    """Собирает ответ ``NPH_RESULT`` на кадр-запрос.

    Args:
        request: Кадр, на который отвечает сервер.
        error: Код результата, 0 — успех.
    """
    return encode_frame(
        request.peer_address,
        request.service_id,
        NPH_RESULT,
        request.request_id,
        RESULT.pack(error),
    )


@dataclass(frozen=True, slots=True)
class ConnRequest:
    """Тело handshake ``NPH_SGC_CONN_REQUEST``.

    Attributes:
        proto_version: Версия протокола: старшая и младшая части.
        flags: Флаги соединения: шифрование, CRC, симуляция.
        peer_address: ``unitId`` трекера.
        max_packet_size: Наибольший размер пакета, который принимает трекер, байт.
    """

    proto_version: tuple[int, int]
    flags: int
    peer_address: int
    max_packet_size: int


def parse_conn_request(body: bytes) -> ConnRequest:
    """Разбирает тело handshake.

    Raises:
        BadFrameError: Тело короче 18 байт.
    """
    if len(body) < CONN_REQUEST.size:
        raise BadFrameError(
            f"CONN_REQUEST body too short: {len(body)} bytes, need {CONN_REQUEST.size}"
        )
    high, low, flags, peer_address, max_packet_size, _ = CONN_REQUEST.unpack_from(body)
    return ConnRequest((high, low), flags, peer_address, max_packet_size)


@dataclass(frozen=True, slots=True)
class CellLayout:
    """Бинарный формат ячейки телематики.

    Attributes:
        name: Имя ячейки, например ``G6CellNav00``.
        struct: Формат полезной нагрузки без заголовка ячейки.
        fields: Имена полей в порядке формата.
        post: Постобработка словаря полей на месте, например распаковка битов.
    """

    name: str
    struct: struct.Struct
    fields: tuple[str, ...]
    post: Callable[[dict[str, Any]], None] | None = None

    @property
    def size(self) -> int:
        """Размер полезной нагрузки без заголовка ячейки, байт."""
        return self.struct.size

    def decode(self, payload: bytes) -> dict[str, Any]:
        """Разбирает полезную нагрузку ячейки в словарь «имя поля → значение»."""
        values = dict(zip(self.fields, self.struct.unpack(payload), strict=True))
        if self.post:
            self.post(values)
        return values


def _layout(
    name: str, fmt: str, fields: str, post: Callable[[dict[str, Any]], None] | None = None
) -> CellLayout:
    """Описание ячейки по формату ``struct`` без порядка байт и именам полей через пробел."""
    return CellLayout(name, struct.Struct("<" + fmt), tuple(fields.split()), post)


def _unpack_bits(values: dict[str, Any], name: str) -> None:
    """Заменяет байт ``name`` восемью булевыми полями ``{name}Bit0…7``, бит 0 — младший."""
    bits = values.pop(name)
    values.update({f"{name}Bit{i}": bool(bits >> i & 1) for i in range(8)})


def _degrees(value: int, positive: bool) -> float:
    """Градусы из модуля координаты × 10⁷ и признака северной или восточной полусферы."""
    return value / 1e7 if positive else -value / 1e7


def _nav(values: dict[str, Any]) -> None:
    """Постобработка ``G6CellNav00``: биты ``extraDop`` и координаты ``lat``, ``lon`` в градусах."""
    _unpack_bits(values, "extraDop")
    values["lat"] = _degrees(values["latitude"], values["extraDopBit5"])
    values["lon"] = _degrees(values["longitude"], values["extraDopBit6"])


def _can(values: dict[str, Any]) -> None:
    """Постобработка ``G6CellCan10``: давление на осях 1…5 собирается в список ``pressureAxis``."""
    values["pressureAxis"] = [values.pop(f"pressureAxis{i}") for i in range(5)]


CELLS: dict[int, CellLayout] = {
    0: _layout(
        "G6CellNav00",
        "IIIBBHHHHHBB",
        "timestamp longitude latitude extraDop batVoltage speedAvg speedMax course track "
        "altitude nsat pdop",
        _nav,
    ),
    2: _layout(
        "G6CellIntSensor02",
        "4HBB4HIBBBb",
        "an_in0 an_in1 an_in2 an_in3 di_in di_out di0_counter di1_counter di2_counter "
        "di3_counter odometer csq gprs_state accel_energy ext_volt",
    ),
    8: _layout("G6CellUsi08", "BHHB", "det_status level_mm level_l temperature"),
    10: _layout(
        "G6CellCan10",
        "IIIIHHhB5HI",
        "secFlagStatus allTimeEngine allTrack allFuelConsum fuelLevel speedTurnEngine tEngine "
        "speed pressureAxis0 pressureAxis1 pressureAxis2 pressureAxis3 pressureAxis4 flagAlarm",
        _can,
    ),
    15: _layout(
        "G6CellLls15",
        "H12I",
        "status main_float_level temperature_average percent_of_volume total_Volume weight "
        "density net_Standard_Volume level_of_water pressure vapor_temperature_average "
        "vapor_Weight liquid_phase_Weight",
    ),
    16: _layout("G6CellTermo16", "Ii", "status temp"),
}


@dataclass(frozen=True, slots=True)
class Cell:
    """Разобранная ячейка телематики.

    Attributes:
        type: Тип ячейки, ключ в ``CELLS``.
        number: Номер среди ячеек того же типа в пакете, например второго датчика топлива.
        name: Имя ячейки, например ``G6CellNav00``.
        fields: Значения полей ячейки по именам.
    """

    type: int
    number: int
    name: str
    fields: dict[str, Any]


def encode_cell(cell_type: int, number: int, *values: int) -> bytes:
    """Кодирует ячейку с заголовком, значения полей — в порядке формата из ``CELLS``."""
    return bytes([cell_type, number]) + CELLS[cell_type].struct.pack(*values)


def parse_cells(body: bytes) -> list[Cell]:
    """Разбирает тело пакета ``NPH_SND_REALTIME`` на ячейки.

    На ячейке неизвестного типа или обрезанной ячейке разбор останавливается: длина ячейки
    неизвестна, поэтому остаток тела пропускается с предупреждением в лог.
    """
    cells = []
    offset = 0
    while offset + CELL_HEADER_SIZE <= len(body):
        layout = _layout_at(body, offset)
        if layout is None:
            break
        cells.append(_decode_cell(body, offset, layout))
        offset += CELL_HEADER_SIZE + layout.size
    return cells


def _layout_at(body: bytes, offset: int) -> CellLayout | None:
    """Формат ячейки по смещению ``offset``, или ``None``, если её не разобрать."""
    cell_type = body[offset]
    layout = CELLS.get(cell_type)
    if layout is None:
        log.warning(f"Cell type={cell_type} not supported: skipped {len(body) - offset} bytes")
        return None
    available = len(body) - offset - CELL_HEADER_SIZE
    if available < layout.size:
        log.warning(f"Cell {layout.name} truncated: {available} of {layout.size} bytes")
        return None
    return layout


def _decode_cell(body: bytes, offset: int, layout: CellLayout) -> Cell:
    """Разбирает ячейку по смещению ``offset`` вместе с заголовком."""
    start = offset + CELL_HEADER_SIZE
    payload = body[start : start + layout.size]
    return Cell(body[offset], body[offset + 1], layout.name, layout.decode(payload))

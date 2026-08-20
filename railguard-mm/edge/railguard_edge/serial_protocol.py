from __future__ import annotations
import struct
import math
import zlib
from datetime import datetime, timezone
from dataclasses import dataclass

SYNC = b"RG"
HEADER = struct.Struct("<2sBBHIII")
FEATURE_PAYLOAD = struct.Struct("<7f")
SENSOR_FEATURE_PAYLOAD = struct.Struct("<BBHf3f4f4f5f")
# sync, version, type, payload_len, seq, pps_epoch, sub_us

@dataclass(frozen=True)
class Packet:
    version: int
    packet_type: int
    seq: int
    pps_epoch: int
    sub_us: int
    payload: bytes


def packet_timestamp_ns(packet: Packet) -> int:
    if packet.pps_epoch >= 946684800 and 0 <= packet.sub_us < 1_000_000:
        return packet.pps_epoch * 1_000_000_000 + packet.sub_us * 1_000
    return 0


def packet_timestamp_iso(packet: Packet, fallback: datetime | None = None) -> str:
    """Convert the PPS epoch + sub-second offset to UTC ISO-8601.

    A zero/uninitialized epoch is expected during bench bring-up, in which case the
    host receive time is used explicitly as a fallback.
    """
    if packet.pps_epoch >= 946684800 and 0 <= packet.sub_us < 1_000_000:
        dt = datetime.fromtimestamp(packet.pps_epoch + packet.sub_us / 1_000_000.0, tz=timezone.utc)
    else:
        dt = fallback or datetime.now(timezone.utc)
    return dt.isoformat(timespec="microseconds").replace("+00:00", "Z")


def encode_packet(packet: Packet) -> bytes:
    header = HEADER.pack(SYNC, packet.version, packet.packet_type, len(packet.payload), packet.seq, packet.pps_epoch, packet.sub_us)
    body = header + packet.payload
    return body + struct.pack("<I", zlib.crc32(body) & 0xFFFFFFFF)


def decode_packet(data: bytes) -> Packet:
    if len(data) < HEADER.size + 4:
        raise ValueError("packet too short")
    sync, version, packet_type, payload_len, seq, pps_epoch, sub_us = HEADER.unpack_from(data)
    if sync != SYNC:
        raise ValueError("bad sync")
    expected_len = HEADER.size + payload_len + 4
    if len(data) != expected_len:
        raise ValueError(f"length mismatch: expected {expected_len}, got {len(data)}")
    expected_crc = struct.unpack_from("<I", data, expected_len - 4)[0]
    actual_crc = zlib.crc32(data[:-4]) & 0xFFFFFFFF
    if actual_crc != expected_crc:
        raise ValueError("crc mismatch")
    payload = data[HEADER.size:-4]
    return Packet(version, packet_type, seq, pps_epoch, sub_us, payload)


def decode_feature_payload(payload: bytes) -> dict:
    if len(payload) != FEATURE_PAYLOAD.size:
        raise ValueError(f"feature payload must be {FEATURE_PAYLOAD.size} bytes")
    ax, ay, az, temp_c, lat, lon, speed = FEATURE_PAYLOAD.unpack(payload)
    return {"axis_rms": (ax, ay, az), "temperature_c": temp_c, "lat": lat, "lon": lon, "speed_mps": speed}


def decode_sensor_feature_payload(payload: bytes) -> dict:
    if len(payload) != SENSOR_FEATURE_PAYLOAD.size:
        raise ValueError(f"sensor feature payload must be {SENSOR_FEATURE_PAYLOAD.size} bytes")
    v=SENSOR_FEATURE_PAYLOAD.unpack(payload)
    return {
        "sensor_id":v[0],"flags":v[1],"window_samples":v[2],"sample_rate_hz":v[3],
        "axis_rms":tuple(v[4:7]),"rms":v[7],"peak":v[8],"kurtosis":v[9],"crest_factor":v[10],
        "band_energy":tuple(v[11:15]),"temperature_c":v[15],"humidity":v[16],"lat":v[17],"lon":v[18],"speed_mps":v[19]
    }

def validate_sensor_feature_payload(d: dict) -> None:
    """Semantic validation after CRC/structural decoding.

    CRC proves transport integrity, not that a malfunctioning sensor produced finite
    or physically plausible values. Invalid payloads are rejected before they can
    poison fusion, JSON serialization, or model inputs.
    """
    def finite_nonnegative(name: str, value: float) -> None:
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"invalid {name}: {value}")

    sid=int(d["sensor_id"]); window=int(d["window_samples"]); rate=float(d["sample_rate_hz"])
    if sid not in (0,1,2): raise ValueError(f"invalid sensor_id: {sid}")
    if window < 1 or window > 8192: raise ValueError(f"invalid window_samples: {window}")
    if not math.isfinite(rate) or not (1.0 <= rate <= 100000.0): raise ValueError(f"invalid sample_rate_hz: {rate}")
    for i,v in enumerate(d["axis_rms"]): finite_nonnegative(f"axis_rms[{i}]",v)
    for name in ("rms","peak","kurtosis","crest_factor"):
        finite_nonnegative(name,d[name])
    for i,v in enumerate(d["band_energy"]): finite_nonnegative(f"band_energy[{i}]",v)
    flags=int(d["flags"])
    if flags & 0x01:
        lat=float(d["lat"]); lon=float(d["lon"]); speed=float(d["speed_mps"])
        if not math.isfinite(lat) or not -90.0 <= lat <= 90.0: raise ValueError(f"invalid latitude: {lat}")
        if not math.isfinite(lon) or not -180.0 <= lon <= 180.0: raise ValueError(f"invalid longitude: {lon}")
        if not math.isfinite(speed) or not 0.0 <= speed <= 200.0: raise ValueError(f"invalid speed_mps: {speed}")
    if flags & 0x02:
        temp=float(d["temperature_c"]); hum=float(d["humidity"])
        if not math.isfinite(temp) or not -80.0 <= temp <= 125.0: raise ValueError(f"invalid temperature_c: {temp}")
        if not math.isfinite(hum) or not 0.0 <= hum <= 1.0: raise ValueError(f"invalid humidity: {hum}")



class StreamDecoder:
    """Incrementally recover CRC-framed packets from a serial byte stream."""
    def __init__(self):
        self.buf = bytearray()

    def feed(self, data: bytes) -> list[Packet]:
        self.buf.extend(data)
        out: list[Packet] = []
        while True:
            idx = self.buf.find(SYNC)
            if idx < 0:
                self.buf[:] = self.buf[-1:]
                break
            if idx:
                del self.buf[:idx]
            if len(self.buf) < HEADER.size:
                break
            try:
                _, _, _, payload_len, *_ = HEADER.unpack_from(self.buf)
            except struct.error:
                break
            if payload_len > 4096:
                del self.buf[0]
                continue
            total = HEADER.size + payload_len + 4
            if len(self.buf) < total:
                # Do not let a false sync with a plausible large length head-of-line
                # block a complete CRC-valid packet that is already buffered later.
                next_idx = self.buf.find(SYNC, 1)
                recovered = False
                while next_idx >= 0:
                    remaining = len(self.buf) - next_idx
                    if remaining < HEADER.size:
                        break
                    try:
                        _, _, _, next_len, *_ = HEADER.unpack_from(self.buf, next_idx)
                    except struct.error:
                        break
                    next_total = HEADER.size + next_len + 4
                    if next_len <= 4096 and remaining >= next_total:
                        candidate = bytes(self.buf[next_idx:next_idx + next_total])
                        try:
                            decode_packet(candidate)
                        except ValueError:
                            pass
                        else:
                            del self.buf[:next_idx]
                            recovered = True
                            break
                    next_idx = self.buf.find(SYNC, next_idx + 1)
                if recovered:
                    continue
                break
            candidate = bytes(self.buf[:total])
            try:
                out.append(decode_packet(candidate))
                del self.buf[:total]
            except ValueError:
                del self.buf[0]
        return out

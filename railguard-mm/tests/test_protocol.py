from edge.railguard_edge.serial_protocol import Packet, encode_packet, decode_packet, StreamDecoder, decode_sensor_feature_payload, validate_sensor_feature_payload, HEADER, SYNC

def test_roundtrip():
    p=Packet(1,2,99,1234,5678,b"abc")
    assert decode_packet(encode_packet(p))==p

def test_crc_rejects_corruption():
    raw=bytearray(encode_packet(Packet(1,1,1,1,1,b"payload")))
    raw[-5]^=0x55
    try: decode_packet(bytes(raw))
    except ValueError as e: assert "crc" in str(e)
    else: raise AssertionError("corruption not detected")

def test_stream_decoder_recovers_split_packets():
    a=encode_packet(Packet(1,1,1,1,10,b"a")); b=encode_packet(Packet(1,1,2,1,20,b"bb"))
    d=StreamDecoder(); assert d.feed(b"noise"+a[:5])==[]
    out=d.feed(a[5:]+b)
    assert [x.seq for x in out]==[1,2]

def test_c17_sensor_feature_vector_decodes_in_python():
    raw=bytes.fromhex("5247020248002b00000000f1536560e40100020100020056d0460000803f0000004000004040666606406666c6409a995940cdcc3c40cdcccc3dcdcc4c3e9a99993ecdcccc3e00000000cdcc0c3fcdcc20429a999bc20000c0406baafed1")
    p=decode_packet(raw); assert p.version==2 and p.packet_type==2 and p.seq==43
    d=decode_sensor_feature_payload(p.payload)
    assert d['sensor_id']==2 and d['window_samples']==512
    assert abs(d['rms']-2.1)<1e-6 and abs(d['humidity']-.55)<1e-6 and abs(d['speed_mps']-6.0)<1e-6


def test_sensor_payload_semantic_validation_rejects_nonfinite_and_invalid_context():
    raw=bytes.fromhex("5247020248002b00000000f1536560e40100020100020056d0460000803f0000004000004040666606406666c6409a995940cdcc3c40cdcccc3dcdcc4c3e9a99993ecdcccc3e00000000cdcc0c3fcdcc20429a999bc20000c0406baafed1")
    d=decode_sensor_feature_payload(decode_packet(raw).payload)
    validate_sensor_feature_payload(d)
    bad=dict(d); bad["rms"]=float("nan")
    import pytest
    with pytest.raises(ValueError): validate_sensor_feature_payload(bad)
    bad=dict(d); bad["flags"]=3; bad["humidity"]=1.2
    with pytest.raises(ValueError): validate_sensor_feature_payload(bad)
    bad=dict(d); bad["flags"]=3; bad["lat"]=95.0
    with pytest.raises(ValueError): validate_sensor_feature_payload(bad)


def test_stream_decoder_skips_incomplete_false_sync_when_valid_frame_is_buffered_later():
    valid=bytes.fromhex("5247020248002b00000000f1536560e40100020100020056d0460000803f0000004000004040666606406666c6409a995940cdcc3c40cdcccc3dcdcc4c3e9a99993ecdcccc3e00000000cdcc0c3fcdcc20429a999bc20000c0406baafed1")
    # Fake sync/header advertises a 4000-byte payload but is followed immediately
    # by a complete valid packet. Decoder must not wait for 4000 bytes.
    false_header=HEADER.pack(SYNC,9,9,4000,0,0,0)
    d=StreamDecoder()
    out=d.feed(false_header+valid)
    assert len(out)==1 and out[0].seq==43

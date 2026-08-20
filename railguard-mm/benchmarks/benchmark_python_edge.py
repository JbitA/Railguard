from __future__ import annotations
import json, math, sys, time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from edge.railguard_edge.serial_protocol import Packet, encode_packet, decode_packet, decode_sensor_feature_payload, validate_sensor_feature_payload, packet_timestamp_ns, FEATURE_PAYLOAD, SENSOR_FEATURE_PAYLOAD

def ns_per(iterations, fn):
    t0=time.perf_counter_ns();fn();return (time.perf_counter_ns()-t0)/iterations

def main(n=200_000):
    payload=FEATURE_PAYLOAD.pack(.4,.5,.8,23.0,40.2,-77.8,6.0); packet=encode_packet(Packet(1,1,0,1700000000,1000,payload));valid=0
    def parse():
        nonlocal valid
        for _ in range(n): valid += decode_packet(packet).packet_type==1
    parse_ns=ns_per(n,parse)
    sensor_payload=SENSOR_FEATURE_PAYLOAD.pack(1,1,512,26667.0,.4,.5,.8,.65,1.8,3.2,2.77,.1,.2,.3,.4,0.0,.55,40.2,-77.8,6.0);sensor_packet=encode_packet(Packet(2,2,0,1700000000,1000,sensor_payload));sensor_valid=0
    warmup=min(n,10_000)
    for _ in range(warmup):
        wp=decode_packet(sensor_packet); wd=decode_sensor_feature_payload(wp.payload); validate_sensor_feature_payload(wd); packet_timestamp_ns(wp)
    def parse_sensor():
        nonlocal sensor_valid
        for _ in range(n):
            p=decode_packet(sensor_packet);sensor_valid += decode_sensor_feature_payload(p.payload)['sensor_id']==1
    sensor_parse_ns=ns_per(n,parse_sensor)
    sensor_accept=0
    def accept_sensor():
        nonlocal sensor_accept
        for _ in range(n):
            p=decode_packet(sensor_packet); d=decode_sensor_feature_payload(p.payload); validate_sensor_feature_payload(d)
            end_ns=packet_timestamp_ns(p)
            if end_ns and d["sample_rate_hz"]>0 and d["window_samples"]>0:
                center_ns=end_ns-int((0.5*d["window_samples"]/d["sample_rate_hz"])*1e9)
                sensor_accept += center_ns>0
    sensor_accept_ns=ns_per(n,accept_sensor)
    signal=np.asarray([.8*math.sin(2*math.pi*75*i/2000) for i in range(512)],dtype=np.float64); loops=n//20+1;sink=0.0
    def dsp():
        nonlocal sink
        freqs=(25.,75.,200.,500.)
        for _ in range(loops):
            sum2=float(np.dot(signal,signal));sum4=float(np.dot(signal*signal,signal*signal));rms=math.sqrt(sum2/len(signal));sink+=rms
            for f in freqs:
                c=2*math.cos(2*math.pi*f/2000);s0=s1=s2=0.0
                for v in signal:s0=float(v)+c*s1-s2;s2=s1;s1=s0
    dsp_ns=ns_per(loops,dsp)
    print(json.dumps({"language":"python","iterations":n,"warmup_iterations":warmup,"packet_decode_ns":parse_ns,"packet_decode_mpps":1000.0/parse_ns,"sensor_packet_decode_ns":sensor_parse_ns,"sensor_packet_decode_mpps":1000.0/sensor_parse_ns,"sensor_packet_accept_ns":sensor_accept_ns,"sensor_packet_accept_mpps":1000.0/sensor_accept_ns,"dsp_window_us":dsp_ns/1000.0,"valid_packets":valid,"valid_sensor_packets":sensor_valid,"accepted_sensor_packets":sensor_accept,"sink":sink}))
if __name__=="__main__":main(int(sys.argv[1]) if len(sys.argv)>1 else 200_000)

from __future__ import annotations
import argparse,json,platform,statistics,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def capture(cmd): return json.loads(subprocess.check_output(cmd,text=True,cwd=ROOT).strip().splitlines()[-1])
def cpu_model():
    try:
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.lower().startswith('model name'): return line.split(':',1)[1].strip()
    except OSError: pass
    return 'unknown'
def summarize(rows):
    keys=['packet_decode_ns','packet_decode_mpps','sensor_packet_decode_ns','sensor_packet_decode_mpps','sensor_packet_accept_ns','sensor_packet_accept_mpps','dsp_window_us']
    if 'ring_roundtrip_ns' in rows[0]: keys.append('ring_roundtrip_ns')
    return {k:statistics.median(float(r[k]) for r in rows) for k in keys}
def main():
 p=argparse.ArgumentParser();p.add_argument('--iterations',type=int,default=200_000);p.add_argument('--repeats',type=int,default=5);p.add_argument('--out',type=Path,default=ROOT/'benchmarks/results/latest.json');a=p.parse_args()
 cpp=ROOT/'build/edge-cpp/railguard_bench'
 if not cpp.exists():
  subprocess.check_call(['cmake','-S','edge/cpp','-B','build/edge-cpp','-DCMAKE_BUILD_TYPE=Release'],cwd=ROOT);subprocess.check_call(['cmake','--build','build/edge-cpp','-j2'],cwd=ROOT)
 cpp_runs=[];py_runs=[]
 for _ in range(a.repeats):
  cpp_runs.append(capture([str(cpp),str(a.iterations)]));py_runs.append(capture([sys.executable,'benchmarks/benchmark_python_edge.py',str(a.iterations)]))
 cs,ps=summarize(cpp_runs),summarize(py_runs)
 result={'generated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'host':{'platform':platform.platform(),'cpu':cpu_model(),'python':platform.python_version()},'iterations':a.iterations,'repeats':a.repeats,'median':{'cpp20':cs,'python':ps},'comparison':{'packet_decode_speedup_cpp_vs_python':ps['packet_decode_ns']/cs['packet_decode_ns'],'sensor_packet_decode_speedup_cpp_vs_python':ps['sensor_packet_decode_ns']/cs['sensor_packet_decode_ns'],'sensor_packet_accept_speedup_cpp_vs_python':ps['sensor_packet_accept_ns']/cs['sensor_packet_accept_ns'],'dsp_speedup_cpp_vs_python':ps['dsp_window_us']/cs['dsp_window_us']},'raw':{'cpp20':cpp_runs,'python':py_runs}}
 a.out.parent.mkdir(parents=True,exist_ok=True);payload=json.dumps(result,indent=2)+'\n';a.out.write_text(payload);latest=ROOT/'benchmarks/results/latest.json';latest.write_text(payload);print(json.dumps(result,indent=2))
if __name__=='__main__':main()

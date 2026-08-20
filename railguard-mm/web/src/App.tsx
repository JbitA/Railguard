import React, {useEffect,useMemo,useState} from 'react';
import {LineChart,Line,CartesianGrid,XAxis,YAxis,Tooltip,Legend,ResponsiveContainer,AreaChart,Area} from 'recharts';

type Point={ts:string;sample_period_ms:number;vibration_rms:number;vibration_peak:number;vision_motion:number;vision_sharpness:number;speed_mps:number|null;temperature_c:number|null;humidity:number|null;packet_loss:number;spool_depth:number;spool_dropped?:number;camera_matched?:boolean;sync_error_ms?:number|null;sensor_skew_ms?:number;clock_alignment_locked?:boolean;clock_jitter_ms?:number;clock_samples?:number;context_flags?:number;sensor0_rms?:number;sensor1_rms?:number;sensor2_rms?:number};
type Pred={issued_at:string;target_ts:string;horizon_steps:number;step_ms:number;vibration_rms_pred:number;vision_motion_pred:number;anomaly_probability:number;model_version:string;measured_ts?:string|null;match_error_ms?:number|null;vibration_residual?:number|null};
const API=(import.meta as any).env.VITE_API_BASE || '/api';

export default function App(){
 const [device,setDevice]=useState('railguard-001'); const [minutes,setMinutes]=useState(30);
 const [measured,setMeasured]=useState<Point[]>([]); const [pred,setPred]=useState<Pred[]>([]); const [err,setErr]=useState('');
 const load=async()=>{try{const r=await fetch(`${API}/v1/series/${device}?minutes=${minutes}`); if(!r.ok)throw new Error(`${r.status}`); const d=await r.json(); setMeasured(d.measured||[]); setPred(d.predicted||[]); setErr('');}catch(e:any){setErr(String(e));}};
 useEffect(()=>{load(); const id=setInterval(load,3000); return()=>clearInterval(id);},[device,minutes]);
 const merged=useMemo(()=>{
   const m=new Map<string,any>(); measured.forEach(x=>m.set(x.ts,{...x,time:new Date(x.ts).toLocaleTimeString()}));
   pred.forEach(x=>{const k=x.target_ts; const row=m.get(k)||{ts:k,time:new Date(k).toLocaleTimeString()}; row.vibration_rms_pred=x.vibration_rms_pred; row.vision_motion_pred=x.vision_motion_pred; row.anomaly_probability=x.anomaly_probability; row.horizon_steps=x.horizon_steps; row.step_ms=x.step_ms; row.model_version=x.model_version; row.vibration_residual=x.vibration_residual; row.match_error_ms=x.match_error_ms; m.set(k,row);});
   return Array.from(m.values()).sort((a,b)=>a.ts.localeCompare(b.ts));
 },[measured,pred]);
 const latest=measured.length?measured[measured.length-1]:undefined; const latestPred=pred.length?pred[pred.length-1]:undefined;
 return <main>
   <header><div><h1>RailGuard-MM</h1><p>Multimodal railway condition monitoring</p></div><div className="controls"><input value={device} onChange={e=>setDevice(e.target.value)}/><select value={minutes} onChange={e=>setMinutes(Number(e.target.value))}><option value={5}>5 min</option><option value={30}>30 min</option><option value={120}>2 h</option></select></div></header>
   {err&&<div className="error">API error: {err}</div>}
   <section className="cards">
    <Card label="Vibration RMS" value={latest?`${latest.vibration_rms.toFixed(2)} m/s²`:'—'}/>
    <Card label="Predicted RMS" value={latestPred?`${latestPred.vibration_rms_pred.toFixed(2)} m/s²`:'—'}/>
    <Card label="Anomaly probability" value={latestPred?`${(100*latestPred.anomaly_probability).toFixed(1)} %`:'—'}/>
    <Card label="Forecast horizon" value={latestPred?`${(latestPred.horizon_steps*latestPred.step_ms).toFixed(0)} ms`:'—'}/>
    <Card label="Speed" value={latest?.speed_mps!=null?`${latest.speed_mps.toFixed(2)} m/s`:'—'}/>
    <Card label="Packet loss / spool" value={latest?`${latest.packet_loss} / ${latest.spool_depth}`:'—'}/>
    <Card label="Spool drops" value={latest?`${latest.spool_dropped??0}`:'—'}/>
    <Card label="Camera sharpness" value={latest?.vision_sharpness!=null?latest.vision_sharpness.toFixed(2):'—'}/>
    <Card label="Camera sync" value={latest?.camera_matched?(latest.sync_error_ms==null?'matched':`${latest.sync_error_ms.toFixed(1)} ms`):'unmatched'}/>
    <Card label="3-sensor skew" value={latest?.sensor_skew_ms!=null?`${latest.sensor_skew_ms.toFixed(1)} ms`:'—'}/>
    <Card label="Clock alignment" value={latest?.clock_alignment_locked?`locked · ${(latest.clock_jitter_ms??0).toFixed(1)} ms jitter`:'unlocked'}/>
    <Card label="Operating context" value={latest?.context_flags==null?'—':((latest.context_flags&3)===3?'GNSS + environment':`degraded · flags 0x${latest.context_flags.toString(16).padStart(2,'0')}`)}/>
   </section>
   <Panel title="Measured vs predicted vibration RMS"><ResponsiveContainer width="100%" height={300}><LineChart data={merged}><CartesianGrid strokeDasharray="3 3"/><XAxis dataKey="time" minTickGap={35}/><YAxis/><Tooltip/><Legend/><Line type="monotone" dataKey="vibration_rms" name="Measured RMS" dot={false}/><Line type="monotone" dataKey="vibration_rms_pred" name="Predicted RMS" dot={false}/></LineChart></ResponsiveContainer></Panel>
   <div className="grid2"><Panel title="Vision motion: measured vs predicted"><ResponsiveContainer width="100%" height={250}><LineChart data={merged}><CartesianGrid strokeDasharray="3 3"/><XAxis dataKey="time" hide/><YAxis/><Tooltip/><Legend/><Line type="monotone" dataKey="vision_motion" name="Measured motion" dot={false}/><Line type="monotone" dataKey="vision_motion_pred" name="Predicted motion" dot={false}/></LineChart></ResponsiveContainer></Panel>
   <Panel title="Anomaly probability"><ResponsiveContainer width="100%" height={250}><AreaChart data={merged}><CartesianGrid strokeDasharray="3 3"/><XAxis dataKey="time" hide/><YAxis domain={[0,1]}/><Tooltip/><Area type="monotone" dataKey="anomaly_probability" name="P(anomaly)"/></AreaChart></ResponsiveContainer></Panel></div>
   <div className="grid2"><Panel title="Forecast residual (nearest measured target)"><ResponsiveContainer width="100%" height={240}><LineChart data={pred.map(x=>({...x,time:new Date(x.target_ts).toLocaleTimeString()}))}><CartesianGrid strokeDasharray="3 3"/><XAxis dataKey="time" hide/><YAxis/><Tooltip/><Line type="monotone" dataKey="vibration_residual" name="Measured - predicted RMS" dot={false}/></LineChart></ResponsiveContainer></Panel>
   <Panel title="Synchronization quality"><ResponsiveContainer width="100%" height={240}><LineChart data={measured.map(x=>({...x,time:new Date(x.ts).toLocaleTimeString()}))}><CartesianGrid strokeDasharray="3 3"/><XAxis dataKey="time" hide/><YAxis/><Tooltip/><Legend/><Line type="monotone" dataKey="sync_error_ms" name="Camera ↔ sensor ms" dot={false}/><Line type="monotone" dataKey="sensor_skew_ms" name="Sensor skew ms" dot={false}/></LineChart></ResponsiveContainer></Panel></div>
   <div className="grid2"><Panel title="Spatial vibration RMS"><ResponsiveContainer width="100%" height={240}><LineChart data={measured.map(x=>({...x,time:new Date(x.ts).toLocaleTimeString()}))}><CartesianGrid strokeDasharray="3 3"/><XAxis dataKey="time" hide/><YAxis/><Tooltip/><Legend/><Line type="monotone" dataKey="sensor0_rms" name="Sensor 0" dot={false}/><Line type="monotone" dataKey="sensor1_rms" name="Sensor 1" dot={false}/><Line type="monotone" dataKey="sensor2_rms" name="Sensor 2" dot={false}/></LineChart></ResponsiveContainer></Panel>
   <Panel title="Edge health"><ResponsiveContainer width="100%" height={240}><LineChart data={measured.map(x=>({...x,time:new Date(x.ts).toLocaleTimeString()}))}><CartesianGrid strokeDasharray="3 3"/><XAxis dataKey="time" hide/><YAxis/><Tooltip/><Legend/><Line type="monotone" dataKey="packet_loss" name="Packet loss" dot={false}/><Line type="monotone" dataKey="spool_depth" name="Spool depth" dot={false}/><Line type="monotone" dataKey="spool_dropped" name="Spool drops" dot={false}/></LineChart></ResponsiveContainer></Panel></div>
   <Panel title="Operating context"><ResponsiveContainer width="100%" height={240}><LineChart data={measured.map(x=>({...x,time:new Date(x.ts).toLocaleTimeString()}))}><CartesianGrid strokeDasharray="3 3"/><XAxis dataKey="time" hide/><YAxis/><Tooltip/><Legend/><Line type="monotone" dataKey="speed_mps" name="Speed m/s" dot={false}/><Line type="monotone" dataKey="temperature_c" name="Temperature °C" dot={false}/><Line type="monotone" dataKey="humidity" name="Humidity" dot={false}/></LineChart></ResponsiveContainer></Panel>
   <footer>{latestPred?`Model: ${latestPred.model_version}`:'Waiting for predictions…'}</footer>
 </main>;
}
function Card({label,value}:{label:string,value:string}){return <div className="card"><span>{label}</span><strong>{value}</strong></div>}
function Panel({title,children}:{title:string,children:React.ReactNode}){return <section className="panel"><h2>{title}</h2>{children}</section>}

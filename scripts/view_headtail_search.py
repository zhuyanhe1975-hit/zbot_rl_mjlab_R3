import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import json,math,time,traceback,itertools,signal
import threading
from pathlib import Path
import numpy as np,mujoco
from types import SimpleNamespace
from scripts.reconfigure_mixed_full_cpu import make_config
import zbot_rl_mjlab.tasks.reconfiguration.full_cpu as mod
from zbot_rl_mjlab.tasks.reconfiguration.closure_states import ClosedRegistrationAudit
from zbot_rl_mjlab.tasks.reconfiguration.kinematic_rz270 import rz
out=Path('outputs/reconfiguration/headtail_order_feedback_2026_10_29')
original=mod.add_full_cpu_connectors

def spec_adapt(spec):
 original(spec)
 site=spec.site('module_5/port_out')
 site.parent.add_site(name='module_5/search_port',pos=site.pos,quat=site.quat,size=(.004,)*3)
mod.add_full_cpu_connectors=spec_adapt

def prepare(f,angle,q):
 m=f.model; old=m.site('module_5/port_out').id; new=m.site('module_5/search_port').id
 delta=math.radians(angle-180); turn=rz(delta)[:3,:3]; inlet=m.site_pos[m.site('module_0/port_in').id]
 R=np.empty(9);mujoco.mju_quat2Mat(R,m.site_quat[old]);R=R.reshape(3,3)
 m.site_pos[new]=m.site_pos[old]+R@(turn@inlet-inlet)
 quat=np.empty(4);mujoco.mju_mulQuat(quat,m.site_quat[old],np.array([math.cos(delta/2),0.,0.,math.sin(delta/2)]));m.site_quat[new]=quat
 m.eq_obj1id[f.old[5]]=new
 # Planner's open chain must include the same fixed head-tail clocking.
 orig_sync=f.planner.sync
 def sync(state,cut):
  anchor=orig_sync(state,cut); f.planner.kin.registration[5]=delta; return anchor
 f.planner.sync=sync
 f.closed_q=np.array(q);mujoco.mj_forward(m,f.data)
 f.phase_loads={};f.max_q=0.
 original_measure=f.measure
 def measure():
  original_measure(); active=f.connected().copy()
  if f.phase in ('LOAD','LOAD_HEADTAIL'):active[f.target]=False
  if active.any():
   vals=np.where(active,f.loads,-1.);idx=int(np.argmax(vals));value=float(vals[idx]);key=f.phase
   if key not in f.phase_loads or value>f.phase_loads[key]['load_N']:
    f.phase_loads[key]={'load_N':value,'face':idx+1,'time_s':float(f.data.time)}
  f.max_q=max(f.max_q,float(abs(f.data.qpos[f.qadr]).max()))
 f.measure=measure
 orig_apply=f.apply
 def apply():
  if f.phase=='CAPTURE_HEADTAIL' and f.step_count%round(f.cfg.control_dt/f.cfg.dt)==0:
   saved=f.new;f.new=np.append(saved,f.old[5])
   try:f.approach_feedback_update()
   finally:f.new=saved
  orig_apply()
 f.apply=apply


def reset_in_place(previous,cfg,branch_q):
 """Restore a fresh FSM while keeping the model and data used by the viewer."""
 fresh=mod.FullCpuReconfiguration(cfg)
 try:
  prepare(fresh,90,branch_q)
  model,data=previous.model,previous.data
  # These are the mutable model fields used by the FSM and the registered site.
  for name in ('eq_solref','eq_obj1id','pair_geom1','pair_geom2',
               'actuator_gainprm','actuator_biasprm','site_pos','site_quat'):
   getattr(model,name)[:]=getattr(fresh.model,name)
  mujoco.mj_copyData(data,model,fresh.data)
  fresh.model=model;fresh.data=data
 except BaseException:
  fresh.close()
  raise
 previous.close()
 return fresh


def main():
 import argparse
 p=argparse.ArgumentParser(description='Visualize the 90-degree head-tail reconfiguration search candidate')
 p.add_argument('--headless',action='store_true')
 args=p.parse_args()
 result=json.loads((out/'verification_90_3_1245.json').read_text())
 mod.ORDER=(0,1,3,4)
 # Increase trajectory speed by 30%, with acceleration scaled by speed squared.
 cfg=make_config(SimpleNamespace(speed_deg_s=105.*1.3,accel_deg_s2=980.*1.3**2))
 cfg.curl_speed_multiplier*=1.3
 f=mod.FullCpuReconfiguration(cfg)
 prepare(f,90,result['branch_q'])
 try:
  if args.headless:
   while f.phase not in ('DONE','FAILED') and f.data.time<60:f.step()
   print('RESULT',f.phase,'time',f.data.time,'phase_loads',f.phase_loads,flush=True)
   return 0 if f.phase=='DONE' else 1
  import mujoco.viewer
  restart=threading.Event()
  def on_key(key):
   if key in (13,257,335):restart.set()
  with mujoco.viewer.launch_passive(f.model,f.data,key_callback=on_key) as viewer:
   viewer.cam.lookat[:]=[0.,0.,.08];viewer.cam.distance=1.4;viewer.cam.elevation=-55
   anchor=time.perf_counter();next_status=0.
   while viewer.is_running():
    if restart.is_set():
     restart.clear()
     with viewer.lock():f=reset_in_place(f,cfg,result['branch_q'])
     anchor=time.perf_counter();next_status=0.
    if f.phase not in ('DONE','FAILED'):
     desired=time.perf_counter()-anchor;deadline=time.perf_counter()+.012
     while f.data.time<desired and time.perf_counter()<deadline and f.phase not in ('DONE','FAILED'):f.step()
    if time.perf_counter()>=next_status:
     peak=max((item['load_N'] for item in f.phase_loads.values()),default=0.)
     labels='Time\nPhase\nTarget face\nFace loads 1-6 (N)\nPeak other load (N)\nMin margin (N)\nConnected 1-6\nHead-tail angle\nRestart'
     values=(f'{f.data.time:.2f}s\n{f.phase}\n{f.target+1}\n'+ ' '.join(f'{v:.1f}' for v in f.loads)+'\n'+f'{peak:.2f}\n{100.-peak:.2f}\n'+ ' '.join('1' if v else '0' for v in f.connected())+'\n90 deg\nEnter')
     if f.failure:labels+='\nFailure';values+='\n'+f.failure[:120]
     viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100.value,mujoco.mjtGridPos.mjGRID_TOPRIGHT.value,labels,values));next_status=time.perf_counter()+.1
    viewer.sync();time.sleep(.002)
  return 0
 finally:f.close()

if __name__=='__main__':raise SystemExit(main())

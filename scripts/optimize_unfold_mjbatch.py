"""Batch-refine a recorded FSM unfold control reference with mjbatch's pool."""
import argparse,json,math,time
from pathlib import Path
import numpy as np
import mujoco
from scripts.optimize_headtail_mjbatch import lib_open,OUT,STATE


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--count',type=int,default=512);p.add_argument('--threads',type=int,default=8)
 p.add_argument('--max-speedup',type=float,default=2.);p.add_argument('--offset-rad',type=float,default=.008)
 p.add_argument('--directory',type=Path,default=OUT,help='directory containing unfold reference and model')
 p.add_argument('--pick-index',type=int,help='save this feasible candidate instead of the fastest')
 p.add_argument('--save-endstates',action='store_true',help='save candidate end states for multi-stage screening')
 args=p.parse_args()
 if args.count<2 or args.threads<1 or args.max_speedup<1 or args.offset_rad<0:p.error('invalid arguments')
 directory=args.directory
 with np.load(directory/'unfold_reference.npz') as x:state=x['state'].copy();ref=x['controls'].copy()
 model=mujoco.MjModel.from_binary_path(str(directory/'unfold_model.mjb'));ns=len(state)
 n=args.count;dt=float(model.opt.timestep);rng=np.random.default_rng(20261029)
 speedups=np.r_[1.,rng.uniform(1.,args.max_speedup,n-1)]
 offsets=np.vstack((np.zeros(6),rng.uniform(-args.offset_rad,args.offset_rad,(n-1,6))))
 lengths=(np.ceil(len(ref)/speedups).astype(np.int32)+250);maxsteps=int(lengths.max())
 controls=np.zeros((n,maxsteps,6));grid=np.arange(len(ref))
 for i in range(n):
  motion=int(lengths[i])-250;t=np.minimum(np.arange(motion)*speedups[i],len(ref)-1)
  envelope=np.sin(np.pi*t/max(len(ref)-1,1))**2
  for j in range(6):controls[i,:motion,j]=np.interp(t,grid,ref[:,j])+envelope*offsets[i,j]
  controls[i,motion:motion+250]=controls[i,motion-1]
 result=np.zeros((n,13));states=np.zeros((n,ns));metrics=np.zeros((n,4));lib=lib_open();ptr=lib.zbatch_create(str((directory/'unfold_model.mjb').resolve()).encode(),args.threads)
 if not ptr:raise RuntimeError('cannot create native batch')
 start=time.perf_counter()
 try:lib.zbatch_eval(ptr,state,n,maxsteps,lengths,controls,result,states,80.,metrics,0)
 finally:lib.zbatch_free(ptr)
 elapsed=time.perf_counter()-start
 # Compare one feasible rollout against independent serial MuJoCo.
 ids=np.flatnonzero(result[:,0]==0);difference=None;rows=[]
 qadr=np.array([model.joint(f'module_{i}/joint{i+1}').qposadr[0] for i in range(6)])
 dofs=np.array([model.joint(f'module_{i}/joint{i+1}').dofadr[0] for i in range(6)])
 acts=np.array([model.actuator(f'module_{i}/joint{i+1}').id for i in range(6)])
 data=mujoco.MjData(model)
 if len(ids):
  check=int(ids[0]);mujoco.mj_setState(model,data,state,STATE)
  for ctrl in controls[check,:lengths[check]]:data.ctrl[acts]=ctrl;mujoco.mj_step(model,data)
  final=np.empty(ns);mujoco.mj_getState(model,data,final,STATE);difference=float(np.max(np.abs(final-states[check])))
  if difference>1e-8:raise RuntimeError(f'batch/serial mismatch {difference}')
 for i in range(n):
  mujoco.mj_setState(model,data,states[i],STATE)
  qmax=float(np.max(np.abs(data.qpos[qadr])));vmax=float(np.max(np.abs(data.qvel[dofs])))
  rows.append({'index':i,'speedup':float(speedups[i]),'offset_rad':offsets[i].tolist(),
   'status':int(result[i,0]),'peak_load_N':float(result[i,1]),'max_torque_Nm':float(result[i,2]),
   'module_penetration_m':float(result[i,3]),'ground_penetration_m':float(result[i,4]),
   'failure_step':int(result[i,5]),'qmax':qmax,'vmax':vmax,'duration_s':float(lengths[i]*dt),
   'terminal_joint_ok':bool(qmax<.02 and vmax<.05)})
 candidates=[r for r in rows if r['status']==0 and r['terminal_joint_ok']];candidates.sort(key=lambda r:r['duration_s'])
 report={'method':'mjbatch upstream ThreadPool / native MuJoCo multi-step rollout','count':n,'threads':args.threads,
         'load_limit_N':80.,'elapsed_s':elapsed,'reference_steps':len(ref),'serial_state_difference':difference,
         'scope':'open-loop unfold screening; full state-machine validation still required','candidates':rows}
 (directory/'unfold_batch_search.json').write_text(json.dumps(report,indent=2)+'\n')
 if args.save_endstates:np.savez(directory/'unfold_endstates.npz',states=states,speedups=speedups,offsets=offsets,results=result,lengths=lengths)
 if candidates:
  best=candidates[0] if args.pick_index is None else next((x for x in candidates if x['index']==args.pick_index),None)
  if best is None:raise ValueError('requested candidate is not feasible')
  i=best['index'];np.savez(directory/'unfold_best.npz',control=controls[i,:lengths[i]],metadata=json.dumps(best))
 print('BATCH_UNFOLD',n,'elapsed_s',round(elapsed,3),'feasible_terminal',len(candidates),'serial_difference',difference,'best',json.dumps(candidates[:3]),flush=True)

if __name__=='__main__':main()

import json, numpy as np, mujoco
from scripts.view_native_reconfiguration import load_recording,set_frame
from zbot_mjbatch.native_reconfiguration import NativeReconfiguration
_,_,_,a=load_recording('outputs/reconfiguration/native_fsm/result.json');x=NativeReconfiguration();set_frame(x.model,x.data,a,40580);x.audit.sample(); response=x._static_force_jacobian(); print('jacobian', response is not None,flush=True)
m=x.model; initial=x.data; out=[]
for horizon in (20,200,1000):
 for joint in range(6):
  data=[]
  for sign in (-1,1):
   d=mujoco.MjData(m);mujoco.mj_copyData(d,m,initial);d.ctrl[x.acts[joint]]+=sign*.01;mujoco.mj_step(m,d,nstep=horizon);mujoco.mj_forward(m,d)
   data.append((x._contact_wrenches(d),d.actuator_force[x.acts].copy(),np.max(np.abs(d.qvel[x.dofs])),x._proxy_connections(d)))
  measured=(data[1][0]-data[0][0])/2; dtau=(data[1][1]-data[0][1])/2; predicted=np.einsum('fij,j->fi',response,dtau)
  out.append(dict(horizon_s=horizon*.001,joint=joint+1,dtau=dtau.tolist(),predicted=predicted.tolist(),measured=measured.tolist(),error_norm=float(np.linalg.norm(predicted-measured)),measured_norm=float(np.linalg.norm(measured)),max_speed=max(data[0][2],data[1][2]),contacts_both=[data[0][3].tolist(),data[1][3].tolist()]))
  print(horizon,joint+1,'error',round(out[-1]['error_norm'],4),'measured',round(out[-1]['measured_norm'],4),'speed',round(out[-1]['max_speed'],4),flush=True)
print('SUMMARY',json.dumps(out),flush=True);x.close()

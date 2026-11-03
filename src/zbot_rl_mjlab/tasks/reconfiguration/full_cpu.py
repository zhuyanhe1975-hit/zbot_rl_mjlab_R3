"""Native CPU MuJoCo: straight -> closed -> mixed registrations -> straight."""
from dataclasses import dataclass
import math,time
from types import SimpleNamespace as NS
from concurrent.futures import ThreadPoolExecutor
from threading import local
import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize,least_squares
from mjlab.scene import Scene
from .reference_dynamic import reference_env_cfg,add_reference_connectors
from .reference_trajectory import AxialPlanner,TimedCurve,clearance_lower_bound
from .kinematic_rz270 import inverse,Frame
from .live_loading import response,peel
from .static_loading_milp import solve_static_loading
from .batched_loading import projected_increments,peel as torch_peel

def headtail_projected_increments(A, w, torque, target, radius, iterations=20,
                                  other_limit=60.,increment=.05,load_step=1.,goal_limit=84.):
    """CPU tensor projections with the signed compression term retained."""
    b = len(w); dev=w.device; dtype=w.dtype
    angles=torch.arange(16,device=dev,dtype=dtype)*(2*torch.pi/16)
    axis=torch.stack((angles.cos(),angles.sin()),-1).repeat_interleave(2,0)
    tension=torch.arange(32,device=dev,dtype=dtype).remainder(2)
    worlds=torch.arange(b,device=dev)
    At=A[worlds,target];wt=w[worlds,target]
    row=tension[None,:,None]*At[:,None,2]+torch.einsum('ca,baj->bcj',axis,At[:,3:5])/radius
    value=tension[None]*wt[:,None,2]+torch.einsum('ca,ba->bc',axis,wt[:,3:5])/radius
    current=torch_peel(w,radius);goal=(current[worlds,target]+load_step).clamp(12.,goal_limit)
    low=torch.maximum(torch.full_like(torque,-increment),-14.-torque)[:,None]
    high=torch.minimum(torch.full_like(torque,increment),14.-torque)[:,None]
    delta=(row*(goal[:,None]-value)[:,:,None]/row.square().sum(-1,keepdim=True).clamp_min(1e-12)).clamp(low,high)
    other=(torch.arange(6,device=dev)[None]!=target[:,None])
    for _ in range(iterations):
        estimated=w[:,None]+torch.einsum('bfwa,bca->bcfw',A,delta)
        loads=torch_peel(estimated,radius)
        bend=estimated[...,3:5]/estimated[...,3:5].norm(dim=-1,keepdim=True).clamp_min(1e-8)
        gradient=(estimated[...,2]>0.)[:,:,:,None]*A[:,None,:,2]+torch.einsum('bcfi,bfij->bcfj',bend,A[:,:,3:5])/radius
        violation=(loads-other_limit).clamp_min(0.)*other[:,None]
        correction=(violation[:,:,:,None]*gradient/gradient.square().sum(-1,keepdim=True).clamp_min(1e-10)).sum(2)
        shortfall=(goal[:,None]-value-(row*delta).sum(-1)).clamp_min(0.)
        correction-=shortfall[:,:,None]*row/row.square().sum(-1,keepdim=True).clamp_min(1e-10)
        count=(violation>0.).sum(-1)+(shortfall>0.)
        delta=(delta-correction/count.clamp_min(1)[:,:,None]).clamp(low,high)
    estimated=w[:,None]+torch.einsum('bfwa,bca->bcfw',A,delta)
    loads=torch_peel(estimated,radius)
    safe_end=torch.where(other[:,None],loads,0.).amax(-1)<=other_limit+1e-4
    safe_start=torch.where(other,current,0.).amax(-1)<=other_limit
    left=torch.zeros_like(value);right=torch.ones_like(value)
    response=torch.einsum('bfwa,bca->bcfw',A,delta)
    for _ in range(10):
        middle=(left+right)*.5
        trial=torch_peel(w[:,None]+response*middle[:,:,None,None],radius)
        safe_middle=torch.where(other[:,None],trial,0.).amax(-1)<=other_limit
        left=torch.where(safe_middle,middle,left);right=torch.where(safe_middle,right,middle)
    scale=torch.where(~safe_end&safe_start[:,None],left,1.)
    delta=delta*scale[:,:,None]
    estimated=w[:,None]+torch.einsum('bfwa,bca->bcfw',A,delta)
    loads=torch_peel(estimated,radius)
    safe=(torch.where(other[:,None],loads,0.).amax(-1)<=other_limit+1e-4)&torch.isfinite(delta).all(-1)
    full=safe&((value+(row*delta).sum(-1))>=goal[:,None]-1e-3)
    objective=delta.square().sum(-1)+.002*(torque[:,None]+delta).square().sum(-1)+.00002*torch.where(other[:,None],loads.square(),0.).sum(-1)
    best_full=torch.where(full,objective,torch.inf).argmin(-1)
    target_load=loads.gather(2,target[:,None,None].expand(-1,32,1)).squeeze(-1)
    best_partial=torch.where(safe,target_load,-torch.inf).argmax(-1)
    choice=torch.where(full.any(-1),best_full,best_partial)
    selected=delta[worlds,choice]
    improves=target_load[worlds,choice]>current[worlds,target]+1e-4
    usable=full.any(-1)|(safe.any(-1)&improves)
    retreat=(-torque).clamp(-increment,increment)
    retreat_load=torch_peel(w+torch.einsum('bfwa,ba->bfw',A,retreat),radius)
    old_peak=torch.where(other,current,0.).amax(-1)
    better=torch.where(other,retreat_load,0.).amax(-1)<old_peak
    fallback=torch.where(better[:,None],retreat,0.)
    result=torch.where(usable[:,None],selected,fallback)
    return result, usable, torch_peel(w+torch.einsum('bfwa,ba->bfw',A,result),radius)



EXPERIMENT_ID='Zbot-Straight-Closed-Mixed-Straight-CPU-v1'
ORDER=(0,1,3,4)


def add_full_cpu_connectors(spec):
    add_reference_connectors(spec)
    # The first five mating half-body pairs are excluded by add_connectors.
    # Add the head-tail pair too; an explicit pair can be enabled for rotation.
    spec.add_exclude(bodyname1='module_5/foot_1',bodyname2='module_0/foot_0')


@dataclass
class FullCpuConfig:
    dt:float=.001
    control_dt:float=.02
    speed_deg_s:float=45.
    acceleration_deg_s2:float=180.
    curl_speed_multiplier:float=1.
    curl_acceleration_deg_s2:float|None=None
    outer_speed_deg_s:float=30.
    outer_acceleration_deg_s2:float=90.
    stiffness:float=400.
    damping:float=4.
    outer_stiffness:float=80.
    outer_damping:float=4.
    break_load:float=80.
    other_stop_load:float=64.
    one_shot_loading:bool=False
    loading_ramp_s:float=.5
    post_break_hold_s:float=0.
    headtail_break_hold_s:float|None=None
    loading_model_headroom:float=0.
    headtail_model_horizon_s:float=.5
    headtail_model_headroom:float|None=None
    loading_other_prediction_limit:float=56.
    loading_goal_margin:float=4.
    continuous_torque:float=14.
    peak_torque:float=20.
    overload_seconds:float=1.
    workspace_radius:float=3.
    headtail_plan_load:float=84.
    headtail_prediction_limit:float=60.
    headtail_increment:float=.05
    headtail_load_step:float=1.
    headtail_one_step_prediction_limit:float=56.
    headtail_one_step_validation_limit:float=62.
    headtail_validation_margin:float=.5
    unfold_prediction_limit:float=54.
    unfold_target_step:float=.008
    unfold_target_step_max:float=.024
    unfold_accel_delay_s:float=2.
    unfold_accel_duration_s:float=8.
    unfold_timeout_s:float=120.
    settle_seconds:float=1.
    initial_hold_seconds:float=1.
    motion_wait_seconds:float=4.
    capture_position_tolerance:float=.002
    docking_stiffness:float=0.
    approach_feedback:bool=False
    capture_angle_tolerance_deg:float=2.
    capture_linear_speed:float=.03
    capture_angular_speed:float=.3
    separation_gap:float=.051
    overlap_rotation:bool=False
    headtail_parallel_gap:float=0.
    closest_unfold:bool=False
    closest_unfold_gain:float=1.5
    closest_unfold_max_speed:float=1.2
    closest_unfold_ramp_s:float=0.
    direct_unfold:bool=False
    staged_unfold:bool=False
    staged_unfold_gap:float=.10
    unfold_speed_deg_s:float=150.
    unfold_acceleration_deg_s2:float=600.


class FullCpuReconfiguration:
    def __init__(self,cfg=None):
        self.cfg=cfg or FullCpuConfig();torch.set_num_threads(1)
        base=reference_env_cfg();scene_cfg=base.scene;scene_cfg.spec_fn=add_full_cpu_connectors
        self.scene=Scene(scene_cfg,device='cpu');self.model=self.scene.compile()
        base.sim.mujoco.apply(self.model);self.model.opt.timestep=self.cfg.dt
        self.data=mujoco.MjData(self.model)
        m=self.model;self.planner=AxialPlanner(self.cfg.separation_gap,self.cfg.overlap_rotation);k=self.planner.kin
        self.qadr=np.array([m.joint(f'module_{i}/joint{i+1}').qposadr[0] for i in range(6)])
        self.dofs=np.array([m.joint(f'module_{i}/joint{i+1}').dofadr[0] for i in range(6)])
        self.roots=np.array([m.joint(f'module_{i}/module_root').qposadr[0] for i in range(6)])
        self.root_dofs=np.array([m.joint(f'module_{i}/module_root').dofadr[0] for i in range(6)])
        self.root_bodies=np.array([m.jnt_bodyid[m.joint(f'module_{i}/module_root').id] for i in range(6)])
        self.old=np.array([m.equality(f'connector_{i}').id for i in range(6)])
        self.new=np.array([m.equality(f'connector_{i}_rz270').id for i in range(5)])
        self.out=np.array([m.site(f'module_{i}/port_out').id for i in range(6)])
        self.inlet=np.array([m.site(f'module_{i}/port_in').id for i in range(6)])
        self.actuators=np.array([m.actuator(f'module_{i}/joint{i+1}').id for i in range(6)])
        self.pairs=np.array([m.pair(f'open_face_{i}').id for i in range(6)])
        self.pair_geoms=np.stack((m.pair_geom1[self.pairs],m.pair_geom2[self.pairs]),-1).copy()
        self.ground=int(np.flatnonzero(m.geom_type==mujoco.mjtGeom.mjGEOM_PLANE)[0])
        names=[m.body(int(b)).name for b in m.geom_bodyid]
        self.geom_module=np.array([int(n.split('/')[0][7:]) if n.startswith('module_') else -1 for n in names])
        self.phase='INITIAL_STRAIGHT_HOLD';self.phase_start=0.;self.step_count=0;self.stage=0;self.target=5
        self.events=[];self.history=[];self.loading_records=[];self.path_records=[];self.failure='';self.curve=None;self.curves={}
        self.qref=np.zeros(6);self.vref=np.zeros(6);self.loading_torque=np.zeros(6);self.capture=0.;self.overload=np.zeros(6,dtype=int)
        self.loads=np.zeros(6);self.force=np.zeros((6,3));self.moment=np.zeros((6,3));self.peak_load=np.zeros(6)
        self.peak_other=0.;self.peak_torque=0.;self.penetration=0.;self.ground_penetration=0.;self.clearance_cache={}
        self.worst_module_pair=None;self.worst_ground_pair=None
        self.loading_mode=False;self.loading_hold=False;self.loading_qhold=np.zeros(6);self.last_loading_step=-20;self.last_cpu_control=-20;self.next_log=0.;self.next_progress=0.;self.trace=[]
        self.closed_q=k.qref.copy();self.headtail_pool=None;self.headtail_worker=local()
        # Only initialization writes physical root poses/qpos.
        roots,_,_=k.fk(np.zeros(6),5,k.roots[0].copy())
        # Level the straight backbone so all six modules establish ground
        # support together, rather than dropping an inclined cantilever.
        axis=roots[-1,:3,3]-roots[0,:3,3];axis/=np.linalg.norm(axis)
        up=np.array([0.,0.,1.]);up-=axis*np.dot(up,axis);up/=np.linalg.norm(up)
        rigid=np.stack((axis,np.cross(up,axis),up));origin=roots[0,:3,3].copy()
        for root in roots:
            root[:3,3]=rigid@(root[:3,3]-origin)+origin;root[:3,:3]=rigid@root[:3,:3]
        for i,adr in enumerate(self.roots):
            self.data.qpos[adr:adr+3]=roots[i,:3,3]
            q=np.empty(4);mujoco.mju_mat2Quat(q,roots[i,:3,:3].reshape(9));self.data.qpos[adr+3:adr+7]=q
        self.data.qpos[self.qadr]=0.;self.data.qvel[:]=0.;self.data.eq_active[:]=False;self.data.eq_active[self.old[:5]]=True
        self.set_contact_phase();mujoco.mj_forward(m,self.data)
        # Orient the transverse part of the alternating hinge axes vertically:
        # bending then starts in the ground plane instead of lifting a long arm.
        axis=self.data.xaxis[m.joint('module_0/joint1').id]
        roll=Rotation.from_rotvec([math.atan2(axis[1],axis[2]),0.,0.]).as_matrix()
        origin=self.data.qpos[self.roots[0]:self.roots[0]+3].copy()
        for adr in self.roots:
            self.data.qpos[adr:adr+3]=roll@(self.data.qpos[adr:adr+3]-origin)+origin
            R=np.empty(9);mujoco.mju_quat2Mat(R,self.data.qpos[adr+3:adr+7]);q=np.empty(4)
            mujoco.mju_mat2Quat(q,(roll@R.reshape(3,3)).reshape(9));self.data.qpos[adr+3:adr+7]=q
        mujoco.mj_forward(m,self.data)
        low=np.inf
        for g in range(m.ngeom):
            if self.geom_module[g]<0 or m.geom_type[g]!=mujoco.mjtGeom.mjGEOM_MESH:continue
            mesh=m.geom_dataid[g];a=m.mesh_vertadr[mesh];n=m.mesh_vertnum[mesh]
            verts=m.mesh_vert[a:a+n]@self.data.geom_xmat[g].reshape(3,3).T+self.data.geom_xpos[g]
            low=min(low,float(verts[:,2].min()))
        self.data.qpos[self.roots+2]+=-.0003-low
        self.set_pd();mujoco.mj_forward(m,self.data)
        self.initial_qpos=self.data.qpos.copy();self.events.append({'time_s':0.,'to':self.phase,'connected':self.connected().tolist()})

    def ids(self):
        ids=self.old.copy();ids[:5]=np.where(self.data.eq_active[self.new],self.new,ids[:5]);return ids

    def connected(self):return self.data.eq_active[self.ids()].astype(bool)

    @property
    def phase_time(self):return self.data.time-self.phase_start

    def transition(self,phase):
        old=self.phase;elapsed=self.phase_time;self.phase=phase;self.phase_start=float(self.data.time);self.capture=0.
        event={'time_s':float(self.data.time),'from':old,'to':phase,'elapsed_s':elapsed,
               'connection':self.target+1,'connected':self.connected().tolist()}
        if phase in ('SETTLE_HEADTAIL','SETTLE'):
            gap,angle,v,w=self.geometry(new_weld=True)
            event.update(capture_gap_m=gap,capture_angle_deg=math.degrees(angle),
                         capture_linear_speed_m_s=v,capture_angular_speed_rad_s=w)
        self.events.append(event)
        print(f'CPU_FSM t={self.data.time:.3f} {old} -> {phase} target={self.target+1}',flush=True)
        self.set_contact_phase()

    def stop(self,reason):
        if self.phase=='FAILED':return
        self.failure=reason;self.loading_mode=False;self.transition('FAILED')
        self.model.actuator_gainprm[self.actuators,0]=1.;self.model.actuator_biasprm[self.actuators,1]=0.
        self.model.actuator_biasprm[self.actuators,2]=-1.5;self.data.ctrl[self.actuators]=0.

    def set_contact_phase(self):
        # Explicit pairs bypass the static flange exclusions. A plane-plane pair
        # has no collision function, so it produces no contact; restore the real
        # opened pair during rotation and during final head-tail unfolding.
        self.model.pair_geom1[self.pairs]=self.ground;self.model.pair_geom2[self.pairs]=self.ground
        if self.phase=='ROTATE' or (self.phase=='SEPARATE' and self.cfg.overlap_rotation) or (self.target==5 and self.phase in ('HEADTAIL_SEPARATE','HEADTAIL_BREAK_HOLD','UNFOLD_MIDDLE','UNFOLD','FINAL_STRAIGHT_HOLD')):
            self.model.pair_geom1[self.pairs[self.target]]=self.pair_geoms[self.target,0]
            self.model.pair_geom2[self.pairs[self.target]]=self.pair_geoms[self.target,1]

    def set_pd(self,outer=False):
        self.loading_mode=False;m=self.model
        self.kp=self.cfg.outer_stiffness if outer else self.cfg.stiffness
        self.kd=self.cfg.outer_damping if outer else self.cfg.damping
        m.actuator_gainprm[self.actuators,0]=self.kp
        m.actuator_biasprm[self.actuators,1]=-self.kp;m.actuator_biasprm[self.actuators,2]=-self.kd

    def start_load(self,phase,target):
        self.target=target;self.loading_torque[:]=0.;self.last_loading_step=-20
        self.loading_hold=target==5
        self.loading_damping=4. if target==5 else 1.5
        if self.loading_hold:
            # The final mixed closed assembly has a compliant/self-motion
            # direction. Retain native PD while adding loading feedforward.
            self.loading_qhold=self.data.qpos[self.qadr].copy();self.set_pd()
            self.data.ctrl[self.actuators]=self.loading_qhold
        else:
            m=self.model;m.actuator_gainprm[self.actuators,0]=1.;m.actuator_biasprm[self.actuators,1]=0.;m.actuator_biasprm[self.actuators,2]=-self.loading_damping
            self.data.ctrl[self.actuators]=0.
        self.headtail_fast_attempted=False;self.loading_solution=None;self.loading_nullspace=np.empty((0,6))
        if target==5:
            k=self.planner.kin;anchor=self.planner.sync(self.shim(),5);q=k.q.copy();h=1e-6
            J=np.stack([(k.errors(q+h*np.eye(6)[j],5,anchor,np.eye(3),np.zeros(3))-
                         k.errors(q-h*np.eye(6)[j],5,anchor,np.eye(3),np.zeros(3)))/(2*h) for j in range(6)],-1)
            _,singular,Vt=np.linalg.svd(J);self.loading_nullspace=Vt[singular<singular[0]*1e-3]
            self.path_records.append({'phase':'HEADTAIL_LOADING','closure_singular_values':singular.tolist(),
                                      'weak_motion_directions':self.loading_nullspace.tolist()})
        self.loading_mode=True;self.transition(phase)

    def curve_from(self,points):
        acceleration = math.radians(self.cfg.acceleration_deg_s2) if getattr(self.cfg,'acceleration_limit_enabled',True) else None
        return TimedCurve(points,math.radians(self.cfg.speed_deg_s),acceleration,self.cfg.dt)

    def start_curve(self,phase,goal):
        factor=self.cfg.curl_speed_multiplier if phase=='CURL' else 1.
        acceleration=(self.cfg.curl_acceleration_deg_s2 if phase=='CURL' and self.cfg.curl_acceleration_deg_s2 is not None else self.cfg.outer_acceleration_deg_s2*factor*factor)
        acceleration = math.radians(acceleration) if getattr(self.cfg,'acceleration_limit_enabled',True) else None
        self.curve=TimedCurve(np.stack((self.data.qpos[self.qadr].copy(),goal)),math.radians(self.cfg.outer_speed_deg_s*factor),acceleration,self.cfg.dt)
        self.set_pd(outer=True);self.transition(phase)

    def start_unfold(self):
        cfg=self.cfg
        if cfg.closest_unfold:
            self.set_pd(outer=True);self.qref=self.data.qpos[self.qadr].copy();self.vref[:]=0.;self.transition('UNFOLD');return
        if not cfg.direct_unfold:
            self.start_curve('UNFOLD',np.zeros(6));return
        current=self.data.qpos[self.qadr].copy()
        goal=current.copy() if cfg.staged_unfold else np.zeros(6)
        if cfg.staged_unfold:goal[2:4]=0.
        acceleration=math.radians(cfg.unfold_acceleration_deg_s2) if getattr(cfg,'acceleration_limit_enabled',True) else None
        self.curve=TimedCurve(np.stack((current,goal)),math.radians(cfg.unfold_speed_deg_s),acceleration,cfg.dt)
        self.set_pd(outer=True);self.transition('UNFOLD_MIDDLE' if cfg.staged_unfold else 'UNFOLD')

    def geometry(self,face=None,new_weld=False):
        i=self.target if face is None else face;d=self.data;m=self.model
        if new_weld:
            eq=self.new[i] if i<5 else self.old[5];s0=int(m.eq_obj1id[eq]);s1=int(m.eq_obj2id[eq]);goal=np.eye(3)
        else:
            s0=self.out[i];s1=self.inlet[(i+1)%6];angle=0. if i==5 else (-math.pi/2 if i<2 else math.pi/2)
            goal=Rotation.from_euler('z',angle).as_matrix()
        R0=d.site_xmat[s0].reshape(3,3);R1=d.site_xmat[s1].reshape(3,3)
        gap=float(np.linalg.norm(d.site_xpos[s0]-d.site_xpos[s1]));angle=float(np.linalg.norm(Rotation.from_matrix(goal.T@R1.T@R0).as_rotvec()))
        velocity=[]
        for s in (s0,s1):
            v=np.zeros(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_SITE,int(s),v,0);velocity.append(v)
        dv=velocity[1]-velocity[0]
        return gap,angle,float(np.linalg.norm(dv[3:])),float(np.linalg.norm(dv[:3]))

    def measure(self):
        m=self.model;d=self.data;ids=self.ids();active=self.connected();self.force[:]=0.;self.moment[:]=0.
        J=d.efc_J.reshape(d.nefc,m.nv)
        for i,eq in enumerate(ids):
            if not active[i]:continue
            rows=np.flatnonzero((d.efc_type==0)&(d.efc_id==eq))
            if len(rows)!=6:self.stop(f'missing constraint rows on face {i+1}');return
            g=J[rows,self.root_dofs[i]:self.root_dofs[i]+6].T@d.efc_force[rows]
            R=d.site_xmat[self.out[i]].reshape(3,3);rootR=d.xmat[self.root_bodies[i]].reshape(3,3)
            arm=d.site_xpos[self.out[i]]-d.xpos[self.root_bodies[i]]
            self.force[i]=R.T@g[:3];self.moment[i]=R.T@(rootR@g[3:]-np.cross(arm,g[:3]))
        self.loads=peel(np.c_[self.force,self.moment]);self.peak_load=np.maximum(self.peak_load,self.loads)
        other=np.arange(6)!=self.target;self.peak_other=max(self.peak_other,float(self.loads[other].max()))
        bad=np.flatnonzero(active&(self.loads>=self.cfg.break_load))
        if len(bad):
            self.data.eq_active[ids[bad]]=False
            self.events.append({'time_s':float(d.time),'event':'passive_break','faces':(bad+1).tolist(),'loads_N':self.loads.tolist()})
            if any(i!=self.target for i in bad):self.stop(f'non-target rupture: {(bad+1).tolist()}');return
            # Stop loading on the rupture substep, not at the next 20ms FSM tick.
            self.qref=self.data.qpos[self.qadr].copy();self.vref[:]=0.
            self.set_pd(outer=self.phase=='LOAD_HEADTAIL')
        if np.any(self.loads[other]>self.cfg.other_stop_load):self.stop(f'other load protection: {self.loads.tolist()}');return
        actual=d.qfrc_actuator[self.dofs];self.peak_torque=max(self.peak_torque,float(abs(actual).max()))
        self.overload=np.where(abs(actual)>self.cfg.continuous_torque,self.overload+1,0)
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():self.stop('nonfinite dynamics');return
        roots=d.xpos[self.root_bodies]
        if np.any(np.linalg.norm(roots[:,:2],axis=1)>self.cfg.workspace_radius) or np.any((roots[:,2]<-.2)|(roots[:,2]>2.)):
            self.stop('workspace protection');return
        if np.any(abs(actual)>self.cfg.peak_torque+1e-6) or np.any(self.overload>=math.ceil(self.cfg.overload_seconds/self.cfg.dt)):
            self.stop('actual motor torque protection');return
        for contact in d.contact:
            if contact.exclude!=0:continue
            x,y=int(contact.geom1),int(contact.geom2);a,b=self.geom_module[[x,y]];depth=max(0.,-float(contact.dist))
            if a>=0 and b>=0 and a!=b and depth>self.penetration:
                self.penetration=depth;self.worst_module_pair=[m.geom(x).name,m.geom(y).name]
            if (a>=0)!=(b>=0) and depth>self.ground_penetration:
                self.ground_penetration=depth;self.worst_ground_pair=[m.geom(x).name,m.geom(y).name]
        if self.penetration>.002 or self.ground_penetration>.003:self.stop('collision penetration protection')
        if not self.connected()[:5].all() and self.phase not in ('LOAD','POST_BREAK_HOLD','SEPARATE','ROTATE','APPROACH','SETTLE'):
            self.stop('internal chain disconnected')

    def rollout_wrenches(self, controls, nstep=None):
        if nstep is None:nstep=max(1,round(.05/self.cfg.dt))
        # Independent MuJoCo CPU rollouts; live state is never written.
        if self.headtail_pool is None:
            self.headtail_pool=ThreadPoolExecutor(max_workers=8,thread_name_prefix='zbot-load')
        m=self.model;ids=self.ids()
        def evaluate(i):
            if getattr(self.headtail_worker,'model',None) is not m:
                self.headtail_worker.model=m;self.headtail_worker.data=mujoco.MjData(m)
            d=self.headtail_worker.data;mujoco.mj_copyData(d,m,self.data)
            d.ctrl[self.actuators]=controls[i]
            mujoco.mj_step(m,d,nstep=nstep)
            J=d.efc_J.reshape(d.nefc,m.nv);w=[]
            for j,eq in enumerate(ids):
                rows=np.flatnonzero((d.efc_type==0)&(d.efc_id==eq))
                g=J[rows,self.root_dofs[j]:self.root_dofs[j]+6].T@d.efc_force[rows]
                R=d.site_xmat[self.out[j]].reshape(3,3);rootR=d.xmat[self.root_bodies[j]].reshape(3,3)
                arm=d.site_xpos[self.out[j]]-d.xpos[self.root_bodies[j]]
                w.append(np.r_[R.T@g[:3],R.T@(rootR@g[3:]-np.cross(arm,g[:3]))])
            return np.asarray(w)
        return np.asarray(list(self.headtail_pool.map(evaluate,range(len(controls)))))

    def headtail_impulse(self):
        """Choose a bounded torque whose actual next substep breaks only face 6."""
        A,w=response(self.model,self.data,self.ids());candidates=[]
        for angle in np.arange(16)*2*np.pi/16:
            axis=np.array([math.cos(angle),math.sin(angle)])
            for tension in (0.,1.):
                row=tension*A[5,2]+axis@A[5,3:5]/.05
                value=tension*w[5,2]+axis@w[5,3:5]/.05
                def inequalities(torque):
                    return np.r_[self.cfg.headtail_one_step_prediction_limit-peel(w+A@torque)[:5],
                                 value+row@torque-self.cfg.headtail_plan_load]
                constraints=[{'type':'ineq','fun':inequalities}]
                fit=minimize(lambda torque:np.dot(torque,torque),np.zeros(6),method='SLSQP',
                             bounds=[(-14.,14.)]*6,constraints=constraints,
                             options={'maxiter':150,'ftol':1e-9})
                if fit.success and inequalities(fit.x).min()>-1e-4:
                    candidates.append(fit.x)
        if not candidates:return None
        # Contact modes can change with the full candidate. Test scaled copies
        # against the actual MuJoCo solver before issuing any live torque.
        candidates=[candidate*scale for candidate in candidates for scale in (1.,.875,.75,.625)]
        controls=self.loading_qhold+np.asarray(candidates)/self.kp
        predicted=self.rollout_wrenches(controls,nstep=1)
        ranked=[]
        for torque,wrench in zip(candidates,predicted):
            loads=peel(wrench)
            demand=self.kp*(self.loading_qhold-self.data.qpos[self.qadr])-self.kd*self.data.qvel[self.dofs]+torque
            if (loads[5]>=self.cfg.break_load+self.cfg.headtail_validation_margin and loads[:5].max()<=self.cfg.headtail_one_step_validation_limit
                and abs(demand).max()<self.cfg.peak_torque):
                ranked.append((float(torque@torque),torque,loads))
        return min(ranked,key=lambda item:item[0]) if ranked else None

    def headtail_response(self,horizon_s=.05):
        # Differentiate the actual native-PD + constrained dynamics response,
        # including compliance and the weak closed-loop motion direction.
        epsilon=.05
        torques=np.repeat(self.loading_torque[None],13,axis=0)
        torques[1:7]+=np.eye(6)*epsilon;torques[7:]-=np.eye(6)*epsilon
        controls=self.loading_qhold+torques/self.kp if self.loading_hold else torques
        samples=self.rollout_wrenches(controls,nstep=max(1,round(horizon_s/self.cfg.dt)))
        A=((samples[1:7]-samples[7:])/(2*epsilon)).transpose(1,2,0)
        return A,samples[0]

    def unfold_update(self):
        # Preserve a safety margin while choosing each native-PD position
        # target toward the straight pose. Other joints may support the motion.
        q=self.data.qpos[self.qadr].copy();epsilon=.002
        controls=np.repeat(q[None],13,axis=0)
        controls[1:7]+=np.eye(6)*epsilon;controls[7:]-=np.eye(6)*epsilon
        samples=self.rollout_wrenches(controls)
        A=((samples[1:7]-samples[7:])/(2*epsilon)).transpose(1,2,0);w=samples[0]
        baseline=float(peel(w)[:5].max());limit=max(self.cfg.unfold_prediction_limit,baseline)
        if limit>=self.cfg.other_stop_load-.25:
            self.stop(f'unfold prediction exceeds safety margin: {baseline:.3f}N');return
        def constraint(delta):return limit-peel(w+A@delta)[:5]
        ramp=np.clip((self.phase_time-self.cfg.unfold_accel_delay_s)/self.cfg.unfold_accel_duration_s,0.,1.)
        step=self.cfg.unfold_target_step+ramp*(self.cfg.unfold_target_step_max-self.cfg.unfold_target_step)
        bounds=[(-step,step)]*6
        fit=minimize(lambda delta:float(np.dot(q+delta,q+delta)),np.zeros(6),method='SLSQP',bounds=bounds,
                     constraints=[{'type':'ineq','fun':constraint}],options={'maxiter':60,'ftol':1e-10})
        if not fit.success or constraint(fit.x).min()<-1e-5:
            if not self.cfg.overlap_rotation:
                self.stop(f'unfold safety optimization failed: {fit.message}');return
            # A failed numerical solve may still yield a feasible improving step.
            # Otherwise hold and retry on the next control tick.
            delta=fit.x if np.isfinite(fit.x).all() and constraint(fit.x).min()>=-1e-5 and np.linalg.norm(q+fit.x)<np.linalg.norm(q) else np.zeros(6)
        else:delta=fit.x
        self.qref=q+delta;self.vref[:]=0.

    def close(self):
        if self.headtail_pool is not None:self.headtail_pool.shutdown();self.headtail_pool=None

    def validate_loading_solution(self, torque):
        """Predict ramp on a copied MuJoCo state before applying any live load."""
        m=self.model;d=mujoco.MjData(m);mujoco.mj_copyData(d,m,self.data)
        ids=self.ids();other=np.arange(6)!=self.target;other_peak=0.
        ramp=self.cfg.loading_ramp_s
        steps=max(1,math.ceil((ramp+.5)/self.cfg.dt))
        for step in range(steps):
            scale=min(1.,step*self.cfg.dt/max(ramp,1e-9));applied=torque*scale
            d.ctrl[self.actuators]=self.loading_qhold+applied/self.kp if self.loading_hold else applied
            mujoco.mj_step(m,d)
            J=d.efc_J.reshape(d.nefc,m.nv);w=[]
            for i,eq in enumerate(ids):
                rows=np.flatnonzero((d.efc_type==0)&(d.efc_id==eq))
                if len(rows)!=6:return {'success':False,'reason':'constraint rows missing'}
                g=J[rows,self.root_dofs[i]:self.root_dofs[i]+6].T@d.efc_force[rows]
                R=d.site_xmat[self.out[i]].reshape(3,3);rootR=d.xmat[self.root_bodies[i]].reshape(3,3)
                arm=d.site_xpos[self.out[i]]-d.xpos[self.root_bodies[i]]
                w.append(np.r_[R.T@g[:3],R.T@(rootR@g[3:]-np.cross(arm,g[:3]))])
            loads=peel(np.asarray(w));other_peak=max(other_peak,float(loads[other].max()))
            if not np.isfinite(loads).all() or not np.isfinite(d.qpos).all():return {'success':False,'reason':'nonfinite rollout'}
            if loads[other].max()>=self.cfg.break_load:return {'success':False,'reason':'non-target overload in validation','other_peak_N':other_peak}
            if np.max(np.abs(d.qfrc_actuator[self.dofs]))>self.cfg.peak_torque+1e-6:return {'success':False,'reason':'motor overload in validation','other_peak_N':other_peak}
            if loads[self.target]>=self.cfg.break_load:return {'success':True,'target_N':float(loads[self.target]),'other_peak_N':other_peak,'elapsed_s':step*self.cfg.dt}
        return {'success':False,'reason':'target not reached in validation','target_N':float(loads[self.target]),'other_peak_N':other_peak}

    def load_update(self):
        if self.step_count-self.last_loading_step<round(self.cfg.control_dt/self.cfg.dt):return
        self.last_loading_step=self.step_count;start=time.perf_counter()
        if self.cfg.one_shot_loading:
            if self.loading_solution is None:
                A,w=self.headtail_response(self.cfg.headtail_model_horizon_s if self.target==5 else .05)
                result=solve_static_loading(A,w,self.target,break_load=self.cfg.break_load+(self.cfg.headtail_model_headroom if self.target==5 and self.cfg.headtail_model_headroom is not None else self.cfg.loading_model_headroom),
                    other_limit=self.cfg.loading_other_prediction_limit,torque_limit=self.cfg.continuous_torque)
                record={'time_s':float(self.data.time),'face':self.target+1,'wall_ms':1000*(time.perf_counter()-start),
                        'method':'one_shot_highs_milp','feasible':result['success'],'solver_status':result['status'],
                        'solver_message':result['message']}
                if not result['success']:
                    self.loading_records.append(record);self.stop('one-shot loading optimization infeasible or failed');return
                self.loading_solution=result['torque'].copy()
                record.update(torque_Nm=self.loading_solution.tolist(),predicted_N=result['predicted_loads'].tolist())
                validation=self.validate_loading_solution(self.loading_solution)
                record['ramp_validation']=validation
                if not validation['success']:
                    record['feasible']=False;self.loading_records.append(record)
                    self.stop('one-shot loading validation failed: '+validation['reason']);return
                self.loading_records.append(record)
            ramp=self.cfg.loading_ramp_s
            progress=min(1.,self.phase_time/max(ramp,1e-9))
            self.loading_torque=self.loading_solution*progress
            self.data.ctrl[self.actuators]=self.loading_qhold+self.loading_torque/self.kp if self.loading_hold else self.loading_torque
            return
        if self.target==5 and not self.headtail_fast_attempted:
            self.headtail_fast_attempted=True;candidate=self.headtail_impulse()
            if candidate is not None:
                _,torque,predicted=candidate;self.loading_torque=torque.copy()
                self.data.ctrl[self.actuators]=self.loading_qhold+self.loading_torque/self.kp
                self.loading_records.append({'time_s':float(self.data.time),'face':6,
                    'wall_ms':1000*(time.perf_counter()-start),'torque_Nm':torque.tolist(),
                    'method':'validated_one_substep','prediction_dt_s':self.cfg.dt,
                    'predicted_N':predicted.tolist(),'feasible':True})
                return
        A,w=self.headtail_response() if self.target==5 else response(self.model,self.data,self.ids())
        if self.target==5:
            delta,feasible,predicted=headtail_projected_increments(torch.from_numpy(A[None]).float(),torch.from_numpy(w[None]).float(),
                torch.from_numpy(self.loading_torque[None]).float(),torch.tensor([self.target]),.05,
                other_limit=self.cfg.headtail_prediction_limit,increment=self.cfg.headtail_increment,load_step=self.cfg.headtail_load_step,goal_limit=self.cfg.break_load+self.cfg.loading_goal_margin)
        else:
            delta,feasible,predicted=projected_increments(torch.from_numpy(A[None]).float(),torch.from_numpy(w[None]).float(),
                torch.from_numpy(self.loading_torque[None]).float(),torch.tensor([self.target]),.05,goal_limit=self.cfg.break_load+self.cfg.loading_goal_margin,other_limit=self.cfg.loading_other_prediction_limit)
        self.loading_torque+=delta[0].numpy()
        self.data.ctrl[self.actuators]=self.loading_qhold+self.loading_torque/self.kp if self.loading_hold else self.loading_torque
        self.loading_records.append({'time_s':float(self.data.time),'face':self.target+1,'wall_ms':1000*(time.perf_counter()-start),
            'torque_Nm':self.loading_torque.tolist(),'feasible':bool(feasible[0]),'predicted_N':predicted[0].tolist()})

    def shim(self):
        d=self.data
        modules=[NS(data=NS(joint_pos=torch.from_numpy(d.qpos[self.qadr[i]:self.qadr[i]+1].copy())[None],
            root_link_pos_w=torch.from_numpy(d.xpos[self.root_bodies[i]].copy())[None],
            root_link_quat_w=torch.from_numpy(d.xquat[self.root_bodies[i]].copy())[None])) for i in range(6)]
        return NS(modules=modules,eq_ids=torch.tensor(self.old),variant_ids=torch.tensor(self.new),
                  env=NS(sim=NS(data=NS(eq_active=torch.from_numpy(d.eq_active.copy())[None]))))

    def prepare_internal(self):
        plan=self.planner.plan(self.shim(),self.target)
        self.curves={name:self.curve_from(points) for name,points in plan.points.items()}
        k=self.planner.kin;anchor=k.roots[(self.target+1)%6].copy()
        saved_q=k.q.copy();saved_roots=k.roots.copy()
        try:
            for name,curve in self.curves.items():
                for q in curve.q[::5]:
                    k.q=q;k.roots,_,_=k.fk(q,self.target,anchor)
                    penetration,pair,clearance=self.planner.screen(self.target,name)
                    if penetration>1e-5 or (name=='ROTATE' and clearance<.001):
                        raise RuntimeError(f'{name}: pair={pair}, penetration={penetration}, clearance={clearance}')
        finally:k.q=saved_q;k.roots=saved_roots
        self.path_records.append({'connection':self.target+1,'segments':plan.diagnostics,
            'durations_s':{name:curve.duration for name,curve in self.curves.items()}})
        self.set_pd();self.curve=self.curves['SEPARATE'];self.transition('SEPARATE')

    def prepare_headtail_parallel(self):
        k=self.planner.kin;anchor=self.planner.sync(self.shim(),5)
        _,source,_=k.fk(k.q,5,anchor);initial=inverse(anchor)@source
        points=[k.q.copy()];rng=np.random.default_rng(7)
        for u in np.linspace(0.,1.,41)[1:]:
            target=initial[:3,3]+np.array([0.,0.,-self.cfg.headtail_parallel_gap*u]);previous=k.q.copy()
            def residual(q):
                _,source,_=k.fk(q,5,anchor);relative=inverse(anchor)@source
                return np.r_[10*(relative[:3,3]-target),Rotation.from_matrix(initial[:3,:3].T@relative[:3,:3]).as_rotvec(),.0001*(q-previous)]
            best=None
            seeds=[previous] + ([previous+rng.normal(0,.15,6) for _ in range(8)] if len(points)==1 else [])
            for seed in seeds:
                fit=least_squares(residual,seed,max_nfev=400,ftol=1e-10,xtol=1e-10,gtol=1e-10)
                r=residual(fit.x)
                if np.linalg.norm(r[:3])>2e-4 or np.linalg.norm(r[3:6])>1e-3:continue
                k.q=fit.x;k.roots,_,_=k.fk(k.q,5,anchor)
                penetration,pair,_=self.planner.screen(5,'SEPARATE')
                if penetration>1e-5:continue
                if best is None or np.linalg.norm(fit.x-previous)<best[0]:best=(np.linalg.norm(fit.x-previous),fit.x.copy())
            if best is None:raise RuntimeError(f'parallel head-tail path failed at {u*self.cfg.headtail_parallel_gap:.4f} m')
            k.q=best[1];k.roots,_,_=k.fk(k.q,5,anchor);points.append(k.q.copy())
        acceleration=math.radians(self.cfg.outer_acceleration_deg_s2) if getattr(self.cfg,'acceleration_limit_enabled',True) else None
        self.curve=TimedCurve(np.asarray(points),math.radians(self.cfg.outer_speed_deg_s),acceleration,self.cfg.dt)
        self.set_pd(outer=True);self.transition('HEADTAIL_SEPARATE')

    def closest_unfold_update(self):
        """Linearize all head-tail mesh distances along the connected-chain FK."""
        k=self.planner.kin;anchor=self.planner.sync(self.shim(),5);q=k.q.copy();m=k.model;d=k.data
        pairs=[(m.geom(f'module_0/{a}1_geom').id,m.geom(f'module_5/{b}6_geom').id) for a in ('a','b') for b in ('a','b')]
        def distances(x):
            roots,_,_=k.fk(x,5,anchor)
            k.apply_frame(Frame(0.,'UNFOLD',5,x,roots,k.active.copy()))
            return np.array([mujoco.mj_geomDistance(m,d,int(a),int(b),.5,None) for a,b in pairs])
        base=distances(q);eps=1e-4;eye=np.eye(6)
        jac=np.stack([(distances(q+eps*eye[j])-distances(q-eps*eye[j]))/(2*eps) for j in range(6)],axis=1)
        ramp=min(1.,self.phase_time/max(self.cfg.closest_unfold_ramp_s,1e-9))
        speed=1.2+ramp*(self.cfg.closest_unfold_max_speed-1.2)
        gain=1.5+ramp*(self.cfg.closest_unfold_gain-1.5)
        qvel=np.clip(-gain*q,-speed,speed);active=base<.02
        # A soft escape requirement is bounded by feasible joint speed; the
        # measured geometry remains protected by real collision dynamics.
        needed=np.clip((.003-base)*1.5,0.,.015)
        if active.any():
            fit=minimize(lambda v:float(np.dot(v-qvel,v-qvel)),np.zeros(6),method='SLSQP',bounds=[(-speed,speed)]*6,
                         constraints=[{'type':'ineq','fun':lambda v:jac[active]@v-needed[active]}],options={'maxiter':60,'ftol':1e-9})
            if not fit.success or not np.isfinite(fit.x).all() or np.min(jac[active]@fit.x-needed[active])<-1e-5:
                self.stop(f'closest-point clearance infeasible: min_distance={base.min():.5f}m');return
            qvel=fit.x
        self.qref=self.data.qpos[self.qadr].copy()+self.cfg.control_dt*qvel;self.vref[:]=0.

    def approach_feedback_update(self):
        """Small reference corrections from finite-difference MuJoCo rollouts."""
        gap,angle,_,_=self.geometry(new_weld=True)
        # Once within pose tolerance hold the target to let relative velocity decay.
        if gap<self.cfg.capture_position_tolerance*.8 and angle<math.radians(self.cfg.capture_angle_tolerance_deg)*.8:
            return
        if self.headtail_pool is None:self.headtail_pool=ThreadPoolExecutor(max_workers=8,thread_name_prefix='zbot-dock')
        m=self.model;base=self.qref.copy();eps=.004;controls=np.repeat(base[None],13,axis=0)
        controls[1:7]+=np.eye(6)*eps;controls[7:]-=np.eye(6)*eps
        eq=self.new[self.target]
        src=int(m.eq_obj1id[eq]);dst=int(m.eq_obj2id[eq]);goal=np.eye(3)
        def sample(index):
            if getattr(self.headtail_worker,'model',None) is not m:
                self.headtail_worker.model=m;self.headtail_worker.data=mujoco.MjData(m)
            d=self.headtail_worker.data;mujoco.mj_copyData(d,m,self.data);d.ctrl[self.actuators]=controls[index]
            mujoco.mj_step(m,d,nstep=max(1,round(.05/self.cfg.dt)))
            R0=d.site_xmat[src].reshape(3,3);R1=d.site_xmat[dst].reshape(3,3)
            return np.r_[R1.T@(d.site_xpos[src]-d.site_xpos[dst]),.05*Rotation.from_matrix(goal.T@R1.T@R0).as_rotvec()]
        errors=np.asarray(list(self.headtail_pool.map(sample,range(13))))
        J=((errors[1:7]-errors[7:])/(2*eps)).T
        correction=-J.T@np.linalg.solve(J@J.T+1e-5*np.eye(6),errors[0])
        if not np.isfinite(correction).all():raise ValueError('nonfinite docking correction')
        q=self.data.qpos[self.qadr];self.qref=np.clip(base+np.clip(.05*correction,-.0003,.0003),q-.08,q+.08);self.vref[:]=0.

    def apply(self):
        if self.loading_mode:
            try:self.load_update()
            except Exception as error:
                self.stop(f'loading planner error: {type(error).__name__}: {error}');return
            demand=(self.kp*(self.loading_qhold-self.data.qpos[self.qadr])-self.kd*self.data.qvel[self.dofs]+self.loading_torque
                    if self.loading_hold else self.loading_torque-self.loading_damping*self.data.qvel[self.dofs])
            if not np.isfinite(demand).all() or (getattr(self.cfg,'raw_torque_limit_enabled',True) and abs(demand).max()>self.cfg.peak_torque):self.stop('requested loading torque protection')
            return
        if self.phase=='UNFOLD' and self.cfg.closest_unfold and self.step_count%round(self.cfg.control_dt/self.cfg.dt)==0:
            try:self.closest_unfold_update()
            except Exception as error:self.stop(f'closest-point planning error: {type(error).__name__}: {error}');return
            if self.phase=='FAILED':return
        if self.phase=='UNFOLD' and not self.cfg.direct_unfold and not self.cfg.closest_unfold and self.step_count%round(self.cfg.control_dt/self.cfg.dt)==0:
            try:self.unfold_update()
            except Exception as error:
                self.stop(f'unfold planner error: {type(error).__name__}: {error}');return
            if self.phase=='FAILED':return
        if self.curve is not None and not (self.cfg.approach_feedback and self.phase=='APPROACH' and self.phase_time>=self.curve.duration) and (self.phase in ('CURL','SEPARATE','ROTATE','APPROACH','HEADTAIL_SEPARATE','UNFOLD_MIDDLE') or (self.phase=='UNFOLD' and self.cfg.direct_unfold and not self.cfg.closest_unfold)):
            idx=min(max(0,int(self.phase_time/self.cfg.dt)),len(self.curve.q)-1)
            self.qref=self.curve.q[idx].copy();self.vref=self.curve.v[idx].copy() if self.phase_time<self.curve.duration else np.zeros(6)
        else:self.vref[:]=0.
        if self.cfg.approach_feedback and self.phase=='APPROACH' and self.curve is not None and self.phase_time>=self.curve.duration and self.step_count%round(self.cfg.control_dt/self.cfg.dt)==0:
            try:self.approach_feedback_update()
            except Exception as error:self.stop(f'docking feedback error: {type(error).__name__}: {error}');return
        if self.phase=='APPROACH' and self.curve is not None and self.phase_time>=self.curve.duration and self.cfg.docking_stiffness>0.:
            self.kp=self.cfg.docking_stiffness
            self.model.actuator_gainprm[self.actuators,0]=self.kp
            self.model.actuator_biasprm[self.actuators,1]=-self.kp
        target=self.qref+self.kd/self.kp*self.vref
        demand=self.kp*(target-self.data.qpos[self.qadr])-self.kd*self.data.qvel[self.dofs]
        if not np.isfinite(demand).all() or (getattr(self.cfg,'raw_torque_limit_enabled',True) and abs(demand).max()>self.cfg.peak_torque):self.stop(f'requested torque protection: {demand.tolist()}');return
        self.data.ctrl[self.actuators]=target

    def advance(self):
        if self.phase in ('DONE','FAILED'):return
        t=self.phase_time;cfg=self.cfg
        if self.phase=='INITIAL_STRAIGHT_HOLD':
            if t>=cfg.initial_hold_seconds:self.start_curve('CURL',self.closed_q)
        elif self.phase=='CURL':
            if t>=self.curve.duration:self.transition('CAPTURE_HEADTAIL')
        elif self.phase=='CAPTURE_HEADTAIL':
            gap,angle,v,w=self.geometry(5,new_weld=True)
            if gap<cfg.capture_position_tolerance and angle<math.radians(cfg.capture_angle_tolerance_deg) and v<cfg.capture_linear_speed and w<cfg.capture_angular_speed:self.capture+=cfg.control_dt
            else:self.capture=0.
            if self.capture>=.03:
                self.model.eq_solref[self.old[5],0]=.03;self.data.eq_active[self.old[5]]=True;self.transition('SETTLE_HEADTAIL')
            elif t>cfg.motion_wait_seconds:self.stop(f'headtail capture timeout: gap={gap}, angle={math.degrees(angle)}')
        elif self.phase in ('SETTLE_HEADTAIL','SETTLE'):
            eq=self.old[5] if self.phase=='SETTLE_HEADTAIL' else self.new[self.target]
            self.model.eq_solref[eq,0]=.03+(.005-.03)*min(t/.2,1.)
            if not self.connected().all():self.stop('connection lost during closed hold');return
            if t>=cfg.settle_seconds-.5*cfg.dt:
                if self.phase=='SETTLE_HEADTAIL':self.stage=0;self.start_load('LOAD',ORDER[0])
                else:
                    self.history.append({'connection':self.target+1,'time_s':float(self.data.time)})
                    self.stage+=1
                    if self.stage<len(ORDER):self.start_load('LOAD',ORDER[self.stage])
                    else:self.start_load('LOAD_HEADTAIL',5)
        elif self.phase in ('LOAD','LOAD_HEADTAIL'):
            if not self.connected()[self.target]:
                if self.phase=='LOAD':
                    if cfg.post_break_hold_s>0.:
                        self.qref=self.data.qpos[self.qadr].copy();self.set_pd();self.transition('POST_BREAK_HOLD')
                    else:
                        try:self.prepare_internal()
                        except (RuntimeError,ValueError) as error:self.stop(f'path planning failed: {error}')
                else:
                    if cfg.headtail_parallel_gap>0:
                        try:self.prepare_headtail_parallel()
                        except (RuntimeError,ValueError) as error:self.stop(f'parallel headtail planning failed: {error}')
                    elif (cfg.headtail_break_hold_s if cfg.headtail_break_hold_s is not None else cfg.post_break_hold_s)>0.:
                        self.qref=self.data.qpos[self.qadr].copy();self.set_pd(outer=True);self.transition('HEADTAIL_BREAK_HOLD')
                    else:self.start_unfold()
            elif t>40.:self.stop('Jacobian loading timeout')
        elif self.phase=='HEADTAIL_BREAK_HOLD':
            hold=cfg.headtail_break_hold_s if cfg.headtail_break_hold_s is not None else cfg.post_break_hold_s
            if t>=hold:self.start_unfold()
        elif self.phase=='POST_BREAK_HOLD':
            if t>=cfg.post_break_hold_s:
                try:self.prepare_internal()
                except (RuntimeError,ValueError) as error:self.stop(f'path planning failed: {error}')
        elif self.phase in ('SEPARATE','ROTATE','APPROACH'):
            motion_phase=self.phase;motion_curve=self.curve
            if t<self.curve.duration:return
            gap,angle,v,w=self.geometry();ga,gb=self.pair_geoms[self.target]
            clearance=float(mujoco.mj_geomDistance(self.model,self.data,int(ga),int(gb),.5,None))
            if clearance<.001:clearance=max(clearance,clearance_lower_bound(self.model,self.data,int(ga),int(gb),self.clearance_cache))
            if self.phase=='SEPARATE' and gap>=cfg.separation_gap-.001 and clearance>=.001:
                if cfg.overlap_rotation:
                    self.curve=self.curves['APPROACH'];self.transition('APPROACH')
                else:
                    self.curve=self.curves['ROTATE'];self.transition('ROTATE')
            elif self.phase=='ROTATE' and clearance>=.001:
                self.curve=self.curves['APPROACH'];self.transition('APPROACH')
            elif self.phase=='APPROACH':
                gap,angle,v,w=self.geometry(new_weld=True)
                if gap<cfg.capture_position_tolerance and angle<math.radians(cfg.capture_angle_tolerance_deg) and v<cfg.capture_linear_speed and w<cfg.capture_angular_speed:self.capture+=cfg.control_dt
                else:self.capture=0.
                if self.capture>=.03:
                    self.model.eq_solref[self.new[self.target],0]=.03;self.data.eq_active[self.new[self.target]]=True;self.transition('SETTLE')
            if self.phase==motion_phase and t>motion_curve.duration+cfg.motion_wait_seconds:self.stop(f'motion tracking/capture timeout: gap={gap:.6f}m, angle={math.degrees(angle):.3f}deg, speed={v:.4f}m/s, angular_speed={w:.4f}rad/s')
        elif self.phase=='HEADTAIL_SEPARATE':
            gap,angle,_,_=self.geometry(5)
            if t>=self.curve.duration and gap>=self.cfg.headtail_parallel_gap-.002 and angle<math.radians(2):
                self.start_unfold()
            elif t>self.curve.duration+cfg.motion_wait_seconds:self.stop(f'headtail separation timeout: gap={gap}, angle={math.degrees(angle)}')
        elif self.phase=='UNFOLD_MIDDLE':
            gap,_,_,_=self.geometry(5)
            if gap>cfg.staged_unfold_gap:
                current=self.data.qpos[self.qadr].copy()
                acceleration=math.radians(cfg.unfold_acceleration_deg_s2) if getattr(cfg,'acceleration_limit_enabled',True) else None
                self.curve=TimedCurve(np.stack((current,np.zeros(6))),math.radians(cfg.unfold_speed_deg_s),acceleration,cfg.dt)
                self.transition('UNFOLD')
            elif t>self.curve.duration+cfg.motion_wait_seconds:self.stop(f'middle-joint unfolding timeout: gap={gap}')
        elif self.phase=='UNFOLD':
            if abs(self.data.qpos[self.qadr]).max()<.02 and abs(self.data.qvel[self.dofs]).max()<.05:
                self.transition('FINAL_STRAIGHT_HOLD')
            elif t>cfg.unfold_timeout_s:self.stop('unfold tracking timeout')
        elif self.phase=='FINAL_STRAIGHT_HOLD':
            if not self.connected()[:5].all() or self.connected()[5]:self.stop('final chain topology incorrect')
            elif t>=cfg.settle_seconds-.5*cfg.dt:
                gap,_,_,_=self.geometry(5)
                if not np.array_equal(self.data.eq_active[self.new].astype(bool),[True,True,False,True,True]):
                    self.stop('final registration angles incorrect')
                elif (abs(self.data.qpos[self.qadr]).max()>=.02 or abs(self.data.qvel[self.dofs]).max()>=.05
                    or self.snapshot()['root_collinearity_error_m']>=.005 or gap<.05):
                    self.stop('final straight pose or separation tolerance not met')
                else:self.transition('DONE')

    def step(self):
        if self.phase in ('DONE','FAILED'):return
        if self.step_count%round(self.cfg.control_dt/self.cfg.dt)==0:self.advance()
        if self.phase in ('DONE','FAILED'):return
        self.apply()
        if self.phase=='FAILED':return
        mujoco.mj_step(self.model,self.data);self.step_count+=1;self.measure()
        if self.data.time>=self.next_log:
            self.trace.append(self.snapshot());self.next_log=self.data.time+.5
        if self.phase in ('UNFOLD','LOAD_HEADTAIL') and self.data.time>=self.next_progress:
            print(f'CPU_FSM t={self.data.time:.2f} {self.phase} target_load={self.loads[5]:.2f}N '
                  f'other_peak={self.peak_other:.2f}N max_joint={abs(self.data.qpos[self.qadr]).max():.3f}rad',flush=True)
            self.next_progress=self.data.time+5.

    def snapshot(self):
        points=self.data.xpos[self.root_bodies];_,_,v=np.linalg.svd(points-points.mean(0),full_matrices=False)
        residual=(points-points.mean(0))@v[1:].T
        return {'time_s':float(self.data.time),'phase':self.phase,'connection':self.target+1,'q_rad':self.data.qpos[self.qadr].tolist(),
            'dq_rad_s':self.data.qvel[self.dofs].tolist(),'connected':self.connected().tolist(),
            'registrations':self.data.eq_active[self.new].astype(bool).tolist(),'loads_N':self.loads.tolist(),
            'peak_loads_N':self.peak_load.tolist(),
            'interface_gap_m':[float(np.linalg.norm(self.data.site_xpos[self.out[i]]-self.data.site_xpos[self.inlet[(i+1)%6]])) for i in range(6)],
            'peak_other_load_N':self.peak_other,'peak_actual_torque_Nm':self.peak_torque,
            'max_module_penetration_m':self.penetration,'max_ground_penetration_m':self.ground_penetration,
            'worst_module_pair':self.worst_module_pair,'worst_ground_pair':self.worst_ground_pair,
            'root_collinearity_error_m':float(np.linalg.norm(residual,axis=1).max()),'failure':self.failure}

    def report(self):
        return {'experiment':EXPERIMENT_ID,'backend':'native_mujoco_cpu','torch_cuda_initialized':torch.cuda.is_initialized(),
            'success':self.phase=='DONE',
            'configuration':vars(self.cfg),'initial_straight_qpos':self.initial_qpos.tolist(),'terminal':self.snapshot(),
            'events':self.events,'completed_connections':self.history,'loading':self.loading_records,'planning':self.path_records,'trace':self.trace,
            'scope':'Actual CPU dynamics; physical qpos/root poses written only at initialization; load-triggered releases are passive at 80N.'}

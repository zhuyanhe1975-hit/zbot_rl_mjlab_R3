"""Geometry-only sequential reconfiguration. No integration, forces or policy.

Five unbroken interfaces define an open serial chain with a fixed inlet root.
We solve joint IK for the cut interface, and reconstruct module root poses by
FK. After recapture, the next cut changes the anchor without moving the robot.
"""
import math
from dataclasses import dataclass
import numpy as np
import mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from mjlab.scene import Scene
from .sequential_rz270 import sequential_rz270_env_cfg,add_sequential_connectors
from .disconnect.model import ROOT_NAMES,TIP_NAMES
from ..snake.rz270 import rz270_spec

EXPERIMENT_ID='Zbot-Closed180-To-Snake-RZ270-Kinematic-v1'

def add_kinematic_connectors(spec):
    add_sequential_connectors(spec,include_headtail=True)


def transform(position,quaternion):
    T=np.eye(4);R=np.empty(9);mujoco.mju_quat2Mat(R,np.asarray(quaternion,dtype=float))
    T[:3,:3]=R.reshape(3,3);T[:3,3]=position
    return T


def inverse(T):
    out=np.eye(4);out[:3,:3]=T[:3,:3].T;out[:3,3]=-T[:3,:3].T@T[:3,3];return out


def rz(angle):
    T=np.eye(4);c=math.cos(angle);s=math.sin(angle)
    T[0,0]=c;T[0,1]=-s;T[1,0]=s;T[1,1]=c
    return T


def smooth(x):return x*x*x*(10+x*(-15+6*x))


@dataclass
class Frame:
    time: float
    phase: str
    connector: int
    q: np.ndarray
    roots: np.ndarray
    eq_active: np.ndarray
    position_error_m: float=0.
    angle_error_rad: float=0.


class KinematicPlanningError(RuntimeError):pass


class KinematicRz270:
    def __init__(self,steps_per_phase=50,phase_duration=1.,order=(0,1,2,3,4)):
        if sorted(order)!=list(range(5)):raise ValueError('order must contain each of 0..4 once')
        if steps_per_phase<2 or phase_duration<=0:raise ValueError('positive duration and >=2 steps required')
        self.steps=steps_per_phase;self.duration=phase_duration;self.order=tuple(order)
        cfg=sequential_rz270_env_cfg(order=order)
        cfg.scene.spec_fn=add_kinematic_connectors
        self.scene=Scene(cfg.scene,device='cpu');self.model=self.scene.compile()
        self.reference=rz270_spec().compile()
        self.data=mujoco.MjData(self.model)
        self.joints=[self.model.joint(f'module_{i}/joint{i+1}').id for i in range(6)]
        self.qadr=np.array([self.model.jnt_qposadr[j] for j in self.joints])
        self.rootadr=np.array([self.model.joint(f'module_{i}/module_root').qposadr[0] for i in range(6)])
        self.tip_ids=[self.model.body(f'module_{i}/{TIP_NAMES[i]}').id for i in range(6)]
        self.old_eq=np.array([self.model.equality(f'connector_{i}').id for i in range(6)])
        self.new_eq=np.array([self.model.equality(f'connector_{i}_rz270').id for i in range(6)])
        self.inlet=[];self.outlet=[];self.tip=[]
        for i,b in enumerate(self.tip_ids):
            si=self.model.site(f'module_{i}/port_in').id;so=self.model.site(f'module_{i}/port_out').id
            self.inlet.append(transform(self.model.site_pos[si],self.model.site_quat[si]))
            self.outlet.append(transform(self.model.site_pos[so],self.model.site_quat[so]))
            self.tip.append(transform(self.model.body_pos[b],self.model.body_quat[b]))
        self.qref=self.model.qpos0[self.qadr].copy()
        self.reference_joint_ranges=np.array([self.reference.joint(f'joint{i+1}').range for i in range(6)])
        # Explicit user request: limits are ignored only in this geometry demo.
        self.bounds=np.tile([-np.inf,np.inf],(6,1))
        self.model.jnt_limited[self.joints]=False
        self.registration=np.zeros(6)
        self.active=self.model.eq_active0.copy()
        self.q=self.qref.copy()
        self.roots=np.array([transform(self.model.qpos0[a:a+3],self.model.qpos0[a+3:a+7]) for a in self.rootadr])
        self.frames=[];self.events=[];self.time=0.;self.stage_errors=[]

    def fk(self,q,cut,anchor):
        roots=[None]*6;first=(cut+1)%6;roots[first]=anchor
        axes=np.empty((6,3));anchors=np.empty((6,3))
        for step in range(6):
            i=(first+step)%6
            tip=roots[i]@self.tip[i]@rz(q[i]-self.qref[i])
            axes[i]=tip[:3,2];anchors[i]=tip[:3,3]
            if step<5:
                roots[(i+1)%6]=tip@self.outlet[i]@inverse(self.inlet[(i+1)%6])@rz(self.registration[i])
            else:source=tip@self.outlet[i]@inverse(self.inlet[first])
        return np.array(roots),source,anchor

    def errors(self,q,cut,anchor,goal_rotation,goal_position=None,gap=None):
        roots,source,destination=self.fk(q,cut,anchor)
        relative=inverse(destination)@source
        angle=Rotation.from_matrix(goal_rotation.T@relative[:3,:3]).as_rotvec()
        if gap is None:position=(relative[:3,3]-goal_position)/.1
        else:position=np.array([(np.linalg.norm(relative[:3,3])-gap)/.1])
        return np.r_[position,angle]

    def solve(self,cut,anchor,R,position=None,gap=None):
        previous=self.q.copy()
        def objective(q):return np.r_[self.errors(q,cut,anchor,R,position,gap),1e-5*(q-previous)]
        result=least_squares(objective,previous,bounds=self.bounds.T,max_nfev=300,
                             ftol=1e-9,xtol=1e-9,gtol=1e-9)
        error=self.errors(result.x,cut,anchor,R,position,gap)
        p=float(np.linalg.norm(error[:-3])*.1);a=float(np.linalg.norm(error[-3:]))
        if p>2e-5 or a>2e-4:
            raise KinematicPlanningError(f'connector={cut}, position={p:.6g}m, angle={math.degrees(a):.6g}deg, q={result.x.tolist()}')
        self.q=result.x;self.roots,_,_=self.fk(self.q,cut,anchor)
        return p,a

    def connectivity(self,active=None):
        active=self.active if active is None else active
        parent=list(range(6));edges=0
        def root(i):
            while parent[i]!=i:i=parent[i]
            return i
        for i in range(6):
            original=bool(active[self.old_eq[i]]);variant=bool(active[self.new_eq[i]])
            if original and variant:raise KinematicPlanningError(f'connector {i}: two registration welds active')
            if original or variant:
                edges+=1;a=root(i);b=root((i+1)%6);parent[a]=b
        return {'components':len({root(i) for i in range(6)}),'connected_faces':edges}

    def record(self,phase,connector,p=0.,a=0.,event=False):
        graph=self.connectivity()
        if graph['components']!=1 or graph['connected_faces']<5:
            raise KinematicPlanningError(f'Disconnected robot is forbidden: {graph}')
        frame=Frame(self.time,phase,connector,self.q.copy(),self.roots.copy(),self.active.copy(),p,a)
        self.frames.append(frame)
        if event:
            info={'time_s':self.time,'phase':phase,'connector':connector,'joint_angle_rad':self.q.tolist(),
                  'registered_270':self.active[self.new_eq[:5]].astype(bool).tolist(),'brace_reclocked':bool(self.active[self.new_eq[5]]),
                  'position_error_m':p,'angle_error_deg':math.degrees(a),'connectivity':graph}
            self.events.append(info);print('KINEMATIC',info,flush=True)

    def plan(self):
        self.record('INITIAL_CLOSED',-1,event=True)
        for cut in (5,*self.order):
            anchor=self.roots[(cut+1)%6].copy()
            self.active[self.old_eq[cut]]=False
            self.record('DISCONNECT',cut,event=True)
            # Keep five edges exactly connected. Only the cut interface moves.
            for phase in ('SEPARATE','ROTATE','APPROACH'):
                start_position=None
                if phase=='APPROACH':
                    _,source,destination=self.fk(self.q,cut,anchor)
                    start_position=(inverse(destination)@source)[:3,3]
                maximum=np.zeros(2)
                for step in range(1,self.steps+1):
                    u=smooth(step/self.steps)
                    angle=0. if phase=='SEPARATE' else (math.pi/2*u if phase=='ROTATE' else math.pi/2)
                    R=rz(-angle)[:3,:3]
                    if phase=='APPROACH':
                        if step==self.steps:p,a=self.solve(cut,anchor,R,position=np.zeros(3))
                        else:p,a=self.solve(cut,anchor,R,gap=np.linalg.norm(start_position)*(1-u))
                    else:p,a=self.solve(cut,anchor,R,gap=.05*u if phase=='SEPARATE' else .05)
                    self.time+=self.duration/self.steps
                    maximum=np.maximum(maximum,[p,a]);self.record(phase,cut,p,a)
                self.stage_errors.append({'connector':cut,'phase':phase,'duration_s':self.duration,
                                          'max_position_error_m':maximum[0],'max_angle_error_rad':maximum[1]})
                self.record(phase+'_DONE',cut,p,a,event=True)
            self.registration[cut]=math.pi/2;self.active[self.new_eq[cut]]=True
            self.record('REBUILD_WELD',cut,p,a,event=True)
        # Temporary loop brace is released only after all five installations change.
        if not self.active[self.new_eq[:5]].all():
            raise KinematicPlanningError('Cannot release brace before all five internal connections are rebuilt')
        self.active[self.old_eq[5]]=False;self.active[self.new_eq[5]]=False
        self.record('OPEN_HEAD_TAIL',5,event=True)
        anchor=self.roots[0].copy();initial=self.q.copy()
        for step in range(1,self.steps+1):
            self.q=initial*(1-smooth(step/self.steps))
            self.roots,_,_=self.fk(self.q,5,anchor)
            self.time+=self.duration/self.steps;self.record('UNFOLD',5)
        self.record('DONE',5,event=True)
        return self.frames

    def active_constraint_errors(self):
        maximum_position=0.;maximum_angle=0.
        for frame in self.frames:
            self.apply_frame(frame)
            for eq in np.flatnonzero(frame.eq_active):
                a=int(self.model.eq_obj1id[eq]);b=int(self.model.eq_obj2id[eq])
                p=np.linalg.norm(self.data.site_xpos[a]-self.data.site_xpos[b])
                R=self.data.site_xmat[a].reshape(3,3).T@self.data.site_xmat[b].reshape(3,3)
                maximum_position=max(maximum_position,float(p))
                maximum_angle=max(maximum_angle,float(np.linalg.norm(Rotation.from_matrix(R).as_rotvec())))
        return {'max_active_weld_position_error_m':maximum_position,'max_active_weld_angle_error_rad':maximum_angle}

    def apply_frame(self,frame):
        d=self.data
        d.qpos[:]=self.model.qpos0
        d.qpos[self.qadr]=frame.q;d.qvel[:]=0.;d.eq_active[:]=frame.eq_active
        for adr,T in zip(self.rootadr,frame.roots):
            d.qpos[adr:adr+3]=T[:3,3]
            quat=np.empty(4);mujoco.mju_mat2Quat(quat,T[:3,:3].flatten());d.qpos[adr+3:adr+7]=quat
        # FK only: no mj_step, force control, dynamics or collision resolution.
        mujoco.mj_kinematics(self.model,d);mujoco.mj_camlight(self.model,d)

    def validate_reference(self):
        d=mujoco.MjData(self.reference)
        free=int(np.where(self.reference.jnt_type==mujoco.mjtJoint.mjJNT_FREE)[0][0]);adr=self.reference.jnt_qposadr[free]
        d.qpos[adr:adr+3]=self.roots[0,:3,3]
        q=np.empty(4);mujoco.mju_mat2Quat(q,self.roots[0,:3,:3].flatten());d.qpos[adr+3:adr+7]=q
        for i in range(6):d.qpos[self.reference.joint(f'joint{i+1}').qposadr]=self.q[i]
        mujoco.mj_kinematics(self.reference,d)
        positions=[];angles=[]
        for i,name in enumerate(ROOT_NAMES):
            b=self.reference.body(name).id
            positions.append(float(np.linalg.norm(self.roots[i,:3,3]-d.xpos[b])))
            angles.append(float(np.linalg.norm(Rotation.from_matrix(d.xmat[b].reshape(3,3).T@self.roots[i,:3,:3]).as_rotvec())))
        return {'max_root_position_error_m':max(positions),'max_root_angle_error_rad':max(angles),
                'joint_zero_error_rad':float(np.max(np.abs(self.q))),
                'registered_270':self.active[self.new_eq[:5]].astype(bool).tolist(),
                'head_tail_open':bool(not self.active[self.old_eq[5]] and not self.active[self.new_eq[5]])}

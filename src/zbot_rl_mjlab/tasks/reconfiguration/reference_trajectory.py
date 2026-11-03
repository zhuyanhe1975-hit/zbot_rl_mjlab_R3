"""Axial cut-interface planning and smooth timing for the deterministic FSM.

Inspired by the user's seven_module_sim_share_20261027 reference; applies to
our six-module geometry, registration convention and load protection.
"""
import math
from dataclasses import dataclass
import mujoco
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.spatial import ConvexHull

from .kinematic_mixed_closed import MixedClosedKinematics
from .kinematic_rz270 import Frame, inverse, rz, KinematicPlanningError


def clearance_lower_bound(model,data,a,b,cache):
    """Conservative convex-hull separation bound from supporting face axes.

    A positive projection gap proves at least that Euclidean clearance. It
    does not claim a signed penetration depth or complete SAT classification.
    """
    shapes=[]
    for geom in (a,b):
        mesh=int(model.geom_dataid[geom])
        if mesh not in cache:
            start=int(model.mesh_vertadr[mesh]);count=int(model.mesh_vertnum[mesh])
            vertices=model.mesh_vert[start:start+count].astype(float)
            hull=ConvexHull(vertices)
            cache[mesh]=(vertices[hull.vertices],np.unique(np.round(hull.equations[:,:3],8),axis=0))
        vertices,normals=cache[mesh];R=data.geom_xmat[geom].reshape(3,3)
        shapes.append((vertices@R.T+data.geom_xpos[geom],normals@R.T))
    A,na=shapes[0];B,nb=shapes[1];axes=np.concatenate((na,nb))
    axes/=np.linalg.norm(axes,axis=1,keepdims=True)
    pa=A@axes.T;pb=B@axes.T
    gap=np.maximum(pb.min(0)-pa.max(0),pa.min(0)-pb.max(0))
    return max(0.,float(gap.max()))


class TimedCurve:
    def __init__(self,points,speed=math.pi/4,acceleration=math.pi,dt=.005):
        points=np.asarray(points,dtype=float)
        points=points[np.r_[True,np.linalg.norm(np.diff(points,axis=0),axis=1)>1e-10]]
        self.endpoint=points[-1].copy();self.dt=dt
        if len(points)==1:
            self.duration=dt;self.q=points.copy();self.v=np.zeros_like(points);self.a=np.zeros_like(points);return
        length=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
        spline=CubicSpline(length/length[-1],points,axis=0,bc_type='natural')
        def evaluate(u):
            x=u**3*(10+u*(-15+6*u));dx=30*u*u*(1-u)**2;ddx=60*u*(1-u)*(1-2*u)
            q=spline(x);q1=spline(x,1);q2=spline(x,2)
            return q,q1*dx[:,None],q2*dx[:,None]**2+q1*ddx[:,None]
        _,d1,d2=evaluate(np.linspace(0.,1.,8193))
        duration=max(.08,float(abs(d1).max())/speed,
                     0. if acceleration is None else math.sqrt(float(abs(d2).max())/acceleration))*1.02
        for _ in range(8):
            steps=math.ceil(duration/dt);self.duration=steps*dt
            q,v,a=evaluate(np.arange(1,steps+1)/steps);v/=self.duration;a/=self.duration**2
            ratio=max(float(abs(v).max())/speed,
                      0. if acceleration is None else math.sqrt(float(abs(a).max())/acceleration))
            if ratio<=1.+1e-10:break
            duration=self.duration*ratio*1.01
        else:raise RuntimeError('Trajectory time scaling failed')
        self.q=q;self.v=v;self.a=a
        self.peak_speed=float(abs(v).max());self.peak_acceleration=float(abs(a).max())


@dataclass
class PlannedMotion:
    points:dict
    diagnostics:list


class AxialPlanner:
    def __init__(self,gap=.05,overlap_rotation=False):
        self.overlap_rotation=overlap_rotation
        self.kin=MixedClosedKinematics(steps_per_phase=20)
        self.gap=gap;self.hull_cache={}
        m=self.kin.model
        self.mates={tuple(sorted((m.geom(f'module_{i}/b{i+1}_geom').id,
                         m.geom(f'module_{(i+1)%6}/a{(i+1)%6+1}_geom').id))) for i in range(6)}
        self.module=[]
        for b in m.geom_bodyid:
            name=m.body(int(b)).name
            self.module.append(int(name.split('/')[0][7:]) if name.startswith('module_') else -1)

    def sync(self,state,cut):
        k=self.kin
        k.q=np.array([float(m.data.joint_pos[0,0]) for m in state.modules])
        k.roots=np.repeat(np.eye(4)[None],6,axis=0)
        for i,m in enumerate(state.modules):
            k.roots[i,:3,3]=m.data.root_link_pos_w[0].cpu().numpy()
            R=np.empty(9);mujoco.mju_quat2Mat(R,m.data.root_link_quat_w[0].cpu().numpy().astype(float))
            k.roots[i,:3,:3]=R.reshape(3,3)
        on=state.env.sim.data.eq_active[0].cpu().numpy()
        old=state.eq_ids.cpu().numpy();new=state.variant_ids.cpu().numpy()
        k.active[:]=False;k.active[k.old_eq]=on[old]
        k.active[k.new_eq[:5]]=on[new]
        k.registration[:]=0.;k.registration[:5]=k.target_delta*np.asarray(on[new],dtype=float)
        k.active[k.old_eq[cut]]=False
        return k.roots[(cut+1)%6].copy()

    def screen(self,cut,phase):
        k=self.kin;m=k.model;d=k.data
        k.apply_frame(Frame(0.,phase,cut,k.q.copy(),k.roots.copy(),k.active.copy()))
        mujoco.mj_collision(m,d)
        penetration=0.;pair=None
        for c in d.contact:
            a,b=int(c.geom1),int(c.geom2)
            if self.module[a]<0 or self.module[b]<0 or self.module[a]==self.module[b]:continue
            if tuple(sorted((a,b))) in self.mates:continue
            if -float(c.dist)>penetration:penetration=-float(c.dist);pair=[m.geom(a).name,m.geom(b).name]
        ga=m.geom(f'module_{cut}/b{cut+1}_geom').id
        gb=m.geom(f'module_{(cut+1)%6}/a{(cut+1)%6+1}_geom').id
        clearance=float(mujoco.mj_geomDistance(m,d,ga,gb,.5,None))
        if clearance<.001:
            clearance=max(clearance,clearance_lower_bound(m,d,ga,gb,self.hull_cache))
        return penetration,pair,clearance

    def plan(self,state,cut):
        k=self.kin;anchor=self.sync(state,cut)
        _,source,_=k.fk(k.q,cut,anchor)
        start=inverse(anchor)@source;R0=start[:3,:3];p0=start[:3,3]
        Rgoal=rz(-k.target_delta[cut])[:3,:3]
        turn=Rotation.from_matrix(R0.T@Rgoal).as_rotvec()
        separated=p0+np.array([0.,0.,-self.gap])
        points={};diagnostics=[];rotation_end=None
        for phase in (('SEPARATE','APPROACH') if self.overlap_rotation else ('SEPARATE','ROTATE','APPROACH')):
            values=[k.q.copy()];max_jump=0.;max_penetration=0.;min_clearance=.5;relaxed=0
            def target(u):
                if phase=='SEPARATE':
                    return (R0@Rotation.from_rotvec(turn*u).as_matrix() if self.overlap_rotation else R0),p0+np.array([0.,0.,-self.gap*u])
                if phase=='ROTATE':return R0@Rotation.from_rotvec(turn*u).as_matrix(),separated
                return Rgoal,rotation_end*(1-u)
            def advance(u0,u1,depth=0):
                nonlocal max_jump,max_penetration,min_clearance,relaxed
                previous=k.q.copy();R,p=target(u1);accepted=False;reason='IK'
                null_basis=None
                for mode in ('axial','clearance',*(f'detour_{i}' for i in range(8))):
                    if mode!='axial' and ((phase=='SEPARATE' and not self.overlap_rotation) or u1>=1. and phase=='APPROACH'):continue
                    try:
                        if mode=='axial':err,ang=k.solve(cut,anchor,R,position=p)
                        else:
                            k.q=previous.copy()
                            radius=float(np.linalg.norm(p));seed=previous.copy();detour=mode.startswith('detour_')
                            if detour:
                                if null_basis is None:
                                    h=1e-6;basis=np.eye(6)
                                    J=np.stack([(k.errors(previous+h*basis[j],cut,anchor,R,gap=radius)-k.errors(previous-h*basis[j],cut,anchor,R,gap=radius))/(2*h) for j in range(6)],-1)
                                    null_basis=np.linalg.svd(J,full_matrices=True)[2][-2:]
                                direction=int(mode.split('_')[1])*2*math.pi/8
                                seed=previous+.04*(math.cos(direction)*null_basis[0]+math.sin(direction)*null_basis[1])
                            def objective(q):
                                roots,source,_=k.fk(q,cut,anchor);relative=inverse(anchor)@source
                                return np.r_[k.errors(q,cut,anchor,R,gap=radius),
                                             (1e-5 if detour else .001)*(relative[:3,3]-p)/.1,
                                             (1e-3 if detour else 1e-5)*(q-seed)]
                            fit=least_squares(objective,seed,max_nfev=500,ftol=1e-10,xtol=1e-10,gtol=1e-10)
                            error=k.errors(fit.x,cut,anchor,R,gap=radius)
                            if np.linalg.norm(error[:-3])*.1>2e-5 or np.linalg.norm(error[-3:])>2e-4:continue
                            k.q=fit.x;k.roots,_,_=k.fk(k.q,cut,anchor)
                        jump=float(abs(k.q-previous).max())
                        penetration,pair,clearance=self.screen(cut,phase)
                        if jump>math.radians(8):reason='joint_step'
                        elif penetration>1e-5:reason=f'nonmating_collision {pair} depth={penetration}'
                        elif phase=='ROTATE' and clearance<.001:reason=f'rotation_clearance={clearance}'
                        else:
                            max_jump=max(max_jump,jump);max_penetration=max(max_penetration,penetration)
                            if phase=='ROTATE':min_clearance=min(min_clearance,clearance)
                            relaxed+=int(mode!='axial');accepted=True;break
                    except KinematicPlanningError:pass
                    k.q=previous.copy();k.roots,_,_=k.fk(k.q,cut,anchor)
                if accepted:values.append(k.q.copy());return
                k.q=previous.copy();k.roots,_,_=k.fk(k.q,cut,anchor)
                if depth>=5:raise KinematicPlanningError(f'connection {cut+1} {phase} at {u1:.4f}: {reason}')
                middle=(u0+u1)*.5;advance(u0,middle,depth+1);advance(middle,u1,depth+1)
            last=0.
            for u in np.linspace(0.,1.,21)[1:]:advance(last,float(u));last=float(u)
            if phase=='ROTATE' or (phase=='SEPARATE' and self.overlap_rotation):
                _,source,_=k.fk(k.q,cut,anchor);rotation_end=(inverse(anchor)@source)[:3,3]
            points[phase]=np.stack(values)
            diagnostics.append({'phase':phase,'waypoints':len(values),'max_joint_step_deg':math.degrees(max_jump),
                                'max_nonmating_penetration_m':max_penetration,'min_rotating_pair_clearance_m':min_clearance if phase=='ROTATE' else None,
                                'side_adjustment_samples':relaxed})
        return PlannedMotion(points,diagnostics)

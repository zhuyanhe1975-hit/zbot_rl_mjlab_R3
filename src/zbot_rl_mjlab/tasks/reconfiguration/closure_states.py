"""Discrete closed-ring registration audit, independent of a motion path."""
import math
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .kinematic_rz270 import KinematicRz270,inverse,rz,Frame


def skew(v):
    x,y,z=v
    return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])


class ClosedRegistrationAudit:
    def __init__(self,target_angle_deg=270,face_angles_deg=None):
        if target_angle_deg not in (90,270):raise ValueError('target angle must be 90 or 270')
        self.target_angle_deg=target_angle_deg
        angles=np.full(6,target_angle_deg) if face_angles_deg is None else np.asarray(face_angles_deg)
        if angles.shape!=(6,) or not np.isin(angles,[90,270]).all():raise ValueError('six face targets of 90 or 270 required')
        self.face_angles_deg=angles.tolist()
        self.delta=np.deg2rad(angles-180)
        self.robot=KinematicRz270(steps_per_phase=2)
        # Independent CPU MuJoCo site model, including mixed per-face clocks.
        m=self.robot.model
        for i,delta in enumerate(self.delta):
            if angles[i]==270:continue
            turn=rz(delta)[:3,:3]
            quaternion=np.array([math.cos(delta/2),0.,0.,math.sin(delta/2)])
            old=m.site(f'module_{i}/port_out').id
            variant=m.site(f'module_{i}/port_out_rz270').id
            inlet=m.site_pos[m.site(f'module_{(i+1)%6}/port_in').id]
            R=np.empty(9);mujoco.mju_quat2Mat(R,m.site_quat[old]);R=R.reshape(3,3)
            m.site_pos[variant]=m.site_pos[old]+R@(turn@inlet-inlet)
            q=np.empty(4);mujoco.mju_mulQuat(q,m.site_quat[old],quaternion);m.site_quat[variant]=q
        self.edges=[self.robot.outlet[i]@inverse(self.robot.inlet[(i+1)%6]) for i in range(6)]
        self.position_tolerance=1e-6
        self.angle_tolerance=1e-5

    def evaluate(self,q,bits,jacobian=False):
        T=np.eye(4);axes=[];anchors=[]
        for i in range(6):
            T=T@self.robot.tip[i]@rz(q[i]-self.robot.qref[i])
            axes.append(T[:3,2].copy());anchors.append(T[:3,3].copy())
            T=T@self.edges[i]@rz(bits[i]*self.delta[i])
        phi=Rotation.from_matrix(T[:3,:3]).as_rotvec()
        value=np.r_[T[:3,3]/.1,phi]
        if not jacobian:return value
        theta=np.linalg.norm(phi);S=skew(phi)
        coefficient=1/12 if theta<1e-5 else 1/theta**2-math.cos(theta/2)/(2*theta*math.sin(theta/2))
        left_inverse=np.eye(3)-.5*S+coefficient*S@S
        axes=np.array(axes);anchors=np.array(anchors)
        Jp=np.cross(axes,T[:3,3]-anchors).T/.1
        return np.concatenate((Jp,left_inverse@axes.T),axis=0)

    def validate_mujoco(self,q,bits):
        k=self.robot;k.registration[:]=np.asarray(bits)*self.delta
        roots,_,_=k.fk(q,5,np.eye(4))
        active=np.zeros_like(k.active)
        for i,b in enumerate(bits):active[k.new_eq[i] if b else k.old_eq[i]]=True
        frame=Frame(0.,'STATIC_CLOSURE',5,np.array(q),roots,active)
        k.frames=[frame]
        return k.active_constraint_errors()

    def solve(self,bits,seeds,max_nfev=200):
        best=None;attempts=0
        for seed in seeds:
            attempts+=1
            result=least_squares(lambda q:self.evaluate(q,bits),np.asarray(seed,dtype=float),
                jac=lambda q:self.evaluate(q,bits,True),max_nfev=max_nfev,
                ftol=1e-10,xtol=1e-10,gtol=1e-10)
            q=(result.x+math.pi)%(2*math.pi)-math.pi
            error=self.evaluate(q,bits)
            p=float(np.linalg.norm(error[:3])*.1);a=float(np.linalg.norm(error[3:]))
            candidate={'position_error_m':p,'angle_error_rad':a,'angle_error_deg':math.degrees(a),
                       'weighted_residual':float(np.linalg.norm(error)),'joint_angle_rad':q.tolist()}
            if best is None or candidate['weighted_residual']<best['weighted_residual']:best=candidate
            if p<=self.position_tolerance and a<=self.angle_tolerance:
                verification=self.validate_mujoco(q,bits)
                if (verification['max_active_weld_position_error_m']<=self.position_tolerance
                    and verification['max_active_weld_angle_error_rad']<=self.angle_tolerance):
                    return {**candidate,'status':'solution_found','attempts':attempts,'mujoco_verification':verification}
        return {**best,'status':'not_found','attempts':attempts}

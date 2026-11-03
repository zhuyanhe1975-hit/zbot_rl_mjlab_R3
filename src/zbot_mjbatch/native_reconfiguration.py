"""Experimental physical six-module state machine using passive geom adhesion.

The initial straight pose and candidate closure goal are saved asset data;
equality constraints are never copied into this model.
Batched local dynamics-response control is experimental, not a validated planner.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import os
from concurrent.futures import ThreadPoolExecutor
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import linprog
from mjbatch import Batch
from zbot_mjbatch.native_batch import DEFAULT_CONTACT_FORCE_N, make_native_batch_model
from zbot_mjbatch.native_safety import NativeSafetyAudit

ORDER = (0, 1, 3, 4)
# Forward reference uses 90 degrees: its search site is port_out rotated
# -90 degrees relative to the original 180-degree registration.
HEADTAIL_QUARTER = 1
HEADTAIL_CLOSED_Q = np.array([-0.7151050279933058, 2.663656161069831,
                              2.341859434055353, -1.093611347372558,
                              1.8997523394142295, -2.8300742095870532])
# Same 90-degree branch, solved by the reference planner with its head-tail
# port 30 mm back along local Z. Live magnetic dynamics still need validation.
HEADTAIL_PREALIGN_Q = np.array([-0.9788350721730494, 2.804096720258383,
                                2.1382758658591805, -0.9052821081039807,
                                1.9159236960408066, -2.7696887740241194])


def _solve_coupled_response(response, desired, drift, weights, *, damping=1e-7):
    """Find joint target changes that track target and retained connections.

    Each row is a physical displacement response over the same prediction
    horizon. `drift` is the predicted motion with no command change; omitting
    it makes an apparently zero command continue existing inertial motion.
    """
    a=np.asarray(response,dtype=float); w=np.asarray(weights,dtype=float)
    b=np.asarray(desired,dtype=float)-np.asarray(drift,dtype=float)
    if (a.ndim!=2 or a.shape[1]!=6 or b.shape!=(len(a),) or w.shape!=b.shape
            or not np.isfinite(a).all() or not np.isfinite(b).all()
            or not np.isfinite(w).all() or np.any(w<=0)):
        raise ValueError('invalid coupled response')
    weighted=a*w[:,None]; rhs=b*w
    return np.linalg.solve(weighted.T@weighted+damping*np.eye(6),weighted.T@rhs)


def _solve_retained_load_response(jac, predicted, current, target, bounds):
    """Maximize target load without increasing predicted retained loads."""
    keep=np.arange(6)!=target
    headroom=current[keep]-predicted[keep]
    # Seek a small predicted reduction on the most-loaded retained interface
    # to offset model error. If that is infeasible, keep the nonincrease rule.
    reserve=np.zeros(5)
    if current[target] <= np.max(current[keep])+5.:
        reserve[int(np.argmax(current[keep]))]=.05
    result=linprog(-jac[target],A_ub=jac[keep],
                   b_ub=headroom-reserve,bounds=bounds,method='highs')
    if not result.success:
        result=linprog(-jac[target],A_ub=jac[keep],
                       b_ub=headroom,bounds=bounds,method='highs')
    if not result.success or not np.isfinite(result.x).all():return None
    if np.any(predicted[keep]+jac[keep]@result.x>current[keep]+1e-7):return None
    return result.x

@dataclass
class Config:
    dt: float = .001
    control_dt: float = .02
    confirm_s: float = .05
    release_support_weight: float = .3
    rotation_support_weight: float = 2.
    rotation_prediction_steps: int = 20
    later_rotation_prediction_steps: int = 60
    later_peel_speed_rad_s: float = .04
    velocity_limit: float = 2 * math.pi
    joint_target_speed: float = .2
    position_tol: float = .001
    angle_tol: float = math.radians(1.)
    separation: float = .012
    axial_start_m: float = .010
    gap: float = .003
    contact_force_n: float = DEFAULT_CONTACT_FORCE_N  # per-contact capacity, uncalibrated




class NativeReconfiguration:
    def __init__(self, cfg=None):
        self.cfg=cfg or Config();c=self.cfg
        if (not math.isfinite(c.dt) or not math.isfinite(c.control_dt) or not math.isfinite(c.joint_target_speed) or not math.isfinite(c.contact_force_n) or c.contact_force_n<=0 or c.joint_target_speed<=0 or c.dt<=0 or c.control_dt<c.dt or abs(round(c.control_dt/c.dt)*c.dt-c.control_dt)>1e-10):
            raise ValueError('control period must be an integer number of physics steps')
        if (not isinstance(c.rotation_prediction_steps,int) or c.rotation_prediction_steps<1 or
                not isinstance(c.later_rotation_prediction_steps,int) or c.later_rotation_prediction_steps<1 or
                not math.isfinite(c.release_support_weight) or c.release_support_weight<=0 or
                not math.isfinite(c.rotation_support_weight) or c.rotation_support_weight<=0 or
                not math.isfinite(c.later_peel_speed_rad_s) or c.later_peel_speed_rad_s<=0):
            raise ValueError('invalid predictive control parameters')
        self.model,self.data=make_native_batch_model(timestep=c.dt,interaction_gap_m=c.gap,contact_force_n=c.contact_force_n)
        m=self.model;m.geom_solref[:,0]=.004
        m.actuator_gainprm[:,0]=15.;m.actuator_biasprm[:,1]=-15.;m.actuator_biasprm[:,2]=-1.5
        self.qadr=np.array([int(m.joint(f'module_{i}/joint{i+1}').qposadr[0]) for i in range(6)])
        self.dofs=np.array([int(m.joint(f'module_{i}/joint{i+1}').dofadr[0]) for i in range(6)])
        self.joints=np.array([m.joint(f'module_{i}/joint{i+1}').id for i in range(6)])
        self.acts=np.array([m.actuator(f'module_{i}/joint{i+1}').id for i in range(6)])
        # Initial target from the reference 90-degree forward branch. The
        # magnetic model must still physically reach and capture this pose.
        self.closed_q=HEADTAIL_CLOSED_Q.copy()
        self.curl_goal_q=HEADTAIL_PREALIGN_Q.copy()
        self.out=np.array([m.site(f'module_{i}/port_out').id for i in range(6)])
        self.inlet=np.array([m.site(f'module_{(i+1)%6}/port_in').id for i in range(6)])
        self.proxies={m.geom(f'physical_connector_{i}_adhesion_{side}_{j}').id:(i,side,j)
                      for i in range(6) for side in ('source','target') for j in range(4)}
        self.qref=self.data.qpos[self.qadr].copy();self.data.ctrl[self.acts]=self.qref
        self.phase='INITIAL_CHECK';self.target=5;self.phase_start=0.;self.step_count=0
        self.stage=0;self.confirm=0.;self.non_target_loss_s=np.zeros(6);self.registration=[0]*6;self.events=[];self.failure='';self.peak_torque=0.;self.peak_speed=0.
        self.max_penetration=0.;self.worst_contact=None;self.overload_s=np.zeros(6)
        self.predictor=None;self.prediction_calls=0;self.prediction_rank=None
        self.rotation_anchor=None
        self.load_prediction_calls=0;self.load_prediction_infeasible=0;self._load_pool=None
        self.release_search_calls=0;self.release_plan=None;self.release_search_anchor=None
        self.axial_advance_samples=0;self.axial_peak_lateral_m=0.;self.axial_peak_angle_rad=0.
        self.axial_armed=False;self.axial_prealign_sign=None;self.phase_axial_advance_samples=0
        self.audit=NativeSafetyAudit(m,self.data)
        initial_issues=self.audit.sample()
        self.initial_status=self.interfaces()
        self._event('start')
        if initial_issues:self._fail('initial safety audit: '+initial_issues[0]['kind'])

    def interfaces(self):
        """Contact is evidence of attraction, seated is a geometric test only."""
        d=self.audit.scratch;matches=[set() for _ in range(6)]
        for contact in d.contact:
            a=self.proxies.get(int(contact.geom1));b=self.proxies.get(int(contact.geom2))
            if a and b and a[0]==b[0] and a[1]!=b[1]:
                src=a if a[1]=='source' else b
                if contact.dist <= self.cfg.gap:matches[a[0]].add(src[2])
        result=[]
        for i in range(6):
            s,t=self.out[i],self.inlet[i];R=d.site_xmat[t].reshape(3,3)
            src,dst,offset=self.audit.frames[i]
            r0=d.site_xmat[src].reshape(3,3);r1=d.site_xmat[dst].reshape(3,3)
            physical0=d.site_xpos[src]+r0@offset;physical1=d.site_xpos[dst]+r1@offset
            p=R.T@(physical0-physical1);relative=R.T@d.site_xmat[s].reshape(3,3)
            pin_error=self.audit.four_pin_clearance_error(i)
            pin_excess=self.audit.four_pin_gate_excess(i)
            velocity0=np.empty(6);velocity1=np.empty(6)
            mujoco.mj_objectVelocity(self.model,d,mujoco.mjtObj.mjOBJ_SITE,src,velocity0,0)
            mujoco.mj_objectVelocity(self.model,d,mujoco.mjtObj.mjOBJ_SITE,dst,velocity1,0)
            linear0=velocity0[3:]+np.cross(velocity0[:3],r0@offset)
            linear1=velocity1[3:]+np.cross(velocity1[:3],r1@offset)
            relative_speed=float(np.linalg.norm(linear0-linear1))
            relative_angular_speed=float(np.linalg.norm(velocity0[:3]-velocity1[:3]))
            angles=[]
            for k in range(4):
                goal=Rotation.from_rotvec([0,0,k*math.pi/2]).as_matrix()
                angles.append(float(np.linalg.norm(Rotation.from_matrix(goal.T@relative).as_rotvec())))
            quarter=int(np.argmin(angles));angle=angles[quarter]
            result.append({'contacts':len(matches[i]),'position_m':float(np.linalg.norm(p)),
                'angle_rad':angle,'quarter':quarter,
                'four_pin_offset_m':float(pin_error),'four_pin_gate_excess_m':float(pin_excess),'relative_speed_m_s':relative_speed,
                'relative_angular_speed_rad_s':relative_angular_speed,
                'seated':bool(len(matches[i])==4 and np.linalg.norm(p)<self.cfg.position_tol and angle<self.cfg.angle_tol and pin_excess<=0.),
                'attracting':len(matches[i])>0})
        return result

    def _event(self,kind):
        self.events.append({'time_s':float(self.data.time),'kind':kind,'phase':self.phase,
                            'target_interface':self.target+1,'interfaces':self.interfaces()})

    def _transition(self,phase):
        if phase in ('CAPTURE','APPROACH'):
            self.axial_armed=False;self.axial_prealign_sign=None;self.phase_axial_advance_samples=0
        if phase=='ROTATE':
            # Keep the measured released-face offset while changing its
            # registration. Express it in the target frame so the offset
            # follows the floating target module.
            R=self.audit.scratch.site_xmat[self.inlet[self.target]].reshape(3,3)
            p0,p1=self._physical_points(self.target)
            self.rotation_anchor=R.T@(p0-p1)
        self.phase=phase;self.phase_start=float(self.data.time);self.confirm=0.;self._event('transition')

    def _fail(self,reason):
        self.failure=reason;self._transition('FAILED')

    def _physical_points(self,face):
        d=self.audit.scratch;src,dst,offset=self.audit.frames[face]
        r0=d.site_xmat[src].reshape(3,3);r1=d.site_xmat[dst].reshape(3,3)
        return d.site_xpos[src]+r0@offset, d.site_xpos[dst]+r1@offset

    def _axial_error(self, face, quarter):
        d=self.audit.scratch; R=d.site_xmat[self.inlet[face]].reshape(3,3)
        p0,p1=self._physical_points(face); local=R.T@(p0-p1)
        relative=R.T@d.site_xmat[self.out[face]].reshape(3,3)
        goal=Rotation.from_rotvec([0,0,quarter*math.pi/2]).as_matrix()
        return local, float(np.linalg.norm(Rotation.from_matrix(goal@relative.T).as_rotvec()))

    def _contact_wrenches(self, d):
        """Interface wrench on the source, including native proxy adhesion.

        Contact forces come from the solver; passive adhesion is a separate
        generalized force in MuJoCo and is added once for each proxy contact.
        """
        m=self.model; result=np.zeros((6,6)); force=np.zeros(6)
        src_bodies=m.site_bodyid[self.out]; dst_bodies=m.site_bodyid[self.inlet]
        for index,c in enumerate(d.contact):
            b0=int(m.geom_bodyid[c.geom1]); b1=int(m.geom_bodyid[c.geom2])
            for face in range(6):
                if b0==src_bodies[face] and b1==dst_bodies[face]:sign=-1.
                elif b1==src_bodies[face] and b0==dst_bodies[face]:sign=1.
                else:continue
                mujoco.mj_contactForce(m,d,index,force)
                frame=c.frame.reshape(3,3); f=sign*(frame.T@force[:3])
                torque=sign*(frame.T@force[3:])
                a=self.proxies.get(int(c.geom1));b=self.proxies.get(int(c.geom2))
                if a and b and a[0]==face and b[0]==face and a[1]!=b[1]:
                    f-=sign*self.cfg.contact_force_n*frame[0]
                R=d.site_xmat[self.out[face]].reshape(3,3)
                arm=c.pos-d.site_xpos[self.out[face]]
                result[face,:3]+=R.T@f
                result[face,3:]+=R.T@(torque+np.cross(arm,f))
                break
        return result

    def _static_force_jacobian(self):
        """Map incremental joint torque to interface wrench at fixed pose.

        Hold external support reactions fixed and solve incremental generalized
        equilibrium. Interface wrenches act on the source body; all six
        floating roots and joint coordinates participate in the solve.
        """
        m=self.model; d=self.audit.scratch; rows=[]; frames=[]
        for i in range(6):
            src,dst,offset=self.audit.frames[i]
            r0=d.site_xmat[src].reshape(3,3); r1=d.site_xmat[dst].reshape(3,3)
            p0=d.site_xpos[src]+r0@offset; p1=d.site_xpos[dst]+r1@offset
            p=(p0+p1)*.5; R=d.site_xmat[self.out[i]].reshape(3,3)
            j0=np.zeros((3,m.nv));j1=np.zeros_like(j0)
            a0=np.zeros_like(j0);a1=np.zeros_like(j0)
            mujoco.mj_jac(m,d,j0,a0,p,int(m.site_bodyid[src]))
            mujoco.mj_jac(m,d,j1,a1,p,int(m.site_bodyid[dst]))
            rows.append(np.vstack((R.T@(j0-j1),R.T@(a0-a1))))
            frames.append((R,p-d.site_xpos[self.out[i]]))
        J=np.vstack(rows); B=np.eye(m.nv)[:,self.dofs]
        response,_,rank,_=np.linalg.lstsq(J.T,-B,rcond=1e-10)
        if rank<36 or np.linalg.norm(J.T@response+B)>1e-7:
            return None
        response=response.reshape(6,6,6)
        for i,(R,arm) in enumerate(frames):
            response[i,3:]+=R.T@np.cross(arm[:,None],R@response[i,:3],axis=0)
        return response

    def _proxy_connections(self,d):
        active=np.zeros(6,dtype=bool)
        for contact in d.contact:
            a=self.proxies.get(int(contact.geom1));b=self.proxies.get(int(contact.geom2))
            if a and b and a[0]==b[0] and a[1]!=b[1] and contact.dist<=self.cfg.gap:
                active[a[0]]=True
        return active

    def _validate_release_trajectory(self,delta):
        """Replay a prospective servo ramp with the live state and 1 ms audit."""
        m=self.model;d=mujoco.MjData(m);mujoco.mj_copyData(d,m,self.data)
        audit=NativeSafetyAudit(m,d);audit.sample()
        lost=np.zeros(6);overload=self.overload_s.copy();base=self.qref.copy()
        steps=round(3./self.cfg.dt);control_steps=round(self.cfg.control_dt/self.cfg.dt)
        limits=self.audit.position_limits
        for k in range(steps):
            if k%control_steps==0:
                fraction=min(1.,(k+control_steps)/steps)
                d.ctrl[self.acts]=np.clip(base+fraction*delta,limits[0],limits[1])
            mujoco.mj_step(m,d)
            overload=np.where(np.abs(d.actuator_force[self.acts])>14.,overload+self.cfg.dt,0.)
            if (np.any(d.warning.number) or not np.isfinite(d.qpos).all() or
                    not np.isfinite(d.qvel).all() or
                    np.max(np.abs(d.qvel[self.dofs]))>self.cfg.velocity_limit or
                    np.max(np.abs(d.actuator_force[self.acts]))>20.+1e-6 or
                    np.any(overload>=1.)):
                return False
            if audit.sample():return False
            active=self._proxy_connections(audit.scratch)
            lost=np.where(active,0.,lost+self.cfg.dt)
            if np.any(lost[:5]+1e-12>=self.cfg.confirm_s):return False
            if lost[5]+1e-12>=self.cfg.confirm_s:return True
        return False

    def _search_headtail_trajectory(self,jac):
        """Screen physical servo ramps in batch and replay finalists at 1 ms."""
        m=self.model;d=self.data;n=128
        batch=Batch(m,num_sims=n,num_threads=min(16,len(os.sched_getaffinity(0))),forward=True)
        states=batch.bind('state');controls=batch.bind('ctrl')
        positions=batch.bind('qpos');velocities=batch.bind('qvel');warnings=batch.bind('warning')
        state=np.empty(batch.nstate)
        mujoco.mj_getState(m,d,state,mujoco.mjtState.mjSTATE_INTEGRATION)
        batch.reset();states[:]=state;controls[:]=d.ctrl
        rng=np.random.default_rng(91+self.release_search_calls)
        directions=rng.uniform(-1.,1.,(n,6));directions[0]=0.
        base=self.qref.copy();delta=(3.*min(.2,self.cfg.joint_target_speed))*directions
        lo,hi=self.audit.position_limits
        batch.forward()
        controls_count=round(3./self.cfg.control_dt)
        for k in range(controls_count):
            controls[:,self.acts]=np.clip(base+delta*((k+1)/controls_count),lo,hi)
            batch.step(nstep=round(self.cfg.control_dt/self.cfg.dt))
        batch.forward();self.release_search_calls+=1
        candidates=[]
        for i in range(n):
            if np.any(warnings[i,:,1]) or np.max(np.abs(velocities[i,self.dofs]))>self.cfg.velocity_limit:
                continue
            probe=mujoco.MjData(m);probe.qpos[:]=positions[i];probe.qvel[:]=velocities[i]
            mujoco.mj_forward(m,probe);active=self._proxy_connections(probe)
            if active[:5].all() and not active[5]:
                # Jacobian score is a ranking hint; every finalist still needs
                # an independent physics replay with all live protections.
                score=float(jac[5]@delta[i]-np.linalg.norm(jac[:5]@delta[i]))
                candidates.append((score,i))
        for _,i in sorted(candidates,reverse=True):
            if self._validate_release_trajectory(delta[i]):
                self.release_plan={'base':base,'delta':delta[i].copy(),'elapsed':0.}
                return True
        return False

    def _load_control(self, face):
        """Differentiate predicted interface loads against six servo targets."""
        if self.phase=='RELEASE_HEADTAIL' and self.release_plan is not None:
            plan=self.release_plan;plan['elapsed']=min(3.,plan['elapsed']+self.cfg.control_dt)
            lo,hi=self.audit.position_limits
            self.qref=np.clip(plan['base']+plan['delta']*(plan['elapsed']/3.),lo,hi)
            if plan['elapsed']>=3.:self.release_plan=None
            return
        m=self.model;epsilon=.01; horizon=20
        if self._load_pool is None:
            self._load_pool=ThreadPoolExecutor(max_workers=min(6,len(os.sched_getaffinity(0))))
        controls=np.repeat(self.qref[None,:],13,axis=0)
        for j in range(6):
            controls[1+2*j,j]+=epsilon;controls[2+2*j,j]-=epsilon
        def rollout(control):
            d=mujoco.MjData(m);mujoco.mj_copyData(d,m,self.data)
            d.ctrl[self.acts]=control
            mujoco.mj_step(m,d,nstep=horizon);mujoco.mj_forward(m,d)
            if np.any(d.warning.number):raise RuntimeError('load prediction MuJoCo warning')
            return self._contact_wrenches(d)
        try:w=np.asarray(list(self._load_pool.map(rollout,controls)))
        except RuntimeError as error:self._fail(str(error));return
        if not np.isfinite(w).all():self._fail('nonfinite load Jacobian');return
        load=np.maximum(w[:,:,2],0.)+np.linalg.norm(w[:,:,3:5],axis=2)/.05
        jac=np.stack([(load[1+2*j]-load[2+2*j])/(2*epsilon) for j in range(6)],axis=1)
        target=jac[face]; other=np.delete(jac,face,axis=0)
        # Maximize the target-load slope while damping load changes on the
        # retained interfaces. This is a local proposal, not a break guarantee.
        self.load_prediction_calls+=1
        step=self.cfg.joint_target_speed*self.cfg.control_dt
        if self.phase=='RELEASE_HEADTAIL':
            if self.release_search_anchor is None or np.max(np.abs(self.qref-self.release_search_anchor))>=.02:
                self.release_search_anchor=self.qref.copy()
                if self._search_headtail_trajectory(jac):
                    self._load_control(face)
                    return
            current_wrench=self._contact_wrenches(self.audit.scratch)
            current=np.maximum(current_wrench[:,2],0.)+np.linalg.norm(current_wrench[:,3:5],axis=1)/.05
            lo,hi=self.audit.position_limits
            bounds=list(zip(np.maximum(-step,lo-self.qref),np.minimum(step,hi-self.qref)))
            dq=_solve_retained_load_response(jac,load[0],current,face,bounds)
            if dq is None:
                self.load_prediction_infeasible+=1
                return
            self.qref=np.clip(self.qref+dq,lo,hi)
            return
        direction=np.linalg.solve(other.T@other+25.*np.eye(6),target)
        scale=np.max(np.abs(direction))
        if not np.isfinite(scale) or scale<1e-12:
            self._fail('no usable target load Jacobian');return
        self.qref=np.clip(self.qref+direction*(step/scale),*self.audit.position_limits)

    def _local_control(self, face, position_delta, rotation_delta):
        """Estimate a local floating-base response with 13 mjbatch rollouts.

        Predictions are not individually safety audited; live motion remains
        subject to the 1 ms audit and the approximation is unvalidated.
        """
        m=self.model;d=self.data
        if self.predictor is None:
            threads=min(13,len(os.sched_getaffinity(0)))
            batch=Batch(m,num_sims=13,num_threads=threads,forward=True)
            self.predictor=(batch,batch.bind('state'),batch.bind('ctrl'),
                            batch.bind('site_xpos'),batch.bind('site_xmat'),batch.bind('warning'),batch.bind('qvel'))
        batch,states,controls,positions,matrices,warnings,velocities=self.predictor
        state=np.empty(batch.nstate)
        mujoco.mj_getState(m,d,state,mujoco.mjtState.mjSTATE_INTEGRATION)
        batch.reset();states[:]=state
        controls[:]=d.ctrl
        delta=.02
        for j in range(6):
            controls[1+2*j,self.acts[j]]+=delta
            controls[2+2*j,self.acts[j]]-=delta
        prediction_steps=(self.cfg.later_rotation_prediction_steps if self.phase=='CAPTURE' or self.stage>=2 else self.cfg.rotation_prediction_steps) if self.phase in ('CAPTURE','ROTATE','APPROACH') else 20
        batch.forward();batch.step(nstep=prediction_steps);batch.forward()
        if np.any(warnings[:,:,1]):
            self._fail('batched dynamics prediction warning');return
        # Include relative motion of all magnetically retained interfaces, not
        # just the interface being moved. These are endpoint predictions, not
        # certified contact forces or a guarantee of future connectivity.
        current=self.audit.scratch
        responses=[]; desired=[]; drift=[]; weights=[]
        coordinated_peel=self.phase in ('RELEASE','RELEASE_HEADTAIL')
        coordinated=self.phase in ('CAPTURE','ROTATE','APPROACH') or (self.phase=='SEPARATE' and self.stage>=2) or coordinated_peel
        faces=range(6) if coordinated else (face,)
        for i in faces:
            src,dst,offset=self.audit.frames[i]
            if i==face:
                v=np.asarray(position_delta); omega=np.asarray(rotation_delta); weight=1.
            else:
                # Preserve the other current contacts while damping their
                # position/orientation errors towards their registered pose.
                p0,p1=self._physical_points(i)
                v=np.clip(-2.*(p0-p1),-.02,.02)
                r_in=current.site_xmat[self.inlet[i]].reshape(3,3)
                r_out=current.site_xmat[self.out[i]].reshape(3,3)
                goal=Rotation.from_rotvec([0,0,self.registration[i]*math.pi/2]).as_matrix()
                omega=r_in@np.clip(2.*Rotation.from_matrix(goal@(r_in.T@r_out).T).as_rotvec(),-.1,.1)
                # Later peel stages need enough freedom to build load. Keep
                # supporting interfaces in the objective with lower weight,
                # rather than locking the whole chain and preventing release.
                weight=self.cfg.release_support_weight if coordinated_peel else self.cfg.rotation_support_weight
            current0=current.site_xmat[src].reshape(3,3)
            current1=current.site_xmat[dst].reshape(3,3)
            current_p0=current.site_xpos[src]+current0@offset
            current_p1=current.site_xpos[dst]+current1@offset
            if self.phase in ('CAPTURE','APPROACH'):
                # Map the predicted full generalized velocities through the
                # physical point and angular velocity Jacobians. Floating root
                # velocities are included, never treated as actuated joints.
                jp0=np.zeros((3,m.nv));jr0=np.zeros((3,m.nv))
                jp1=np.zeros((3,m.nv));jr1=np.zeros((3,m.nv))
                mujoco.mj_jac(m,current,jp0,jr0,current_p0,int(m.site_bodyid[src]))
                mujoco.mj_jac(m,current,jp1,jr1,current_p1,int(m.site_bodyid[dst]))
                frame_motion=np.cross(jr1.T,current_p0-current_p1).T
                features=np.concatenate((velocities@(jp0-jp1-frame_motion).T,
                                         .04*velocities@(jr0-jr1).T),axis=1)*(prediction_steps*self.cfg.dt)
            else:
                r0=matrices[:,src].reshape(-1,3,3);r1=matrices[:,dst].reshape(-1,3,3)
                p0=positions[:,src]+np.einsum('nij,j->ni',r0,offset)
                p1=positions[:,dst]+np.einsum('nij,j->ni',r1,offset)
                relative=r0@np.transpose(r1,(0,2,1))
                current_relative=current0@current1.T
                features=np.concatenate((p0-p1-(current_p0-current_p1),
                    .04*Rotation.from_matrix(relative@current_relative.T).as_rotvec()),axis=1)
            if i==face and self.phase in ('CAPTURE','APPROACH'):
                target_frame=current.site_xmat[self.inlet[face]].reshape(3,3)
                features=np.concatenate((features[:,:3]@target_frame,features[:,3:]@target_frame),axis=1)
                v=target_frame.T@v;omega=target_frame.T@omega
            J=np.stack([(features[1+2*j]-features[2+2*j])/(2*delta) for j in range(6)],axis=1)
            responses.append(J);desired.append(np.r_[v,.04*omega]*(prediction_steps*self.cfg.dt))
            drift.append(features[0])
            if i==face and self.phase in ('CAPTURE','APPROACH'):
                weights.extend([12.,12.,8.,20.,20.,20.])
            else:weights.extend([weight]*6)
            if i==face:self.prediction_rank=int(np.linalg.matrix_rank(J,tol=1e-8))
        self.prediction_calls+=1
        try:
            if coordinated:
                if coordinated_peel:
                    # The target is intentionally loaded until it separates;
                    # do not cancel its natural baseline motion as if holding.
                    drift[face]=np.zeros(6)
                dq=_solve_coupled_response(np.vstack(responses),np.concatenate(desired),
                                           np.concatenate(drift),np.asarray(weights))
            else:
                # A seated magnetic interface initially has almost no short-
                # horizon mobility. Preserve the previous loading policy for
                # capture/peel; an all-interface pose objective would hold it
                # still and prevent the force from accumulating to release.
                J=responses[0]
                dq=J.T@np.linalg.solve(J@J.T+1e-7*np.eye(6),desired[0])
        except (ValueError,np.linalg.LinAlgError):
            self._fail('invalid coupled batched dynamics response');return
        rate=self.cfg.joint_target_speed
        dq=np.clip(dq,-rate*self.cfg.control_dt,rate*self.cfg.control_dt)
        limits=self.audit.position_limits
        self.qref=np.clip(self.qref+dq,limits[0],limits[1])

    def _approach_control(self, face, info, desired):
        # First-order velocity targets for the other five coordinates, while
        # correcting face-normal error. Tracking and actual
        # exponential convergence are not guaranteed by the finite-horizon
        # physical response model.
        d=self.audit.scratch;R=d.site_xmat[self.inlet[face]].reshape(3,3)
        p0,p1=self._physical_points(face);local=R.T@(p0-p1)
        relative=R.T@d.site_xmat[self.out[face]].reshape(3,3)
        goal=Rotation.from_rotvec([0,0,desired*math.pi/2]).as_matrix()
        delta=Rotation.from_matrix(goal@relative.T).as_rotvec()
        local_velocity=np.empty(3)
        local_velocity[:2]=np.clip(-8.*local[:2],-.005,.005)
        if np.linalg.norm(local[:2])<.00005:
            local_velocity[:2]=-2.*local[:2]
        # Coarse prealignment before the pin tips enter the square holes.
        # Once the axial approach starts, real pin/socket contact and native
        # adhesion provide their own guidance; keep correcting XY and rotation.
        aligned=(np.linalg.norm(local[:2]) <= .0005 and
                 np.linalg.norm(delta) <= math.radians(1.))
        # Establish a 15 mm axial standoff while aligning the other five
        # coordinates; only then begin the final axial approach.
        local_velocity[2]=float(np.clip(-.8*local[2],-.005,.005))
        if self.axial_armed and np.linalg.norm(local)<.009:
            local_velocity[2]=float(np.clip(local_velocity[2],-.0002,.0002))
        if np.linalg.norm(local) <= 3.*self.cfg.axial_start_m and not self.axial_armed:
            if self.axial_prealign_sign is None:
                self.axial_prealign_sign=1. if local[2]>=0 else -1.
            standoff=self.axial_prealign_sign*1.5*self.cfg.axial_start_m
            if aligned and abs(local[2])>=self.cfg.axial_start_m:
                self.axial_armed=True
            else:
                local_velocity[2]=float(np.clip(2.*(standoff-local[2]),-.005,.005))
        self._local_control(face,R@local_velocity,R@np.clip(2.*delta,-.05,.05))

    def _advance(self):
        c=self.cfg;d=self.audit.scratch;interfaces=self.interfaces();face=interfaces[self.target]
        if self.phase=='INITIAL_CHECK':
            if not all(x['attracting'] for x in interfaces[:5]) or interfaces[5]['attracting']:
                self._fail('initial topology is not a five-interface open chain');return
            self._transition('CURL')
        elif self.phase=='CURL':
            delta=np.clip(self.curl_goal_q-self.qref,-c.joint_target_speed*c.control_dt,c.joint_target_speed*c.control_dt)
            self.qref+=delta
            if np.max(np.abs(self.curl_goal_q-self.qref))<1e-5:self._transition('CAPTURE')
        elif self.phase in ('CAPTURE','SETTLE'):
            if self.phase=='CAPTURE' and (not self.axial_armed or not face['seated'] or face['quarter']!=HEADTAIL_QUARTER):
                self._approach_control(self.target,face,HEADTAIL_QUARTER)
                if self.phase=='FAILED':return
            expected=HEADTAIL_QUARTER if self.phase=='CAPTURE' else (3 if self.target in (0,1) else 1)
            stable=(face['seated'] and face['quarter']==expected and
                    face['relative_speed_m_s']<.03 and face['relative_angular_speed_rad_s']<.3)
            if self.phase=='CAPTURE':stable=stable and self.axial_armed and self.phase_axial_advance_samples>0
            self.confirm = self.confirm+c.control_dt if stable else 0.
            if self.confirm>=c.confirm_s:
                if self.phase=='CAPTURE':
                    self.registration[5]=HEADTAIL_QUARTER
                    self.target=ORDER[0];self.stage=0;self._transition('RELEASE')
                else:
                    self.registration[self.target]=expected
                    self.stage+=1
                    if self.stage<len(ORDER):self.target=ORDER[self.stage];self._transition('RELEASE')
                    else:self.target=5;self._transition('RELEASE_HEADTAIL')
        elif self.phase in ('RELEASE','RELEASE_HEADTAIL'):
            if face['attracting']:self._load_control(self.target)
            if self.phase=='FAILED':return
            self.confirm=self.confirm+c.control_dt if not face['attracting'] else 0.
            if self.confirm>=c.confirm_s:
                self._transition('UNFOLD' if self.phase=='RELEASE_HEADTAIL' else 'SEPARATE')
        elif self.phase=='SEPARATE':
            R=d.site_xmat[self.inlet[self.target]].reshape(3,3)
            if self.target==3:
                # The fourth face can recoil sideways during release. Align it
                # before recording the rotation anchor, while keeping enough
                # axial clearance for the pins and the approach controller.
                p0,p1=self._physical_points(self.target)
                local=R.T@(p0-p1)
                sign=1. if local[2]>=0. else -1.
                goal=np.array([0.,0.,sign*1.5*c.axial_start_m])
                moving=face['relative_speed_m_s']>=.03 or face['relative_angular_speed_rad_s']>=.3
                velocity=np.zeros(3) if moving else np.clip(2.*(goal-local),-.02,.02)
                self._local_control(self.target,R@velocity,np.zeros(3))
                if self.phase=='FAILED':return
                ready=(not face['attracting'] and not moving and
                       np.linalg.norm(local[:2])<=.0005 and abs(local[2])>=c.axial_start_m)
                self.confirm=self.confirm+c.control_dt if ready else 0.
                if self.confirm>=c.confirm_s:self._transition('ROTATE')
                return
            if self.stage<2:
                self._local_control(self.target,R[:,2]*.02,np.zeros(3))
                if self.phase=='FAILED':return
                if face['position_m']>=c.separation:self._transition('ROTATE')
                return
            # After a magnetic break the interface can recoil quickly. Use the
            # coupled predictor to damp that motion before starting rotation.
            moving=face['relative_speed_m_s']>=.03 or face['relative_angular_speed_rad_s']>=.3
            command=np.zeros(3) if moving or face['position_m']>=c.separation else R[:,2]*.02
            self._local_control(self.target,command,np.zeros(3))
            if self.phase=='FAILED':return
            self.confirm = self.confirm+c.control_dt if face['position_m']>=c.separation and not moving else 0.
            if self.confirm>=c.confirm_s:self._transition('ROTATE')
        elif self.phase=='ROTATE':
            R=d.site_xmat[self.inlet[self.target]].reshape(3,3)
            desired=(3 if self.target in (0,1) else 1)
            relative=R.T@d.site_xmat[self.out[self.target]].reshape(3,3)
            goal=Rotation.from_rotvec([0,0,desired*math.pi/2]).as_matrix()
            delta=Rotation.from_matrix(goal@relative.T).as_rotvec()
            p0,p1=self._physical_points(self.target)
            offset=R.T@(p0-p1)
            if self.rotation_anchor is None:
                self._fail('missing released-interface rotation anchor');return
            velocity=R@np.clip(2.*(self.rotation_anchor-offset),-.02,.02)
            self._local_control(self.target,velocity,R@np.clip(delta,-.1,.1))
            if self.phase=='FAILED':return
            # Once the target registration is the nearest quarter turn, the
            # approach controller finishes orientation at a standoff before
            # moving axially. This avoids a duplicated precision threshold.
            if face['quarter']==desired and not face['attracting']:
                self._transition('APPROACH')
        elif self.phase=='APPROACH':
            desired=(3 if self.target in (0,1) else 1)
            self._approach_control(self.target,face,desired)
            if self.phase=='FAILED':return
            if face['seated'] and face['quarter']==desired and self.axial_armed and self.phase_axial_advance_samples>0:self._transition('SETTLE')
        elif self.phase=='UNFOLD':
            self.qref+=np.clip(-self.qref,-c.joint_target_speed*c.control_dt,c.joint_target_speed*c.control_dt)
            if np.max(abs(d.qpos[self.qadr]))<.02 and np.max(abs(d.qvel[self.dofs]))<.05:self._transition('FINAL_HOLD')
        elif self.phase=='FINAL_HOLD':
            m=self.model
            root_xyz=np.array([d.xpos[m.jnt_bodyid[m.joint(f'module_{i}/module_root').id]] for i in range(6)])
            axis=root_xyz[-1]-root_xyz[0];axis/=max(np.linalg.norm(axis),1e-12)
            collinearity=float(np.max(np.linalg.norm(np.cross(root_xyz-root_xyz[0],axis),axis=1)))
            correct=(all(interfaces[i]['seated'] for i in range(5)) and not interfaces[5]['attracting']
                     and interfaces[5]['position_m']>=.05 and np.max(abs(d.qpos[self.qadr]))<.02
                     and np.max(abs(d.qvel[self.dofs]))<.05 and collinearity<.005)
            correct=correct and all(interfaces[i]['quarter']==(3 if i in (0,1) else 1) for i in ORDER)
            self.confirm=self.confirm+c.control_dt if correct else 0.
            if self.confirm>=c.confirm_s:self._transition('DONE')

    def _track_non_target_connections(self, active=None):
        # Only actual proxy contacts determine lost attraction. Geometric
        # offsets and contact penetration remain diagnostics; do not terminate
        # a run simply because a pin/face penetrates in the soft solver.
        if active is None:
            active=np.zeros(6,dtype=bool)
            for contact in self.audit.scratch.contact:
                a=self.proxies.get(int(contact.geom1));b=self.proxies.get(int(contact.geom2))
                if a and b and a[0]==b[0] and a[1]!=b[1] and contact.dist<=self.cfg.gap:
                    active[a[0]]=True
        required=range(5) if self.phase in ('INITIAL_CHECK','CURL','CAPTURE','FINAL_HOLD','UNFOLD') else (i for i in range(6) if i!=self.target)
        required=set(required)
        for i in range(6):
            if i in required and not active[i]:self.non_target_loss_s[i]+=self.cfg.dt
            else:self.non_target_loss_s[i]=0.
        lost=[i for i in required if self.non_target_loss_s[i]+1e-12>=self.cfg.confirm_s]
        if lost:self._fail('non-target magnetic contact lost: '+','.join(str(i+1) for i in lost))

    def step(self):
        if self.phase in ('FAILED','DONE'):return
        m=self.model;d=self.data;c=self.cfg
        if self.step_count % round(c.control_dt/c.dt)==0:self._advance()
        if self.phase in ('FAILED','DONE'):return
        axial_phase=self.phase=='CURL' or (self.phase in ('CAPTURE','APPROACH') and self.axial_armed)
        axial_quarter=HEADTAIL_QUARTER if self.phase in ('CURL','CAPTURE') else (3 if self.target in (0,1) else 1)
        before_axial=self._axial_error(self.target,axial_quarter)[0] if axial_phase else None
        d.ctrl[self.acts]=self.qref;mujoco.mj_step(m,d);self.step_count+=1
        self.peak_torque=max(self.peak_torque,float(np.max(abs(d.actuator_force))))
        self.peak_speed=max(self.peak_speed,float(np.max(abs(d.qvel[self.dofs]))))
        self.overload_s=np.where(np.abs(d.actuator_force[self.acts])>14.,self.overload_s+c.dt,0.)
        for contact in d.contact:
            depth=max(0.,-float(contact.dist))
            if depth>self.max_penetration:
                self.max_penetration=depth
                self.worst_contact={'time_s':float(d.time),'geom1':m.geom(int(contact.geom1)).name,'geom2':m.geom(int(contact.geom2)).name,'depth_m':depth}
        issues=self.audit.sample()
        if issues:self._fail('safety audit: '+issues[0]['kind'])
        if self.phase=='FAILED':return
        if axial_phase:
            local,angle=self._axial_error(self.target,axial_quarter)
            if np.linalg.norm(local)<=c.axial_start_m and abs(local[2]) < abs(before_axial[2])-1e-8:
                lateral=float(np.linalg.norm(local[:2]))
                self.axial_advance_samples+=1
                self.phase_axial_advance_samples+=1
                self.axial_peak_lateral_m=max(self.axial_peak_lateral_m,lateral)
                self.axial_peak_angle_rad=max(self.axial_peak_angle_rad,angle)
        self._track_non_target_connections()
        if self.phase=='FAILED':return
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all() or np.any(d.warning.number):self._fail('MuJoCo warning or nonfinite state')
        elif self.peak_speed>c.velocity_limit:self._fail('joint speed limit exceeded')
        elif self.peak_torque>20.+1e-6:self._fail('joint torque limit exceeded')
        elif np.any(self.overload_s>=1.):self._fail('continuous joint torque limit exceeded for 1 s')

    def report(self):
        return {'success':self.phase=='DONE' and self.audit.total_violations==0 and self.audit.samples==self.step_count+1,'phase':self.phase,'failure':self.failure,
                'time_s':float(self.data.time),'target_interface':self.target+1,'events':self.events,
                'initial_interfaces':self.initial_status,'interfaces':self.interfaces(),
                'peak_torque_Nm':self.peak_torque,'peak_joint_speed_rad_s':self.peak_speed,
                'max_penetration_m':self.max_penetration,'worst_contact':self.worst_contact,
                'physics_dt_s':self.cfg.dt,'joint_target_speed_rad_s':self.cfg.joint_target_speed,
                'geom_adhesion_capacity_N_per_contact_assumed':self.cfg.contact_force_n,
                'non_target_contact_loss_s':self.non_target_loss_s.tolist(),
                'seating_position_tolerance_m':self.cfg.position_tol,'seating_angle_tolerance_deg':math.degrees(self.cfg.angle_tol),'kp':15.,'kd':1.5,'weld_count':self.model.neq,
                'safety_audit':self.audit.report(),
                'batch_response_predictions':self.prediction_calls,'last_response_rank':self.prediction_rank,
                'load_jacobian_predictions':self.load_prediction_calls,
                'load_jacobian_infeasible':self.load_prediction_infeasible,
                'headtail_trajectory_searches':self.release_search_calls,
                'late_axial_approach':{'start_m':self.cfg.axial_start_m,
                    'armed':self.axial_armed,
                    'status':'diagnostic_only',
                    'advance_samples':self.axial_advance_samples,
                    'max_lateral_m':self.axial_peak_lateral_m,
                    'max_angle_deg':math.degrees(self.axial_peak_angle_rad),
                    'minimum_measured_advance_m_per_step':1e-8},
                'control_settings':{'release_support_weight':self.cfg.release_support_weight,
                    'rotation_support_weight':self.cfg.rotation_support_weight,
                    'rotation_prediction_steps':self.cfg.rotation_prediction_steps,
                    'later_rotation_prediction_steps':self.cfg.later_rotation_prediction_steps,
                    'later_peel_speed_rad_s':self.cfg.later_peel_speed_rad_s},
                'approach_command':'five-coordinate prealignment at axial standoff followed by velocity-Jacobian axial approach; actual discrete advance audited',
                'caveat':'batched local dynamics response is experimental and candidate paths are not safety audited; live discrete contact audit has geometric/filtering limits; uncalibrated adhesion'}

    def close(self):
        if self._load_pool is not None:
            self._load_pool.shutdown(wait=True,cancel_futures=True)
            self._load_pool=None

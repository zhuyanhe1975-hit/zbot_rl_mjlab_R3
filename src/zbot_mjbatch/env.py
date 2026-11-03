"""mjbatch-only batched first head-tail closure environment."""
import math
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from mjbatch import Batch
from .model import make_model
from .monitor import Monitor

class HeadTailBatch:
    """Independent CPU worlds; physics and all safety checks run at 1 ms."""
    dt=.001; control_dt=.02; max_time=20.; action_dim=7
    # Refined viewer CURL: 30 deg/s * 12. No acceleration limit.
    speed_limit=math.radians(30.*12.)
    kp=15.;kd=1.5;torque_limit=20.
    policy_active_observation_index=31
    def __init__(self,num_envs=64,num_threads=8,seed=42):
        if num_envs<1 or num_threads<1:raise ValueError('positive worlds and threads required')
        self.num_envs=num_envs;self.num_threads=min(num_threads,num_envs);self.rng=np.random.default_rng(seed)
        self.model,initial=make_model(self.dt);m=self.model
        self.batch=Batch(m,num_envs,self.num_threads,forward=True);b=self.batch
        self.state=b.bind('state');self.qpos=b.bind('qpos');self.qvel=b.bind('qvel');self.ctrl=b.bind('ctrl');self.eq=b.bind('eq_active')
        self.site_pos=b.bind('site_xpos');self.site_mat=b.bind('site_xmat');self.axes=b.bind('xaxis');self.anchors=b.bind('xanchor')
        self.sensors=b.bind('sensordata');self.forces=b.bind('actuator_force')
        self.warnings=b.bind('warning');self.warning_baseline=np.zeros((num_envs,7),dtype=np.int32)
        self.qids=np.array([m.joint(f'module_{i}/joint{i+1}').qposadr[0] for i in range(6)])
        self.dofs=np.array([m.joint(f'module_{i}/joint{i+1}').dofadr[0] for i in range(6)])
        self.joints=np.array([m.joint(f'module_{i}/joint{i+1}').id for i in range(6)])
        self.actuators=np.array([m.actuator(f'module_{i}/joint{i+1}').id for i in range(6)])
        self.weld=np.array([m.equality(f'connector_{i}').id for i in range(6)])
        self.head=m.site('module_0/port_in').id;self.tail=m.site('module_5/search_port').id
        self.velids=np.array([[m.sensor(f'{side}_{kind}').adr[0]+j for j in range(3)] for side in ('head','tail') for kind in ('linear','angular')])
        self.initial=np.empty(b.nstate);mujoco.mj_getState(m,initial,self.initial,mujoco.mjtState.mjSTATE_INTEGRATION)
        self.monitor=Monitor(m,self.num_threads);self.audit=np.zeros((num_envs,10));self.loads=np.zeros((num_envs,6))
        self.qtarget=np.zeros((num_envs,6));self.velocity=np.zeros((num_envs,6));self.action=np.zeros((num_envs,7))
        self.elapsed=np.zeros(num_envs);self.capture_time=np.zeros(num_envs);self.settle_time=np.zeros(num_envs)
        self.phase=np.zeros(num_envs,dtype=np.int8);self.overload=np.zeros((num_envs,6))
        self.goal=np.array([-.7151050279933058,2.663656161069831,2.341859434055353,
                            -1.093611347372558,1.8997523394142295,-2.8300742095870532])
        self.last_peak_N=np.zeros(num_envs);self.last_peak_torque=np.zeros(num_envs)
        self.reset(np.arange(num_envs))

    def reset(self,ids=None):
        ids=np.arange(self.num_envs) if ids is None else np.asarray(ids,dtype=np.int64)
        if not len(ids):return
        self.batch.reset(ids);self.state[ids]=self.initial
        self.eq[ids,:]=False;self.eq[np.ix_(ids,self.weld[:5])]=True
        self.batch.forward(ids)
        self.warning_baseline[ids]=self.warnings[ids,:,1]
        self.qtarget[ids]=self.qpos[np.ix_(ids,self.qids)];self.velocity[ids]=0;self.action[ids]=0
        self.phase[ids]=0;self.elapsed[ids]=0;self.capture_time[ids]=0;self.settle_time[ids]=0;self.overload[ids]=0
        self.loads[ids]=0;self.last_peak_N[ids]=0;self.last_peak_torque[ids]=0

    def geometry(self):
        Rh=self.site_mat[:,self.head].reshape(-1,3,3);Rt=self.site_mat[:,self.tail].reshape(-1,3,3)
        delta=self.site_pos[:,self.head]-self.site_pos[:,self.tail]
        rot=Rotation.from_matrix(np.swapaxes(Rh,1,2)@Rt).as_rotvec()
        return delta,rot,Rh

    def observation(self):
        delta,rot,R=self.geometry();local=np.einsum('nji,nj->ni',R,delta)
        phase=np.eye(3,dtype=np.float32)[self.phase.clip(0,2)]
        return np.concatenate((self.qpos[:,self.qids]/np.pi,self.qvel[:,self.dofs]/8.,local/.5,rot/np.pi,
            self.loads[:,:5]/100.,self.action,
            phase,(self.elapsed/self.max_time)[:,None],self.velocity/self.speed_limit),axis=1).astype(np.float32)

    def plan(self,delta,rot,R):
        q=self.qpos[:,self.qids];axis=self.axes[:,self.joints];anchor=self.anchors[:,self.joints]
        source=self.site_pos[:,self.tail];Jp=np.cross(axis,source[:,None,:]-anchor).transpose(0,2,1)
        J=np.concatenate((Jp*5.,axis.transpose(0,2,1)*.5),axis=1)
        orientation=-np.einsum('nij,nj->ni',R,rot);goal=np.concatenate((delta*5.,orientation*.5),axis=1)
        JT=J.transpose(0,2,1);vpose=np.linalg.solve(JT@J+.03*np.eye(6)[None],(JT@goal[...,None]))[...,0]
        gap=np.linalg.norm(delta,axis=1);blend=np.clip((.18-gap)/.12,0.,1.)[:,None]
        limit_speed=self.speed_limit
        v=(1.-blend)*np.clip(1.4*(self.goal-q),-limit_speed,limit_speed)+blend*np.clip(vpose,-limit_speed,limit_speed)
        v=np.clip(v*(1.+.5*self.action[:,6,None])+.2*self.action[:,:6],-limit_speed,limit_speed)
        # Reduce requested speed as the measured non-target weld load rises;
        # the 80 N failure threshold remains independent of the motor torque.
        v*=np.clip((65.-self.loads[:,:5].max(axis=1))/30.,0.,1.)[:,None]
        return np.where(np.isfinite(v).all(axis=1)[:,None],v,0.)

    def step(self,action):
        action=np.asarray(action,dtype=np.float64)
        if action.shape!=(self.num_envs,7) or not np.isfinite(action).all():raise ValueError('action must be finite (num_envs,7)')
        self.action[:]=np.clip(action,-1,1);alive=np.ones(self.num_envs,bool);success=np.zeros_like(alive)
        elapsed_before=self.elapsed.copy()
        failed=np.zeros_like(alive);reason=np.zeros(self.num_envs,dtype=np.int8);peak=np.zeros(self.num_envs)
        for _ in range(round(self.control_dt/self.dt)):
            delta,rot,R=self.geometry();gap=np.linalg.norm(delta,axis=1);angle=np.linalg.norm(rot,axis=1)
            motion=(self.phase==1)&alive
            desired=self.plan(delta,rot,R);self.velocity[:]=np.where(motion[:,None],desired,0.)
            q=self.qpos[:,self.qids];tracking_limit=self.torque_limit/self.kp
            self.qtarget=np.where(motion[:,None],np.clip(self.qtarget+self.velocity*self.dt,q-tracking_limit,q+tracking_limit),self.qtarget)
            self.ctrl[:,self.actuators]=self.qtarget
            demand=self.kp*(self.ctrl[:,self.actuators]-q)-self.kd*self.qvel[:,self.dofs]
            requested=alive&~np.isfinite(demand).all(axis=1)
            failed|=requested;reason=np.where(requested,3,reason);alive&=~requested
            if not alive.any():break
            # Preserve PD semantics while respecting the physical actuator
            # torque bound, without introducing an acceleration constraint.
            self.ctrl[:,self.actuators]=q+(np.clip(demand,-self.torque_limit,self.torque_limit)+self.kd*self.qvel[:,self.dofs])/self.kp
            self.batch.step(np.flatnonzero(alive).astype(np.int64));self.elapsed[alive]+=self.dt
            self.monitor.measure(self.state,self.audit);self.loads[:]=self.audit[:,:6]
            peak=np.where(alive,np.maximum(peak,self.loads[:,:5].max(axis=1)),peak)
            self.last_peak_N=np.maximum(self.last_peak_N,peak)
            self.last_peak_torque=np.where(alive,np.maximum(self.last_peak_torque,self.audit[:,6]),self.last_peak_torque)
            delta,rot,R=self.geometry();gap=np.linalg.norm(delta,axis=1);angle=np.linalg.norm(rot,axis=1)
            aligned=(gap<.001)&(angle<math.radians(2.))
            self.overload=np.where(np.abs(self.forces[:,self.actuators])>14.,self.overload+self.dt,0.)
            loadfail=(self.loads[:,:5]>=80.).any(axis=1)|((self.phase==2)&(self.loads[:,5]>=100.))
            torquefail=(self.audit[:,6]>20.00001)|(self.overload>=1.).any(axis=1)
            invalid=(self.audit[:,9]>0)|~np.isfinite(self.qpos).all(axis=1)|~np.isfinite(self.qvel).all(axis=1)|~self.eq[:,self.weld[:5]].all(axis=1)
            warning=(self.warnings[:,:,1]>self.warning_baseline).any(axis=1)
            newfail=alive&(loadfail|torquefail|invalid|warning)
            reason=np.where(newfail,np.where(loadfail,1,np.where(torquefail,3,np.where(warning,7,4))),reason)
            failed|=newfail;alive&=~newfail
            self.phase[(self.phase==0)&(self.elapsed>=.25)&alive]=1
            velocity=self.sensors[:,self.velids]
            linear=np.linalg.norm(velocity[:,2]-velocity[:,0],axis=1);angular=np.linalg.norm(velocity[:,3]-velocity[:,1],axis=1)
            eligible=alive&(self.phase==1)&aligned&(linear<.1)&(angular<.6)
            self.capture_time=np.where(eligible,self.capture_time+self.dt,0.)
            capture=eligible&(self.capture_time>=.03);self.phase[capture]=2;self.eq[capture,self.weld[5]]=True
            self.qtarget[capture]=self.qpos[np.ix_(np.flatnonzero(capture),self.qids)];self.velocity[capture]=0.
            settling=alive&(self.phase==2)&~capture;self.settle_time[settling]+=self.dt
            completed=settling&(self.settle_time>=.25);success|=completed;alive&=~completed
            timed=alive&(self.elapsed>=self.max_time-1e-9);failed|=timed;reason=np.where(timed,5,reason);alive&=~timed
            if not alive.any():break
        done=success|failed;duration=self.elapsed.copy();phase=self.phase.copy()
        delta,rot,_=self.geometry();gap=np.linalg.norm(delta,axis=1);angle=np.linalg.norm(rot,axis=1)
        pose=np.exp(-gap/.15-angle/.75);pose=np.where((phase==1)&~failed,pose,0.)*.1*self.control_dt
        time=np.where(success,self.max_time/np.maximum(duration,self.dt),0.);penalty=-20.*failed
        elapsed=-.2*(self.elapsed-elapsed_before);reward=(time+pose+penalty+elapsed).astype(np.float32)
        info={'success':success.copy(),'failed':failed.copy(),'reason':reason.copy(),'duration_s':duration,
              'peak_load_N':self.last_peak_N.copy(),'peak_torque_Nm':self.last_peak_torque.copy(),
              'reward_time':time,'reward_pose':pose,'penalty_failure':penalty,'penalty_elapsed':elapsed}
        self.reset(np.flatnonzero(done));return self.observation(),reward,done,info

    def close(self):self.monitor.close()

"""Per-interface physical wrench, passive failure and selective separation state."""
import math

import mujoco
import torch
from mjlab.utils.lab_api.math import quat_apply,quat_apply_inverse,quat_conjugate,quat_mul


def port_wrench(generalized,root_quaternion,root_position,port_position,port_quaternion):
    force=generalized[:,:3]
    root_moment=quat_apply(root_quaternion,generalized[:,3:])
    moment=root_moment-torch.cross(port_position-root_position,force,dim=-1)
    return quat_apply_inverse(port_quaternion,force),quat_apply_inverse(port_quaternion,moment)


def tilt_load(moment,radius):
    """Equivalent tensile demand caused by in-plane bending moments only, N."""
    return moment[...,:2].norm(dim=-1)/radius


def peel_load(force,moment,radius):
    """Equivalent opening demand in N in the SOURCE port_out local frame.

    +Z points outward from the source into the next module. Tensile weld
    reaction on the source is +Z. Compression, XY shear and Z twist do not
    contribute to the magnetic peel criterion; the engaged weld still resists
    them, representing the locating protrusions.
    """
    return force[...,2].clamp_min(0.)+tilt_load(moment,radius)


class ConnectorState:
    def __init__(self,env):
        self.env=env;self.cfg=env.cfg
        if not 0<=self.cfg.target_connector<6:
            raise ValueError('target_connector must be 0..5 (5 is head-tail)')
        for name in ('break_force','connector_radius','separation_distance','post_separation_wait_s','torque_limit','safety_penalty_buffer'):
            value=getattr(self.cfg,name)
            if not math.isfinite(value) or value<=0:
                raise ValueError(f'{name} must be positive and finite')
        if not 0<=self.cfg.minimum_other_margin<1:
            raise ValueError('minimum_other_margin must be in [0,1)')
        self.modules=[env.scene[f'module_{i}'] for i in range(6)]
        self.eq_ids=torch.tensor([mujoco.mj_name2id(env.sim.mj_model,mujoco.mjtObj.mjOBJ_EQUALITY,
                                                  f'connector_{i}') for i in range(6)],device=env.device)
        if (self.eq_ids<0).any():
            raise ValueError('Missing six physical connector welds')
        self.out_sites=[m.find_sites('port_out')[0][0] for m in self.modules]
        self.in_sites=[m.find_sites('port_in')[0][0] for m in self.modules]
        n=env.num_envs
        self.worlds=torch.arange(n,device=env.device)
        self.root_dofs=torch.stack([m.indexing.free_joint_v_adr.long() for m in self.modules])
        self.force=torch.zeros(n,6,3,device=env.device)
        self.moment=torch.zeros_like(self.force)
        self.utilization=torch.zeros(n,6,device=env.device)
        self.peak_utilization=torch.zeros_like(self.utilization)
        self.break_utilization=torch.zeros_like(self.utilization)
        self.peak_target_tilt_load=torch.zeros(n,device=env.device)
        self.max_target_gap=torch.zeros(n,device=env.device)
        self.gap_progress=torch.zeros(n,device=env.device)
        self.separation_progress_delta=torch.zeros(n,device=env.device)
        self.wait_count=torch.zeros(n,dtype=torch.long,device=env.device)
        self.waiting=torch.zeros(n,dtype=torch.bool,device=env.device)
        self.succeeded=torch.zeros(n,dtype=torch.bool,device=env.device)
        self.just_succeeded=torch.zeros_like(self.succeeded)
        self.wrong_break=torch.zeros_like(self.succeeded)
        self.safety_margin_violation=torch.zeros_like(self.succeeded)
        self.invalid_state=torch.zeros_like(self.succeeded)
        self.other_ids=[i for i in range(6) if i!=self.cfg.target_connector]
        self.last_substep=None

    def target_indices(self):
        return torch.full((self.env.num_envs,), self.cfg.target_connector, device=self.env.device, dtype=torch.long)

    def other_mask(self):
        return torch.arange(6, device=self.env.device)[None] != self.target_indices()[:, None]

    def constraint_ids(self):
        return self.eq_ids[None].expand(self.env.num_envs,-1)

    def connected(self):
        return self.env.sim.data.eq_active.gather(1,self.constraint_ids()).bool()

    def begin_step(self):
        self.peak_utilization.zero_();self.peak_target_tilt_load.zero_();self.just_succeeded.zero_();self.separation_progress_delta.zero_()

    def reset(self,ids):
        ids=slice(None) if ids is None else ids
        # Outer indexing: reset exactly the selected worlds, never paired indices.
        if isinstance(ids,slice):
            self.env.sim.data.eq_active[:,self.eq_ids]=True
        else:
            self.env.sim.data.eq_active[ids[:,None],self.eq_ids[None]]=True
        for value in (self.force,self.moment,self.utilization,self.peak_utilization,
                      self.break_utilization,self.peak_target_tilt_load,self.max_target_gap,self.gap_progress,self.separation_progress_delta,self.wait_count,self.waiting,self.succeeded,self.just_succeeded,self.wrong_break,self.safety_margin_violation,self.invalid_state):
            value[ids]=0

    def geometry(self,index=None):
        i=self.cfg.target_connector if index is None else index
        a,b=self.modules[i],self.modules[(i+1)%6]
        sa,sb=self.out_sites[i],self.in_sites[(i+1)%6]
        p0,q0=a.data.site_pos_w[:,sa],a.data.site_quat_w[:,sa]
        p1,q1=b.data.site_pos_w[:,sb],b.data.site_quat_w[:,sb]
        relative=quat_mul(quat_conjugate(q0),q1)
        relative=torch.where(relative[:,:1]<0,-relative,relative)
        return {'delta':p1-p0,'distance':(p1-p0).norm(dim=-1),'q0':q0,
                'relative_quat':relative,
                'angle':2.*torch.atan2(relative[:,1:].norm(dim=-1),relative[:,0].abs()),
                'dv':b.data.site_lin_vel_w[:,sb]-a.data.site_lin_vel_w[:,sa],
                'dw':b.data.site_ang_vel_w[:,sb]-a.data.site_ang_vel_w[:,sa]}

    def outgoing_port_frames(self):
        return (torch.stack([m.data.site_pos_w[:,i] for m,i in zip(self.modules,self.out_sites)],1),
                torch.stack([m.data.site_quat_w[:,i] for m,i in zip(self.modules,self.out_sites)],1))

    def update(self):
        if self.last_substep==self.env._sim_step_counter:
            return
        data=self.env.sim.data;efc=data.efc
        if efc.J.shape[1]==1:
            raise RuntimeError('Disconnect wrench measurement requires a dense constraint Jacobian')
        # Extract all six welds in one batch. The physical solve remains the
        # native Warp solve; this only batches GPU reaction-force readout.
        rows=torch.arange(efc.force.shape[1],device=self.env.device)[None,None]
        active=self.connected()
        mask=((rows<data.nefc[:,None,None]) & (efc.type[:,None,:]==0)
              & (efc.id[:,None,:]==self.constraint_ids()[:,:,None]))
        first=torch.where(mask,rows,efc.force.shape[1]).amin(-1)
        present=(first<data.nefc[:,None]) & active
        indices=(first[:,:,None]+torch.arange(6,device=self.env.device)[None,None]).clamp_max(efc.force.shape[1]-1)
        multiplier=efc.force.gather(1,indices.flatten(1)).reshape(-1,6,6)*present[:,:,None]
        jacobian=efc.J[self.worlds[:,None,None,None],indices[:,:,:,None],self.root_dofs[None,:,None,:]]
        generalized=(jacobian*multiplier[:,:,:,None]).sum(2)
        port,frame=self.outgoing_port_frames()
        root_q=torch.stack([m.data.root_link_quat_w for m in self.modules],1)
        root_p=torch.stack([m.data.root_link_pos_w for m in self.modules],1)
        force,moment=port_wrench(generalized.reshape(-1,6),root_q.reshape(-1,4),
                                  root_p.reshape(-1,3),port.reshape(-1,3),frame.reshape(-1,4))
        self.force.copy_(force.reshape(-1,6,3));self.moment.copy_(moment.reshape(-1,6,3))
        current=peel_load(self.force,self.moment,self.cfg.connector_radius)/self.cfg.break_force
        self.utilization.copy_(torch.where(active,current,0.))
        self.invalid_state |= ~torch.isfinite(self.utilization).all(-1)
        self.peak_utilization.copy_(torch.maximum(self.peak_utilization,self.utilization))
        target=self.target_indices()
        target_tilt=torch.where(active[self.worlds,target],tilt_load(self.moment[self.worlds,target],self.cfg.connector_radius),0.)
        self.peak_target_tilt_load.copy_(torch.maximum(self.peak_target_tilt_load,target_tilt))
        if not self.cfg.enable_weld_failure:
            # Static loading mode: measure all six weld reactions but never
            # deactivate constraints, start separation, or latch rupture loads.
            self.safety_margin_violation |= ((self.utilization >1.-self.cfg.minimum_other_margin) & self.other_mask()).any(-1)
            self.last_substep=self.env._sim_step_counter
            return
        broken=active & (self.utilization>=1.)
        self.break_utilization.copy_(torch.where(broken,self.utilization,self.break_utilization))
        next_active=active & ~broken
        self.env.sim.data.eq_active.scatter_(1,self.constraint_ids(),next_active.to(self.env.sim.data.eq_active.dtype))
        self.wrong_break |= (~next_active & self.other_mask()).any(-1)
        # A break is passive and irreversible within the episode. No action
        # is a latch/unlock command, and no target is artificially weakened.
        self.update_separation(next_active)
        self.last_substep=self.env._sim_step_counter

    def update_separation(self,next_active):
        # Count only substeps AFTER the crossing event. Once started, neither
        # a shrinking gap nor a newly sampled policy action cancels the wait.
        self.wait_count.add_(self.waiting.long())
        distance=self.geometry()['distance']
        self.max_target_gap.copy_(torch.maximum(self.max_target_gap,distance))
        progress=torch.where(~next_active[:,self.cfg.target_connector],
                             (distance/self.cfg.separation_distance).clamp(0.,1.),0.)
        progress=torch.where(self.waiting,self.gap_progress,progress)
        self.separation_progress_delta.add_(progress-self.gap_progress)
        self.gap_progress.copy_(progress)
        separated=(~next_active[:,self.cfg.target_connector]
                   & (distance>self.cfg.separation_distance))
        self.waiting |= separated
        # Compare utilization, rather than a rounded 1-u margin, so exactly
        # 64/80 load remains on the allowed .2-margin boundary in float32.
        unsafe=(1.-self.other_margin())>(1.-self.cfg.minimum_other_margin)
        self.safety_margin_violation |= unsafe
        achieved=(separated & next_active[:,self.other_ids].all(-1) & ~unsafe)
        self.just_succeeded |= achieved & ~self.succeeded & ~self.wrong_break
        self.succeeded |= achieved & ~self.wrong_break

    def wait_done(self):
        return self.waiting & (self.wait_count>=math.ceil(
            self.cfg.post_separation_wait_s/self.env.physics_dt-1e-8))

    def other_margin(self):
        active=self.connected()[:,self.other_ids]
        margin=(1.-self.utilization[:,self.other_ids]).amin(-1)
        return torch.where(active.all(-1),margin,margin.new_full(margin.shape,-1.))


def state(env):
    if not hasattr(env,'_disconnect_state'):
        env._disconnect_state=ConnectorState(env)
    return env._disconnect_state


def update_connectors(env):
    env.action_manager.get_term('joint_torque').monitor_torque()
    state(env).update()
    if env._sim_step_counter%env.cfg.decimation==0:
        env.action_manager.get_term('joint_torque').update_jacobian_gain(state(env))
    return state(env).other_margin()


def reset_connectors(env,env_ids):
    state(env).reset(env_ids)


def observations(env):
    """31 channels: q/pi, dq, current peel/80N, actual torque/limit, active, gap."""
    s=state(env)
    q=torch.cat([m.data.joint_pos for m in s.modules],dim=-1)
    velocity=torch.cat([m.data.joint_vel for m in s.modules],dim=-1)
    active=s.connected()
    # Broken welds carry no current load. Never substitute the historical
    # rupture peak; the separate active flags disambiguate unloaded/broken.
    load=torch.where(active,peel_load(s.force,s.moment,s.cfg.connector_radius)/s.cfg.break_force,0.)
    torque=env.action_manager.get_term('joint_torque').torque
    base=torch.cat((q/math.pi,velocity,load,torque/s.cfg.continuous_torque_limit,
                    active.float(),s.geometry()['distance'].unsqueeze(-1)),dim=-1)
    if env.cfg.jacobian_feedforward:
        ff=env.action_manager.get_term('joint_torque').jacobian_command(q,active)
        base=torch.cat((base,ff/env.cfg.torque_limit),-1)
    if env.cfg.separation_velocity_pd:
        ctrl=env.action_manager.get_term('joint_torque')
        base=torch.cat((base,(ctrl.position_target-q)/math.pi),-1)
    return base


def target_loading(env):
    s=state(env)
    # Only bending-induced tensile demand is rewarded. Direct Fz tension,
    # compression, shear, twist and historical rupture load do not contribute.
    load=(s.peak_target_tilt_load/s.cfg.break_force).clamp(0.,s.cfg.loading_reward_cap)
    safety=(s.other_margin()/s.cfg.minimum_other_margin).clamp(0.,1.) if s.cfg.minimum_other_margin>0 else torch.ones_like(load)
    return load*safety*(~s.wrong_break).float()


def separation_reward(env):
    return state(env).separation_progress_delta.clamp_min(0.)/env.step_dt


def separation_regression(env):
    return (-state(env).separation_progress_delta).clamp_min(0.)/env.step_dt


def safety_penalty(env):
    """Nonnegative approach cost; zero outside the safety buffer."""
    s=state(env)
    peak_margin=1.-torch.where(s.other_mask(), s.peak_utilization, 0.).amax(-1)
    start=s.cfg.minimum_other_margin+s.cfg.safety_penalty_buffer
    return ((start-peak_margin)/s.cfg.safety_penalty_buffer).clamp_min(0.).square()


def success_reward(env):
    return success(env).float()/env.step_dt


def failure_cost(env):
    return failure(env).float()/env.step_dt


def success(env):
    s=state(env)
    return s.wait_done() & s.succeeded & ~s.wrong_break & ~s.safety_margin_violation & ~env.action_manager.get_term('joint_torque').torque_failure


def failure(env):
    s=state(env)
    finite=torch.stack([torch.isfinite(m.data.root_link_pose_w).all(-1)
                        & torch.isfinite(m.data.joint_pos).all(-1)
                        & torch.isfinite(m.data.joint_vel).all(-1) for m in s.modules],dim=-1).all(-1)
    position_bad=torch.stack([((m.data.root_link_pos_w[:,:2]-env.scene.env_origins[:,:2]).norm(dim=-1)>env.cfg.workspace_radius)
                              | (m.data.root_link_pos_w[:,2]<-.2)
                              | (m.data.root_link_pos_w[:,2]>2.) for m in s.modules],dim=-1).any(-1)
    return s.wrong_break | s.safety_margin_violation | s.invalid_state | ~finite | position_bad | env.action_manager.get_term('joint_torque').torque_failure


def target_gap(env):
    return state(env).geometry()['distance']


def other_margin(env):
    return state(env).other_margin()


def target_broken(env):
    s=state(env)
    return (~s.connected()[s.worlds,s.target_indices()]).float()


def wrong_break(env):
    return state(env).wrong_break.float()


def connector_load(env,connector_index):
    """Current equivalent opening force in N; disconnected interfaces carry zero."""
    s=state(env)
    load=peel_load(s.force[:,connector_index],s.moment[:,connector_index],s.cfg.connector_radius)
    return torch.where(s.connected()[:,connector_index],load,0.)


def wait_done(env):
    return state(env).wait_done()


def time_out(env):
    # Preserve the four-second active budget, but never cut short an ongoing wait.
    return (env.episode_length_buf>=env.max_episode_length) & ~state(env).waiting


def waiting(env):
    return state(env).waiting.float()


def wait_elapsed(env):
    return state(env).wait_count.float()*env.physics_dt


def safety_margin_violation(env):
    return state(env).safety_margin_violation.float()


def torque_violation(env):
    return env.action_manager.get_term('joint_torque').torque_failure.float()


def torque_overload_time(env):
    return env.action_manager.get_term('joint_torque').overload_count.amax(-1).float()*env.physics_dt


def target_utilization(env):
    s=state(env)
    return s.utilization[s.worlds,s.target_indices()]


def target_tilt_load(env):
    s=state(env);i=s.target_indices()
    return torch.where(s.connected()[s.worlds,i],tilt_load(s.moment[s.worlds,i],s.cfg.connector_radius),0.)


def target_direct_tension(env):
    s=state(env);i=s.target_indices()
    return torch.where(s.connected()[s.worlds,i],s.force[s.worlds,i,2].clamp_min(0.),0.)


def critic_observations(env):
    s=state(env)
    wait_fraction=(s.wait_count.float()*env.physics_dt/s.cfg.post_separation_wait_s).clamp(0.,1.)
    return torch.cat((observations(env),s.waiting.float()[:,None],wait_fraction[:,None]),-1)


def separation_pd_active(env):
    s=state(env)
    return (env.action_manager.get_term('joint_torque').pd_active & ~s.waiting).float()


def separation_command_speed(env):
    return env.action_manager.get_term('joint_torque').velocity_command.abs().amax(-1)


def separation_predicted_gap_speed(env):
    c=env.action_manager.get_term('joint_torque').distance_controller
    return torch.zeros(env.num_envs,device=env.device) if c is None else c.predicted_speed


def target_max_gap(env):
    return state(env).max_target_gap

"""Independent torque task: selected connector opens, the other five stay safe."""
import math
from dataclasses import dataclass
from functools import partial

import torch
from mjlab.actuator import IdealPdActuatorCfg,BuiltinPositionActuatorCfg
from mjlab.actuator.builtin_actuator import BuiltinPositionActuator
from mjlab.entity import EntityCfg,EntityArticulationInfoCfg
from mjlab.envs import ManagerBasedRlEnvCfg,mdp as common_mdp
from mjlab.managers.action_manager import ActionTerm,ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg,ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.spec_config import GeomCfg
from mjlab.viewer import ViewerConfig

from .model import ROOT_NAMES,load_snapshot,module_spec,add_connectors
from . import state as mdp
from .collision import DisconnectSimulationCfg


@dataclass
class DisconnectTorqueEnvCfg(ManagerBasedRlEnvCfg):
    disconnect_torque: bool=True
    enable_weld_failure: bool=True
    jacobian_feedforward: bool=False
    jacobian_seed_tilt_force: float=50.
    jacobian_target_tilt_force: float=84.
    jacobian_other_load_limit: float=60.
    jacobian_axis_xy: tuple[float,float]=(-.7071067811865476,.7071067811865476)
    separation_distance_jacobian: bool=False
    separation_cartesian_speed: float=.02
    separation_distance_gain: float=2.
    separation_goal_buffer: float=.001
    separation_jacobian_damping: float=.005
    separation_joint_acceleration: float=.5
    separation_target_error_limit: float=.05
    separation_velocity_pd: bool=False
    separation_stiffness: float=15.
    separation_damping: float=1.5
    separation_max_target_speed: float=1.
    target_connector: int=5
    break_force: float=80.
    connector_radius: float=.05
    minimum_other_margin: float=.2
    safety_penalty_buffer: float=.2
    separation_distance: float=.05
    post_separation_wait_s: float=2.
    torque_limit: float=20.
    continuous_torque_limit: float=14.
    torque_overload_duration_s: float=1.
    torque_action_scale: float=20.
    loading_reward_cap: float=1.2
    joint_stiffness: float=0.
    joint_damping: float=1.5
    workspace_radius: float=3.
    closure_checkpoint: str=''
    closure_checkpoint_sha256: str=''


@dataclass(kw_only=True)
class TorqueActionCfg(ActionTermCfg):
    def build(self,env):
        return TorqueAction(self,env)


class TorqueAction(ActionTerm):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self._raw=torch.zeros(env.num_envs,6,device=env.device)
        for name in ('torque_action_scale','joint_damping','loading_reward_cap','torque_limit',
                     'continuous_torque_limit','torque_overload_duration_s'):
            if not math.isfinite(getattr(env.cfg,name)) or getattr(env.cfg,name)<=0:
                raise ValueError(f'{name} must be positive and finite')
        if env.cfg.continuous_torque_limit>=env.cfg.torque_limit:
            raise ValueError('continuous_torque_limit must be below torque_limit')
        if env.cfg.joint_stiffness!=0.:
            raise ValueError('Torque feedforward experiment requires joint_stiffness (P)=0')
        self.residual=torch.zeros_like(self._raw)
        self.feedforward=torch.zeros_like(self._raw)
        self.requested_torque=torch.zeros_like(self._raw)
        self.overload_count=torch.zeros_like(self._raw,dtype=torch.long)
        self.torque_failure=torch.zeros(env.num_envs,dtype=torch.bool,device=env.device)
        self.peak_overload_cost=torch.zeros(env.num_envs,device=env.device)
        self.modules=[env.scene[f'module_{i}'] for i in range(6)]
        self.native_pd=isinstance(self.modules[0].actuators[0],BuiltinPositionActuator)
        if self.native_pd:
            env.sim.expand_model_fields(('actuator_gainprm','actuator_biasprm','actuator_forcerange'))
            self.native_ids=[int(env.sim.mj_model.actuator(f'module_{i}/joint{i+1}').id) for i in range(6)]
            for aid in self.native_ids:
                env.sim.model.actuator_forcerange[:,aid,0]=-env.cfg.torque_limit
                env.sim.model.actuator_forcerange[:,aid,1]=env.cfg.torque_limit
        # Card/CLI values are applied after recipe construction. Update the
        # native actuator parameter tensors, also used by its fused group.
        for module in ([] if self.native_pd else self.modules):
            for actuator in module.actuators:
                actuator.stiffness.fill_(0.)
                actuator.damping.fill_(env.cfg.joint_damping)
                actuator.force_limit.fill_(env.cfg.torque_limit)
                actuator.default_stiffness.fill_(0.)
                actuator.default_damping.fill_(env.cfg.joint_damping)
                actuator.default_force_limit.fill_(env.cfg.torque_limit)

        self.pd_active=torch.zeros(env.num_envs,dtype=torch.bool,device=env.device)
        self.position_target=torch.cat([m.data.joint_pos for m in self.modules],-1).clone()
        self.velocity_command=torch.zeros_like(self._raw)
        if env.cfg.separation_velocity_pd:
            for name in ('separation_stiffness','separation_damping','separation_max_target_speed'):
                if not math.isfinite(getattr(env.cfg,name)) or getattr(env.cfg,name)<=0:
                    raise ValueError(f'{name} must be positive and finite')
        self.distance_controller=None
        if env.cfg.separation_distance_jacobian:
            if not env.cfg.separation_velocity_pd:
                raise ValueError('Distance Jacobian control requires velocity-integrated PD')
            for name in ('separation_cartesian_speed','separation_distance_gain','separation_goal_buffer',
                         'separation_jacobian_damping','separation_joint_acceleration','separation_target_error_limit'):
                if not math.isfinite(getattr(env.cfg,name)) or getattr(env.cfg,name)<=0:
                    raise ValueError(f'{name} must be positive and finite')
            from .separation import DistanceJacobianController
            self.distance_controller=DistanceJacobianController(env,self.modules)
        self.jacobian_torque=torch.zeros(6,device=env.device)
        self.jacobian_qref=torch.zeros(6,device=env.device)
        self.jacobian_gain=torch.ones(env.num_envs,device=env.device)
        self.jacobian_report={}
        if env.cfg.jacobian_feedforward:
            for name in ('jacobian_seed_tilt_force','jacobian_target_tilt_force','jacobian_other_load_limit'):
                if not math.isfinite(getattr(env.cfg,name)) or getattr(env.cfg,name)<=0:
                    raise ValueError(f'{name} must be positive and finite')
            if env.cfg.jacobian_other_load_limit>=env.cfg.break_force*(1.-env.cfg.minimum_other_margin):
                raise ValueError('Jacobian reference load limit must leave measured safety headroom')
            from .jacobian import solve_reference
            torque,qref,self.jacobian_report=solve_reference(env.sim.mj_model,env.cfg)
            self.jacobian_torque.copy_(torch.as_tensor(torque,device=env.device,dtype=self._raw.dtype))
            self.jacobian_qref.copy_(torch.as_tensor(qref,device=env.device,dtype=self._raw.dtype))

    def jacobian_command(self,positions=None,active=None):
        if not self._env.cfg.jacobian_feedforward:return torch.zeros_like(self.residual)
        s=mdp.state(self._env)
        active=s.connected() if active is None else active
        positions=torch.cat([m.data.joint_pos for m in self.modules],-1) if positions is None else positions
        valid=(active.all(-1) & ~s.waiting & ~s.safety_margin_violation & ~s.wrong_break
               & ~self.torque_failure & ((positions-self.jacobian_qref).abs().amax(-1)<.05))
        return self.jacobian_gain[:,None]*self.jacobian_torque[None]*valid[:,None]

    def update_jacobian_gain(self,s):
        if not self._env.cfg.jacobian_feedforward:return
        positions=torch.cat([m.data.joint_pos for m in self.modules],-1)
        local=((positions-self.jacobian_qref).abs().amax(-1)<.05)
        active=s.connected().all(-1) & ~s.waiting & ~self.torque_failure & local & ~s.safety_margin_violation
        self.jacobian_gain.copy_(torch.where(local,self.jacobian_gain,1.))
        load=s.peak_target_tilt_load
        error=(self._env.cfg.jacobian_target_tilt_force-load)/self._env.cfg.jacobian_target_tilt_force
        delta=(.5*error).clamp(-.1,.1)
        other=s.peak_utilization[:,s.other_ids].amax(-1)*self._env.cfg.break_force
        delta=torch.where(other>self._env.cfg.jacobian_other_load_limit,
                          -.1*((other-self._env.cfg.jacobian_other_load_limit)/4.).clamp(0.,1.),delta)
        self.jacobian_gain.copy_(torch.where(active,(self.jacobian_gain+delta).clamp(.1,1.5),self.jacobian_gain))

    @property
    def action_dim(self):
        return 6

    @property
    def raw_action(self):
        return self._raw

    def process_actions(self,actions):
        mdp.state(self._env).begin_step()
        self.peak_overload_cost.zero_()
        self._raw.copy_(torch.nan_to_num(actions).clamp(-1.,1.))
        self.velocity_command.copy_(torch.where(self.pd_active[:,None],
            self._raw*self._env.cfg.separation_max_target_speed,0.))
        self.residual.copy_(self._raw*self._env.cfg.torque_action_scale)
        self.feedforward.copy_(self.residual+self.jacobian_command())

    @property
    def torque(self):
        return torch.cat([module.data.qfrc_actuator for module in self.modules],dim=-1)

    def apply_actions(self):
        s=mdp.state(self._env)
        cfg=self._env.cfg
        position=torch.cat([m.data.joint_pos for m in self.modules],-1)
        velocity=torch.cat([m.data.joint_vel for m in self.modules],-1)
        stopped=s.waiting | s.safety_margin_violation | s.wrong_break | self.torque_failure
        if cfg.separation_velocity_pd:
            entered=(~s.connected()[s.worlds,s.target_indices()]) & ~self.pd_active & ~stopped
            # Capture measured position, never reinterpret the loading action as
            # a velocity. The first velocity command comes from the next policy step.
            self.position_target.copy_(torch.where(entered[:,None],position,self.position_target))
            self.velocity_command.masked_fill_(entered[:,None],0.)
            self.pd_active |= entered
            pd=self.pd_active & ~stopped
            if self.distance_controller is not None:
                self.velocity_command.copy_(self.distance_controller.update(s,pd))
                stopped=s.waiting | s.safety_margin_violation | s.wrong_break | self.torque_failure
                pd=self.pd_active & ~stopped
            limits=torch.cat([m.data.joint_pos_limits for m in self.modules],1)
            integrated=(self.position_target+self.velocity_command*self._env.physics_dt).clamp(
                min=limits[...,0],max=limits[...,1])
            if self.distance_controller is not None:
                # Do not accumulate a large position demand against blocked motion.
                bound=cfg.separation_target_error_limit
                integrated=integrated.clamp(min=position-bound,max=position+bound)
            self.position_target.copy_(torch.where(pd[:,None],integrated,position))
        else:
            pd=torch.zeros_like(stopped)
        self._raw.masked_fill_(stopped[:,None],0.)
        self.residual.masked_fill_((stopped|pd)[:,None],0.)
        self.velocity_command.masked_fill_(stopped[:,None],0.)
        self.feedforward.copy_(self.residual+self.jacobian_command())
        self.feedforward.masked_fill_((stopped|pd)[:,None],0.)
        kp=pd[:,None]*cfg.separation_stiffness
        kd=torch.where(pd[:,None],cfg.separation_damping,cfg.joint_damping)
        self.requested_torque.copy_(kp*(self.position_target-position)-kd*velocity+self.feedforward)
        self.check_torque_demand()
        # A newly latched demand failure removes P and feedforward immediately;
        # waiting likewise retains only the original damping, not position hold.
        enabled=(pd & ~self.torque_failure)[:,None]
        kp=enabled*cfg.separation_stiffness
        kd=torch.where(enabled,cfg.separation_damping,cfg.joint_damping)
        self.feedforward.masked_fill_(self.torque_failure[:,None],0.)
        for i,module in enumerate(self.modules):
            if self.native_pd:
                aid=self.native_ids[i]
                gain=torch.where(enabled[:,0],cfg.separation_stiffness,1.)
                gain_parameters=self._env.sim.model.actuator_gainprm
                bias_parameters=self._env.sim.model.actuator_biasprm
                if getattr(self,'capture_native_parameters',False):
                    # SimData wrappers select a stored stream on assignment.
                    # Capture-enabled standalone control must write tensor views
                    # on the active capture stream instead.
                    gain_parameters=gain_parameters._tensor
                    bias_parameters=bias_parameters._tensor
                gain_parameters[:,aid,0]=gain
                bias_parameters[:,aid,1]=-kp[:,0]
                bias_parameters[:,aid,2]=-kd[:,0]
                # BuiltinPositionActuator forwards ctrl. Loading uses gain=1
                # and zero position bias (torque + native damping); separation
                # uses standard MuJoCo position PD with implicit derivatives.
                control=torch.where(enabled,self.position_target[:,i:i+1],self.feedforward[:,i:i+1])
                module.set_joint_position_target(control)
                module.set_joint_velocity_target(torch.zeros_like(control))
                module.set_joint_effort_target(torch.zeros_like(control))
            else:
                for actuator in module.actuators:
                    actuator.stiffness.copy_(kp)
                    actuator.damping.copy_(kd)
                module.set_joint_position_target(self.position_target[:,i:i+1])
                module.set_joint_velocity_target(torch.zeros_like(self.feedforward[:,i:i+1]))
                module.set_joint_effort_target(self.feedforward[:,i:i+1])

    def check_torque_demand(self):
        self.torque_failure |= ((self.requested_torque.abs()>self._env.cfg.torque_limit)
                                | ~torch.isfinite(self.requested_torque)).any(-1)

    def monitor_torque(self):
        """Called once after every physical substep; all joints are independent."""
        self.record_torque(self.torque)

    def record_torque(self,torque):
        cfg=self._env.cfg
        magnitude=torque.abs()
        self.overload_count.copy_(torch.where(magnitude>cfg.continuous_torque_limit,
                                             self.overload_count+1,0))
        duration_steps=math.ceil(cfg.torque_overload_duration_s/self._env.physics_dt-1e-8)
        self.torque_failure |= ((magnitude>cfg.torque_limit) | ~torch.isfinite(torque)
                                | (self.overload_count>=duration_steps)).any(-1)
        finite_magnitude=torch.nan_to_num(magnitude,nan=0.,posinf=2*cfg.torque_limit)
        cost=((finite_magnitude-cfg.continuous_torque_limit).clamp_min(0.)
              /(cfg.torque_limit-cfg.continuous_torque_limit)).square().mean(-1)
        self.peak_overload_cost.copy_(torch.maximum(self.peak_overload_cost,cost))

    def reset(self,env_ids=None):
        ids=slice(None) if env_ids is None else env_ids
        self._raw[ids]=0.
        self.feedforward[ids]=0.
        self.residual[ids]=0.
        self.jacobian_gain[ids]=1.
        if self.distance_controller is not None:
            self.distance_controller.reset(ids)

        self.pd_active[ids]=False
        self.velocity_command[ids]=0.
        self.position_target[ids]=torch.cat([m.data.default_joint_pos for m in self.modules],-1)[ids]
        if self.native_pd:
            for aid in self.native_ids:
                self._env.sim.model.actuator_gainprm[ids,aid,0]=1.
                self._env.sim.model.actuator_biasprm[ids,aid,:]=0.
        for module in ([] if self.native_pd else self.modules):
            for actuator in module.actuators:
                actuator.stiffness[ids]=0.
                actuator.damping[ids]=self._env.cfg.joint_damping
        self.requested_torque[ids]=0.
        self.overload_count[ids]=0
        self.torque_failure[ids]=False
        self.peak_overload_cost[ids]=0.


def torque_cost(env):
    action=env.action_manager.get_term('joint_torque')
    return action.peak_overload_cost


def disconnect_env_cfg(play=False,native_pd=False):
    actuator_cls=BuiltinPositionActuatorCfg if native_pd else IdealPdActuatorCfg
    snapshot=load_snapshot()
    modules={}
    for i,module in enumerate(snapshot['modules']):
        root=module['root_state']
        modules[f'module_{i}']=EntityCfg(
            spec_fn=partial(module_spec,i),
            articulation=EntityArticulationInfoCfg(
                actuators=(actuator_cls(target_names_expr=(f'joint{i+1}',),
                                                  stiffness=1. if native_pd else 0.,damping=0. if native_pd else 1.5,
                                                  effort_limit=20.,armature=.001),),
                soft_joint_pos_limit_factor=1.),
            init_state=EntityCfg.InitialStateCfg(pos=tuple(root[:3]),rot=tuple(root[3:7]),
                lin_vel=tuple(root[7:10]),ang_vel=tuple(root[10:13]),
                joint_pos={f'joint{i+1}':module['joint_position']},
                joint_vel={f'joint{i+1}':module['joint_velocity']}))
    cfg=DisconnectTorqueEnvCfg(
        seed=42,decimation=4,episode_length_s=4.,
        closure_checkpoint=snapshot['source_checkpoint'],
        closure_checkpoint_sha256=snapshot['source_sha256'],
        scene=SceneCfg(num_envs=1 if play else 1024,env_spacing=4.,entities=modules,
            terrain=TerrainEntityCfg(terrain_type='plane',
                geoms=(GeomCfg(geom_names_expr=('terrain$',),friction=(.6,.005,.0001)),)),
            spec_fn=add_connectors),
        sim=DisconnectSimulationCfg(mujoco=MujocoCfg(timestep=.005,integrator='implicitfast',
            solver='newton',jacobian='dense',iterations=50,ls_iterations=20,ccd_iterations=50),njmax=512,nconmax=128),
        commands={},actions={'joint_torque':TorqueActionCfg(entity_name='module_0')},
        observations={name:ObservationGroupCfg({'disconnect':ObservationTermCfg(func=mdp.observations if name=='actor' else mdp.critic_observations)},
            concatenate_terms=True,enable_corruption=False) for name in ('actor','critic')},
        events={'reset_scene':EventTermCfg(func=common_mdp.reset_scene_to_default,mode='reset'),
                'reset_connectors':EventTermCfg(func=mdp.reset_connectors,mode='reset')},
        rewards={'target_loading':RewardTermCfg(func=mdp.target_loading,weight=1.),
                 'target_separation':RewardTermCfg(func=mdp.separation_reward,weight=5.),
                 'penalty_separation_regression':RewardTermCfg(func=mdp.separation_regression,weight=-5.),
                 'success':RewardTermCfg(func=mdp.success_reward,weight=10.),
                 'penalty_safety_margin':RewardTermCfg(func=mdp.safety_penalty,weight=-.2),
                 'penalty_failure':RewardTermCfg(func=mdp.failure_cost,weight=-10.),
                 'penalty_torque':RewardTermCfg(func=torque_cost,weight=-.01)},
        terminations={'failure':TerminationTermCfg(func=mdp.failure),
                      'time_out':TerminationTermCfg(func=mdp.time_out,time_out=True),
                      'separation_wait_done':TerminationTermCfg(func=mdp.wait_done)},
        metrics={'physics_connector_update':MetricsTermCfg(func=mdp.update_connectors,per_substep=True),
                 'target_gap_m':MetricsTermCfg(func=mdp.target_gap),
                 'target_max_gap_m':MetricsTermCfg(func=mdp.target_max_gap,reduce='max'),
                 'separation_pd_active':MetricsTermCfg(func=mdp.separation_pd_active,reduce='max'),
                 'separation_command_speed_rad_s':MetricsTermCfg(func=mdp.separation_command_speed,reduce='max'),
                 'separation_predicted_gap_speed_m_s':MetricsTermCfg(func=mdp.separation_predicted_gap_speed),
                 'target_broken':MetricsTermCfg(func=mdp.target_broken,reduce='max'),
                 'wrong_break':MetricsTermCfg(func=mdp.wrong_break,reduce='max'),
                 'success':MetricsTermCfg(func=mdp.success,reduce='max'),
                 'waiting':MetricsTermCfg(func=mdp.waiting),
                 'wait_elapsed_s':MetricsTermCfg(func=mdp.wait_elapsed),
                 'other_min_margin':MetricsTermCfg(func=mdp.other_margin),
                 'torque_violation':MetricsTermCfg(func=mdp.torque_violation,reduce='max'),
                 'max_torque_overload_s':MetricsTermCfg(func=mdp.torque_overload_time,reduce='max'),
                 'safety_margin_violation':MetricsTermCfg(func=mdp.safety_margin_violation,reduce='max'),
                 'target_load_utilization':MetricsTermCfg(func=mdp.target_utilization),
                 'target_tilt_load_N':MetricsTermCfg(func=mdp.target_tilt_load),
                 'target_direct_tension_N':MetricsTermCfg(func=mdp.target_direct_tension)},
        viewer=ViewerConfig(entity_name='module_0',body_name=ROOT_NAMES[0],distance=1.2,
            lookat=(-.25,0.,.08),elevation=-25.,azimuth=135.,origin_type=ViewerConfig.OriginType.WORLD))
    for i in range(6):
        cfg.metrics[f'connector_{i}_load_N']=MetricsTermCfg(
            func=mdp.connector_load,params={'connector_index':i})
    return cfg

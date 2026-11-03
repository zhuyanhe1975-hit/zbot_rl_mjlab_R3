"""Experimental MJLab full-trajectory batch environment; not a validated task."""
import torch
import numpy as np
from mjlab.envs import mdp as common_mdp
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg,ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from .mixed_ppo import motion_ppo_env_cfg
from .disconnect.state import state,update_connectors
from .full_batched_action import FullBatchedActionCfg
from .full_batched_fsm import P
from .full_batched_scene import configure_straight_initial_scene,add_full_batched_connectors_headtail_90

def observation(env):
    s=state(env);f=s.fsm;a=env.action_manager.get_term('joint_torque')
    q,v=a.joint_state();g=s.geometry()
    phase=torch.nn.functional.one_hot(f.phase,len(P)).float();target=torch.nn.functional.one_hot(f.target,6).float()
    result=torch.cat((torch.sin(q),torch.cos(q),v/5.,(a.position_target-q)/3.2,s.utilization,s.connected().float(),
        env.sim.data.eq_active[:,s.variant_ids].float(),phase,target,(f.phase_time/10.)[:,None],(f.phase_elapsed/10.)[:,None],f.stage[:,None]/4.,
        g['distance'][:,None]/.5,g['angle'][:,None]/torch.pi,g['dv'].norm(dim=-1,keepdim=True)/2.,
        g['dw'].norm(dim=-1,keepdim=True)/5.,a.torque/20.,a.policy,
        a.teacher_velocity/5.,(a.motion_velocity-v)/5.,a.policy_torque/20.),-1)
    if getattr(env.cfg,'full_policy_curl_only',False):
        result=torch.cat((result,a.curl_residual_offset/float(env.cfg.full_curl_offset_limit)),-1)
    return result

def failed(env):
    s=state(env);margin=s.safety_margin_violation if getattr(env.cfg,'full_enforce_other_margin',False) else torch.zeros_like(s.wrong_break)
    return (s.fsm.phase==P['FAILED'])|s.wrong_break|s.invalid_state|s.contact_violation|margin|env.action_manager.get_term('joint_torque').torque_failure

def success(env):
    s=state(env);result=(s.fsm.phase==P['DONE'])&~failed(env)
    if getattr(env.cfg,'full_terminal_recheck',False):
        q,v=env.action_manager.get_term('joint_torque').joint_state()
        result&=s.terminal_ready(q,v,float(getattr(env.cfg,'full_final_angle_tolerance',.02)))
    return result

def phase_timeout(env):return state(env).fsm.failure_code==5

def time_penalty(env):return torch.ones(env.num_envs,device=env.device)

def safety_penalty(env):
    s=state(env);peak=torch.where(s.other_mask(),s.peak_utilization*env.cfg.break_force,0.).amax(-1)
    return ((peak-60.)/20.).clamp_min(0.).square()

def tracking_penalty(env):
    s=state(env);f=s.fsm;a=env.action_manager.get_term('joint_torque');ref=a.reference.sample(f.phase,f.target,f.stage,f.phase_time)
    _,v=a.joint_state()
    moving=(f.phase==P['CURL'])|(f.phase==P['CAPTURE_HEADTAIL'])|(f.phase==P['SEPARATE'])|(f.phase==P['ROTATE'])|(f.phase==P['APPROACH'])|(f.phase==P['UNFOLD'])
    return ((ref['qvel']-v).square().mean(-1))*ref['valid']*moving

def progress_reward(env):
    f=state(env).fsm;a=env.action_manager.get_term('joint_torque')
    return (f.stage.float()*4.+f.phase.float()*.01-a.progress_before).clamp_min(0.)/env.step_dt

def terminal_success(env):return success(env).float()/env.step_dt

def terminal_failure(env):return failed(env).float()/env.step_dt

def make_full_batched_env_cfg(num_envs=4,reference=None,velocity_scale=.5,torque_scale=2.):
    cfg=configure_straight_initial_scene(motion_ppo_env_cfg(play=False));cfg.scene.num_envs=num_envs
    refined=False
    if reference is not None:
        with np.load(reference,allow_pickle=False) as teacher:
            refined='reference_version' in teacher and str(teacher['reference_version'])=='refined_forward_90_overlap_v1'
    if refined:cfg.scene.spec_fn=add_full_batched_connectors_headtail_90
    cfg.full_refined_reference=refined
    cfg.minimum_other_margin=0.;cfg.sim.mujoco.timestep=.001;cfg.decimation=20;cfg.episode_length_s=120.
    cfg.full_outer_stiffness=30.;cfg.full_outer_damping=2.;cfg.full_inner_stiffness=30.;cfg.full_inner_damping=2.
    if reference is not None:cfg.full_teacher_reference=str(reference)
    cfg.full_velocity_scale=float(velocity_scale);cfg.full_torque_scale=float(torque_scale)
    cfg.full_curl_symmetric=not refined
    cfg.full_cuda_graphs=True
    cfg.actions={'joint_torque':FullBatchedActionCfg(entity_name='module_0')}
    cfg.observations={key:ObservationGroupCfg({'full':ObservationTermCfg(func=observation)},concatenate_terms=True,enable_corruption=False) for key in ('actor','critic')}
    cfg.metrics={'physics_connector_update':MetricsTermCfg(func=update_connectors,per_substep=True)}
    cfg.rewards={'progress':RewardTermCfg(func=progress_reward,weight=1.),'success':RewardTermCfg(func=terminal_success,weight=10.),
       'penalty_failure':RewardTermCfg(func=terminal_failure,weight=-10.),'penalty_time':RewardTermCfg(func=time_penalty,weight=-.08),
       'penalty_safety_margin':RewardTermCfg(func=safety_penalty,weight=-.5),'penalty_tracking':RewardTermCfg(func=tracking_penalty,weight=-.2)}
    cfg.terminations={'failure':TerminationTermCfg(func=failed),'phase_timeout':TerminationTermCfg(func=phase_timeout),'success':TerminationTermCfg(func=success),
                      'time_out':TerminationTermCfg(func=common_mdp.time_out,time_out=True)}
    return cfg


def make_refined_teacher_cfg(num_envs,reference,reference_rate=.2):
    """Fixed-initial-state diagnostic configuration, with no policy residual."""
    if num_envs<1 or not 0.<reference_rate<=1.:
        raise ValueError('invalid batch size or reference rate')
    cfg=make_full_batched_env_cfg(num_envs,reference,0.,0.)
    if not cfg.full_refined_reference:
        raise ValueError('requires the refined forward reference')
    cfg.auto_reset=False;cfg.break_force=100.;cfg.minimum_other_margin=.2;cfg.full_enforce_other_margin=True
    cfg.full_outer_stiffness=80.;cfg.full_outer_damping=4.;cfg.full_inner_stiffness=400.;cfg.full_inner_damping=4.
    cfg.full_reference_rate=reference_rate;cfg.full_teacher_load_replay=True;cfg.full_teacher_load_extend=True
    cfg.full_teacher_headtail_replay=True;cfg.full_teacher_separation_position=True;cfg.full_final_angle_tolerance=.04
    cfg.full_terminal_recheck=True
    return cfg

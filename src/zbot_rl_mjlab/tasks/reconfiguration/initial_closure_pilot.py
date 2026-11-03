"""First head-tail closure as an independent, reference-guided RL episode."""
import torch
from mjlab.envs import mdp as common_mdp
from mjlab.managers.observation_manager import ObservationGroupCfg,ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.utils.lab_api.math import quat_apply,quat_conjugate,quat_mul
from .disconnect.state import state
from .full_batched_env import failed,progress_reward,time_penalty
from .full_batched_fsm import P
from .state_machine import rotvec


def initial_closure_observation(env):
    """Compact task state: joints, target pose, load, phase, controller memory."""
    s=state(env);f=s.fsm;a=env.action_manager.get_term('joint_torque')
    q,v=a.joint_state();pose=s.capture_geometry()
    inverse=quat_conjugate(pose['q1'])
    local_delta=quat_apply(inverse,pose['delta'])/.5
    orientation=rotvec(quat_mul(inverse,pose['q0']))/torch.pi
    phases=torch.stack((f.phase==P['INITIAL_STRAIGHT_HOLD'],f.phase==P['CURL'],
                        f.phase==P['CAPTURE_HEADTAIL'],f.phase==P['SETTLE_HEADTAIL']),-1).float()
    compact=torch.cat((q/torch.pi,v/8.,local_delta,orientation,s.utilization[:,:5],
                      phases,(f.phase_time/5.)[:,None],
                      a.curl_residual_offset/float(env.cfg.full_curl_offset_limit),a.policy),-1)
    # Predict approach/overshoot from relative motion, not just current pose.
    local_velocity=quat_apply(inverse,pose['dv'])/.5
    local_angular_velocity=quat_apply(inverse,pose['dw'])/5.
    elapsed=(env.episode_length_buf*env.step_dt/env.max_episode_length_s)[:,None]
    values=(compact,local_velocity,local_angular_velocity,elapsed)
    if getattr(env.cfg,'full_online_headtail',False):
        distances=env.sim.data.sensordata[:,a.online_planner.distance_ids]
        values=(*values,distances.clamp(-.15,.15)/.15)
    return torch.cat(values,-1)


def initial_closure_success(env):
    """The verified FSM has captured and held the head-tail weld for 0.25s."""
    s=state(env);f=s.fsm
    return ((f.phase==P['LOAD'])&(f.stage==0)&(f.target==0)&s.connected().all(-1)
            &~failed(env)&~common_mdp.time_out(env))


def initial_closure_reward(env):
    return initial_closure_success(env).float()/env.step_dt


def headtail_pose_nearness(env):
    """Bounded dense reward for approaching the actual 90-degree capture pose."""
    s=state(env);f=s.fsm
    pose=s.capture_geometry()
    active=((f.phase==P['CURL'])|(f.phase==P['CAPTURE_HEADTAIL']))&(f.target==5)
    active &= ~failed(env)&~common_mdp.time_out(env)
    return torch.exp(-pose['gap']/.15-pose['angle']/.75)*active


def headtail_pose_progress(env):
    """Reward improvement in capture-pose potential, not merely waiting near it."""
    s=state(env);a=env.action_manager.get_term('joint_torque')
    pose=s.capture_geometry()
    current=torch.exp(-pose['gap']/.15-pose['angle']/.75)
    return (current-a.pose_potential_before)*a.pose_progress_active/env.step_dt


def closure_progress(env):
    """A failure phase must not be mistaken for forward progress."""
    value=progress_reward(env)
    return torch.where(failed(env),torch.zeros_like(value),value)


def closure_failure_penalty(env):
    """Penalize physical failure and the ordinary episode timeout."""
    return (failed(env)|common_mdp.time_out(env)).float()/env.step_dt


def closure_time_reward(env):
    """Positive inverse completion time, paid once only on safe completion."""
    elapsed=(env.episode_length_buf*env.step_dt).clamp_min(env.step_dt)
    return initial_closure_success(env).float()*env.max_episode_length_s/elapsed/env.step_dt


def configure_initial_closure(cfg):
    """End immediately after initial closure, before any release/loading."""
    cfg.full_stop_after_initial_closure=True
    cfg.full_initial_closure_collision_pruning=True
    cfg.full_headtail_penetration_guard=True
    cfg.full_headtail_alignment_position_tolerance=.001
    cfg.full_headtail_alignment_angle_tolerance=2.*torch.pi/180.
    cfg.episode_length_s=20.
    # The deadline is a task failure, not a truncation to bootstrap beyond.
    cfg.is_finite_horizon=True
    cfg.observations={name:ObservationGroupCfg({'closure':ObservationTermCfg(func=initial_closure_observation)},
                      concatenate_terms=True,enable_corruption=False) for name in ('actor','critic')}
    cfg.full_initial_closure_diagnostics=True
    cfg.terminations['success']=TerminationTermCfg(func=initial_closure_success)
    # Completion speed is the main reward; retain weak pose-nearness guidance.
    # Failure cannot collect the terminal time reward.
    cfg.rewards={
        'penalty_failure':RewardTermCfg(func=closure_failure_penalty,weight=-20.),
        'time':RewardTermCfg(func=closure_time_reward,weight=1.),
        'pose_nearness':RewardTermCfg(func=headtail_pose_nearness,weight=.1),
        # Larger than the maximum .1/s guidance income: waiting never pays.
        'penalty_elapsed_time':RewardTermCfg(func=time_penalty,weight=-.2),
    }
    cfg.full_pose_progress_reward=False
    return cfg

"""学生可见的归直阶段仅依赖可部署量，不替代策略动作。"""
from types import SimpleNamespace

import torch
from test_universal_task import state_env

from zbot_rl_mjlab.tasks.universal import deployment, mdp


def make_env(monkeypatch):
    env,_=state_env()
    env.scene['robot'].data.joint_pos_biased=env.scene['robot'].data.joint_pos.clone()
    cfg=deployment.DeploymentCommandCfg(resampling_time_range=(5.,8.),training=False)
    term=cfg.build(env)
    env.command_manager=SimpleNamespace(get_term=lambda name:term)
    monkeypatch.setattr(mdp.standing,'base_gravity_alignment_cost',lambda env:torch.ones(2))
    return env,term


def test_stage_is_visible_without_adding_dimensions_or_changing_user_request(monkeypatch):
    env,term=make_env(monkeypatch)
    q=env.scene['robot'].data.joint_pos_biased;q[:]=.15
    env.action_manager.get_term('joint_pos').commanded_target[:]=.15
    term._clear(torch.arange(2));term._update_straightening(torch.arange(2))
    assert term.straightening_mask.all()
    assert term.command[:,0].eq(1).all()
    assert term.observation_mode.eq(0).all()
    assert deployment.command(env).shape==(2,4)
    assert mdp.critic_state(env).shape==(2,53)
    assert mdp.critic_state(env)[:,38].eq(0).all()
    assert not mdp.configuration_ready(env).any()


def test_release_waits_for_encoder_and_target_and_does_not_rearm(monkeypatch):
    env,term=make_env(monkeypatch)
    q=env.scene['robot'].data.joint_pos_biased;q[:]=.15
    target=env.action_manager.get_term('joint_pos').commanded_target;target[:]=.15
    term._clear(torch.arange(2));term._update_straightening(torch.arange(2))
    q[0]=0
    term._update_straightening(torch.arange(2));assert term.straightening_mask.all()
    target[0]=0
    term._update_straightening(torch.arange(2))
    assert term.straightening_mask.tolist()==[False,True]
    q[0]=.2;target[0]=.2
    term._update_straightening(torch.arange(2))
    assert not term.straightening_mask[0]
    term.set_target('snake');term._update_straightening(torch.arange(2))
    assert not term.straightening_mask.any() and term.observation_mode.eq(-1).all()
    term.set_target('stand');term._update_straightening(torch.arange(2))
    assert term.straightening_mask.all()


def test_initial_standing_robot_is_not_flattened(monkeypatch):
    env,term=make_env(monkeypatch)
    monkeypatch.setattr(mdp.standing,'base_gravity_alignment_cost',lambda env:torch.zeros(2))
    env.scene['robot'].data.joint_pos_biased[:]=torch.tensor(env.cfg.standing_joint_pos)
    term._clear(torch.arange(2));term._update_straightening(torch.arange(2))
    assert not term.straightening_mask.any()


def test_height_contact_phase_and_true_velocity_do_not_change_software_stage(monkeypatch):
    env,term=make_env(monkeypatch)
    env.scene['robot'].data.joint_pos_biased[:]=.15
    env.action_manager.get_term('joint_pos').commanded_target[:]=.15
    term._clear(torch.arange(2));term._update_straightening(torch.arange(2))
    before=deployment.command(env).clone()
    data=env.scene['robot'].data;data.body_link_pos_w[...,2]+=100;data.body_link_lin_vel_w[:]=99
    env.scene['feet_ground_contact'].data.found[:]=0
    term.phase[:]=0;term.transition_mask[:]=False
    term._update_straightening(torch.arange(2))
    torch.testing.assert_close(deployment.command(env),before)


def test_straightening_cost_has_no_reward_cliff_and_preserves_actions(monkeypatch):
    env,term=make_env(monkeypatch)
    env.scene['robot'].data.joint_pos_biased[:]=.15
    action=env.action_manager.get_term('joint_pos');action.commanded_target[:]=.15
    old=action.commanded_target.clone()
    term._clear(torch.arange(2));term._update_straightening(torch.arange(2))
    assert (deployment.straightening_cost(env)>0).all()
    torch.testing.assert_close(action.commanded_target,old)
    term.straightening_mask[:]=False
    assert deployment.straightening_cost(env).eq(0).all()

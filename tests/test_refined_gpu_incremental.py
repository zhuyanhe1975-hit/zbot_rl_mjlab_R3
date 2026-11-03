import numpy as np
import pytest
import torch

from scripts.train_refined_gpu_incremental import ActorCritic, make_pilot_cfg, make_initial_closure_cfg, policy_action, episode_outcomes
from zbot_rl_mjlab.tasks.reconfiguration.full_batched_fsm import P


def test_pilot_starts_from_zero_and_masks_non_curl():
    net=ActorCritic(12)
    obs=torch.randn(3,12)
    assert torch.equal(net.actor(obs),torch.zeros(3,6))
    raw=torch.ones(3,6)
    actions=policy_action(raw,torch.tensor([P['CURL'],P['LOAD'],P['UNFOLD']]))
    assert (actions[0,:6]>0).all()
    assert (actions[1:]==0).all()
    assert (actions[:,6]==0).all()
    assert torch.allclose(net.distribution(obs).stddev,torch.full((3,6),.1))
    extra=policy_action(torch.ones(3,7),torch.tensor([P['CURL'],P['LOAD'],P['UNFOLD']]))
    assert extra[0,6]>0 and (extra[1:]==0).all()
    controls=policy_action(torch.ones(3,7),torch.tensor([P['CURL']]*3),torch.tensor([True,False,True]))
    assert (controls[[0,2]]==0).all() and (controls[1]>0).all()


def test_pilot_reference_configuration():
    cfg=make_pilot_cfg(4096)
    assert cfg.scene.num_envs==4096
    assert cfg.sim.mujoco.timestep==.001 and cfg.decimation==20
    assert cfg.full_reference_rate==.2 and cfg.break_force==100.
    assert cfg.minimum_other_margin==.2 and cfg.full_enforce_other_margin
    assert cfg.full_velocity_scale==.02 and cfg.full_torque_scale==0.
    assert cfg.full_curl_offset_limit==.02 and cfg.full_policy_curl_only
    assert 'penalty_tracking' not in cfg.rewards


def test_aggressive_closure_configuration_and_episode_outcomes():
    cfg=make_initial_closure_cfg(4096)
    assert cfg.sim.mujoco.timestep==.002 and cfg.decimation==10
    assert cfg.full_curl_clock_fraction==.5 and cfg.full_velocity_scale==.2
    assert cfg.full_curl_offset_limit==.15 and cfg.full_enforce_other_margin
    done=torch.tensor([True,True,True,False]);success=torch.tensor([True,False,True,False]);baseline=torch.tensor([True,True,False,False])
    failed,ref_ok,policy_ok,ref_fail,policy_fail=episode_outcomes(done,success,baseline)
    assert failed.tolist()==[False,True,False,False]
    assert ref_ok.tolist()==[True,False,False,False]
    assert policy_ok.tolist()==[False,False,True,False]
    assert ref_fail.tolist()==[False,True,False,False] and not policy_fail.any()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_curl_offset_saturates_and_phase_clock_ignores_policy():
    from mjlab.envs import ManagerBasedRlEnv
    cfg=make_pilot_cfg(2);cfg.auto_reset=False
    env=ManagerBasedRlEnv(cfg,device='cuda:0')
    try:
        with torch.inference_mode():
            env.reset();s=env._disconnect_state;a=env.action_manager.get_term('joint_torque')
            s.fsm.phase.fill_(P['CURL']);s.fsm.phase_time.fill_(.1)
            a.process_actions(torch.ones((2,7),device=env.device))
            torch.testing.assert_close(s.fsm.phase_time,torch.full_like(s.fsm.phase_time,.104),atol=1e-6,rtol=0.)
            assert float(a.curl_residual_velocity.abs().max())<=.02
            a.curl_residual_offset.fill_(.01999);a.curl_residual_velocity.fill_(.02);a.motion_velocity.fill_(.02)
            a._apply_actions_eager();a._apply_actions_eager()
            assert float(a.curl_residual_offset.max())<=.020001
            s.fsm.phase.fill_(P['LOAD_HEADTAIL'])
            a.policy.fill_(1.);a._apply_actions_eager()
            assert (a.policy_torque==0).all()
    finally:env.close()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_bounded_tempo_only_changes_curl_phase_clock():
    from mjlab.envs import ManagerBasedRlEnv
    cfg=make_pilot_cfg(1);cfg.auto_reset=False;cfg.full_curl_clock_fraction=.05
    env=ManagerBasedRlEnv(cfg,device='cuda:0')
    try:
        with torch.inference_mode():
            env.reset();s=env._disconnect_state;a=env.action_manager.get_term('joint_torque')
            s.fsm.phase.fill_(P['CURL']);s.fsm.phase_time.fill_(.1)
            a.process_actions(torch.ones((1,7),device=env.device))
            torch.testing.assert_close(s.fsm.phase_time,torch.full_like(s.fsm.phase_time,.10402),atol=1e-6,rtol=0.)
            cfg.full_curl_clock_fraction=.5;s.fsm.phase_time.fill_(.1);a.policy.zero_()
            a.process_actions(torch.ones((1,7),device=env.device))
            torch.testing.assert_close(s.fsm.phase_time,torch.full_like(s.fsm.phase_time,.1042),atol=1e-6,rtol=0.)
            s.fsm.phase.fill_(P['CAPTURE_HEADTAIL']);s.fsm.phase_time.zero_()
            a.process_actions(torch.ones((1,7),device=env.device))
            torch.testing.assert_close(s.fsm.phase_time,torch.full_like(s.fsm.phase_time,.02),atol=1e-6,rtol=0.)
    finally:env.close()

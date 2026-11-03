import pytest
import torch

@pytest.fixture
def env():
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    from mjlab.envs import ManagerBasedRlEnv
    from zbot_rl_mjlab.tasks.reconfiguration.full_batched_env import make_full_batched_env_cfg
    with torch.inference_mode():
        e=ManagerBasedRlEnv(make_full_batched_env_cfg(2),device='cuda:0');e.reset()
        try:yield e
        finally:e.close()

def test_motion_integrates_joint_velocity_and_no_torque_residual(env):
    from zbot_rl_mjlab.tasks.reconfiguration.full_batched_fsm import P
    a=env.action_manager.get_term('joint_torque');s=env._disconnect_state
    s.fsm.phase[:]=P['CURL'];a.position_target.zero_();a.motion_velocity.fill_(.4);a.policy[:,:6]=1.
    a.apply_actions()
    torch.testing.assert_close(a.position_target,torch.full_like(a.position_target,.0004),atol=1e-7,rtol=0.)
    assert not a.policy_torque.any()

def test_closed_loading_policy_adds_real_torque_residual(env):
    from zbot_rl_mjlab.tasks.reconfiguration.full_batched_fsm import P
    a=env.action_manager.get_term('joint_torque');s=env._disconnect_state
    s.fsm.phase[:]=P['LOAD'];s.fsm.target[:]=0
    env.sim.data.eq_active[:,s.eq_ids[5]]=True
    a.policy[:,:6]=.5;a.loading.torque.fill_(5.);a.apply_actions()
    torch.testing.assert_close(a.policy_torque,torch.ones_like(a.policy_torque))
    assert torch.all((a.requested_torque-6.).abs()<.1)

def test_motion_uses_velocity_not_teacher_position(env):
    from zbot_rl_mjlab.tasks.reconfiguration.full_batched_fsm import P
    a=env.action_manager.get_term('joint_torque');s=env._disconnect_state
    s.fsm.phase[:]=P['CURL'];s.fsm.phase_time[:]=.3
    a.process_actions(torch.zeros((2,7),device=env.device));before=a.motion_velocity.clone()
    s.fsm.phase[:]=P['CURL'];s.fsm.phase_time[:]=.3
    a.reference.q.fill_(999.);a.reference.qref.fill_(-999.)
    a.process_actions(torch.zeros((2,7),device=env.device))
    torch.testing.assert_close(a.motion_velocity,before)

def test_jacobian_pose_feedback_is_active_during_approach_reference(env,monkeypatch):
    from zbot_rl_mjlab.tasks.reconfiguration.full_batched_fsm import P
    a=env.action_manager.get_term('joint_torque');s=env._disconnect_state
    s.fsm.phase[:]=P['APPROACH'];s.fsm.target[:]=0;s.fsm.phase_time[:]=.1
    env.sim.data.eq_active[:,s.eq_ids[5]]=True;env.sim.data.eq_active[:,s.eq_ids[0]]=False
    monkeypatch.setattr(a,'approach_correction',lambda *args:torch.full_like(a.motion_velocity,.02))
    a.process_actions(torch.zeros((2,7),device=env.device))
    assert (a.teacher_velocity.abs().sum(-1)>0).all()
    torch.testing.assert_close(a.motion_velocity-a.teacher_velocity,torch.full_like(a.motion_velocity,.0004),atol=1e-6,rtol=0.)

def test_timeout_resets_only_stalled_gpu_world(env):
    from zbot_rl_mjlab.tasks.reconfiguration.full_batched_fsm import P
    s=env._disconnect_state;f=s.fsm
    f.phase[0]=P['FINAL_STRAIGHT_HOLD'];f.phase_elapsed[0]=2.99
    _,_,terminated,_,_=env.step(torch.zeros((2,7),device=env.device))
    assert terminated.tolist()==[True,False]
    assert env.termination_manager.get_term('phase_timeout').tolist()==[True,False]
    assert f.phase.tolist()==[P['INITIAL_STRAIGHT_HOLD'],P['INITIAL_STRAIGHT_HOLD']]
    assert f.phase_elapsed[0].item()==0. and f.phase_elapsed[1].item()>0.

def test_shared_joint_buffers_match_entity_views(env):
    from zbot_rl_mjlab.tasks.reconfiguration.disconnect.env_cfg import TorqueAction
    a=env.action_manager.get_term('joint_torque')
    generator=torch.Generator(device=env.device).manual_seed(6)
    env.sim.data.qvel.copy_(torch.randn(env.sim.data.qvel.shape,device=env.device,generator=generator)*.003)
    env.sim.forward()
    q,v=a.joint_state()
    torch.testing.assert_close(q,torch.cat([m.data.joint_pos for m in a.modules],-1))
    torch.testing.assert_close(v,torch.cat([m.data.joint_vel for m in a.modules],-1))
    torch.testing.assert_close(a.torque,TorqueAction.torque.fget(a))

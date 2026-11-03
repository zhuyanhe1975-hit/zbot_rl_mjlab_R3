import pytest
import torch
from zbot_rl_mjlab.tasks.reconfiguration.full_batched_fsm import P

def test_gpu_full_state_contact_phase_and_selective_reset():
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    from mjlab.envs import ManagerBasedRlEnv
    from zbot_rl_mjlab.tasks.reconfiguration.mixed_ppo import motion_ppo_env_cfg
    from zbot_rl_mjlab.tasks.reconfiguration.full_batched_scene import configure_straight_initial_scene
    from zbot_rl_mjlab.tasks.reconfiguration.full_batched_state import FullBatchedState
    cfg=configure_straight_initial_scene(motion_ppo_env_cfg(play=True));cfg.scene.num_envs=4
    with torch.inference_mode():
        env=ManagerBasedRlEnv(cfg,device='cuda:0');env.reset()
        try:
            s=FullBatchedState(env);env._disconnect_state=s;s.reset(None)
            s.fsm.phase[:]=torch.tensor([P['ROTATE'],P['UNFOLD'],P['CURL'],P['ROTATE']],device=env.device)
            s.fsm.target[:]=torch.tensor([1,5,5,4],device=env.device)
            s.set_contact_phase()
            assert s.phase_contacts.nonzero().tolist()==[[0,1],[1,5],[3,4]]
            s.reset(torch.tensor([1,3],device=env.device))
            assert s.fsm.phase.tolist()==[P['ROTATE'],P['INITIAL_STRAIGHT_HOLD'],P['CURL'],P['INITIAL_STRAIGHT_HOLD']]
            assert s.target_indices().tolist()==[1,5,5,5]
            assert s.connected()[[1,3]].tolist()==[[True,True,True,True,True,False]]*2
        finally:env.close()

def test_capture_geometry_uses_actual_registered_site():
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    from mjlab.envs import ManagerBasedRlEnv
    from zbot_rl_mjlab.tasks.reconfiguration.mixed_ppo import motion_ppo_env_cfg
    from zbot_rl_mjlab.tasks.reconfiguration.full_batched_scene import configure_straight_initial_scene
    from zbot_rl_mjlab.tasks.reconfiguration.full_batched_state import FullBatchedState
    cfg=configure_straight_initial_scene(motion_ppo_env_cfg(play=True));cfg.scene.num_envs=2
    with torch.inference_mode():
        env=ManagerBasedRlEnv(cfg,device='cuda:0');env.reset()
        try:
            s=FullBatchedState(env);s.fsm.target[:]=torch.tensor([0,5],device=env.device)
            actual=s.capture_geometry()['gap']
            v=s.modules[0].data.site_pos_w[0,s.capture_sites[0]]
            a=s.modules[1].data.site_pos_w[0,s.in_sites[1]]
            b=s.modules[5].data.site_pos_w[1,s.out_sites[5]]
            c=s.modules[0].data.site_pos_w[1,s.in_sites[0]]
            torch.testing.assert_close(actual,torch.stack(((a-v).norm(),(c-b).norm())))
        finally:env.close()

def test_fast_port_frames_and_target_gap_match_entity_geometry():
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    from mjlab.envs import ManagerBasedRlEnv
    from zbot_rl_mjlab.tasks.reconfiguration.full_batched_env import make_full_batched_env_cfg
    from zbot_rl_mjlab.tasks.reconfiguration.disconnect.state import ConnectorState
    with torch.inference_mode():
        env=ManagerBasedRlEnv(make_full_batched_env_cfg(4),device='cuda:0');env.reset()
        try:
            s=env._disconnect_state
            for _ in range(5):env.step(torch.zeros((4,7),device=env.device))
            s.fsm.target[:]=torch.tensor([0,1,4,5],device=env.device)
            pos,quat=s.outgoing_port_frames();base_pos,base_quat=ConnectorState.outgoing_port_frames(s)
            torch.testing.assert_close(pos,base_pos,atol=1e-6,rtol=1e-6)
            # q and -q encode the same rotation.
            assert ((quat*base_quat).sum(-1).abs()>1.-1e-5).all()
            expected=s.geometry()['distance'];s.max_target_gap.zero_();s.update_separation(s.connected())
            torch.testing.assert_close(s.max_target_gap,expected,atol=1e-6,rtol=1e-6)
        finally:env.close()

def test_fast_target_port_geometry_matches_original_with_motion():
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    from mjlab.envs import ManagerBasedRlEnv
    from zbot_rl_mjlab.tasks.reconfiguration.full_batched_env import make_full_batched_env_cfg
    from zbot_rl_mjlab.tasks.reconfiguration.batched_state import BatchedState
    with torch.inference_mode():
        env=ManagerBasedRlEnv(make_full_batched_env_cfg(4),device='cuda:0');env.reset()
        try:
            generator=torch.Generator(device=env.device).manual_seed(123)
            env.sim.data.qvel.copy_(torch.randn(env.sim.data.qvel.shape,device=env.device,generator=generator)*.002)
            env.sim.forward();s=env._disconnect_state;s.fsm.target[:]=torch.tensor([0,1,4,5],device=env.device)
            fast=s.geometry();baseline=BatchedState.geometry(s)
            for key in ('delta','distance','angle','dv','dw'):
                torch.testing.assert_close(fast[key],baseline[key],atol=2e-6,rtol=2e-5)
            for key in ('q0','q1','relative_quat'):
                assert ((fast[key]*baseline[key]).sum(-1).abs()>1.-1e-5).all()
        finally:env.close()

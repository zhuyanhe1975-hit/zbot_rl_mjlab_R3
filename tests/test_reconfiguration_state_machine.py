"""Real deterministic cycle, timing and registered native-weld reconstruction."""
import pytest
import torch


def test_native_state_machine_completes_three_motion_phases_within_one_second():
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.actuator.builtin_actuator import BuiltinPositionActuator
    from zbot_rl_mjlab.tasks.reconfiguration.state_machine import state_machine_env_cfg,Phase
    from zbot_rl_mjlab.tasks.reconfiguration.disconnect.state import state
    cfg=state_machine_env_cfg()
    assert not cfg.rewards and not cfg.auto_reset
    env=ManagerBasedRlEnv(cfg,device='cuda:0')
    try:
        with torch.inference_mode():
            env.reset();s=state(env);a=env.action_manager.get_term('joint_torque');c=a.distance_controller
            assert all(isinstance(m.actuators[0],BuiltinPositionActuator) for m in a.modules)
            assert env.sim.mj_model.neq==7
            assert not env.sim.data.eq_active[:,s.new_eq].any()
            seen={Phase.LOAD};peak=0.
            for _ in range(400):
                # The external action is ignored by this standalone controller.
                _,_,done,timeout,_=env.step(torch.ones(1,6,device=env.device))
                phase=Phase(c.phase.item());seen.add(phase)
                assert phase!=Phase.FAILED
                assert not timeout.any()
                assert s.connected()[:,:5].all()
                assert (s.peak_utilization[:,:5]<=.8+1e-6).all()
                assert not a.torque_failure.any()
                peak=max(peak,s.max_target_gap.item())
                if done.any():break
            assert Phase(c.phase.item())==Phase.DONE
            assert seen=={Phase.LOAD,Phase.SEPARATE,Phase.ROTATE,Phase.APPROACH,Phase.SETTLE,Phase.DONE}
            assert peak>.05
            for phase in (Phase.SEPARATE,Phase.ROTATE,Phase.APPROACH):
                assert 0<c.phase_durations[0,phase]<=1.+1e-5
            assert c.phase_durations[0,Phase.SETTLE]>=2.
            assert env.sim.data.eq_active[:,s.new_eq].all()
            assert not env.sim.data.eq_active[:,s.eq_ids[5]].any()
            assert s.connected().all()
            assert torch.allclose(env.sim.model.eq_solref[:,s.new_eq,0],torch.full((1,),.005,device=env.device))
            assert s.geometry()['distance'].item()<cfg.capture_distance
            assert c.angle_error.item()<cfg.capture_angle
            # Independent reset restores old registration and disables new weld.
            env.reset()
            assert s.connected().all() and not env.sim.data.eq_active[:,s.new_eq].any()
            assert Phase(c.phase.item())==Phase.LOAD
    finally:env.close()



def test_state_machine_deadline_stops_without_forcing_reconnection():
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    from mjlab.envs import ManagerBasedRlEnv
    from zbot_rl_mjlab.tasks.reconfiguration.state_machine import state_machine_env_cfg,Phase
    from zbot_rl_mjlab.tasks.reconfiguration.disconnect.state import state
    cfg=state_machine_env_cfg();cfg.motion_duration_s=.05
    env=ManagerBasedRlEnv(cfg,device='cuda:0')
    try:
        with torch.inference_mode():
            env.reset();s=state(env);a=env.action_manager.get_term('joint_torque');c=a.distance_controller
            for _ in range(30):
                _,_,done,_,_=env.step(torch.zeros(1,6,device=env.device))
                if done.any():break
            assert c.phase.item()==Phase.FAILED and c.failure_code.item()==3
            assert not env.sim.data.eq_active[:,s.new_eq].any()
            assert not a.velocity_command.any() and not a.feedforward.any()
            assert s.connected()[:,:5].all()
            for aid in a.native_ids:assert not env.sim.model.actuator_biasprm[:,aid,1].any()
    finally:env.close()

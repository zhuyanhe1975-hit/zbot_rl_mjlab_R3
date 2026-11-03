"""GPU graph replay must follow target changes, mode switches, and real resets."""
import pytest
import torch
from mjlab.envs import ManagerBasedRlEnv
from zbot_rl_mjlab.tasks.reconfiguration.reference_dynamic import reference_env_cfg,ReferencePhase
from zbot_rl_mjlab.tasks.reconfiguration.batched_state import BatchedState


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA graph execution')
def test_fast_reference_matches_eager_and_replays_native_mode_switches():
    with torch.inference_mode():
        env=ManagerBasedRlEnv(reference_env_cfg(),device='cuda:0')
        try:
            env.reset();action=torch.zeros(1,6,device=env.device)
            env.step(action);env.step(action)
            s=env._disconnect_state;a=env.action_manager.get_term('joint_torque');c=a.distance_controller
            generator=torch.Generator(device=env.device).manual_seed(31)
            env.sim.data.qvel.copy_(torch.randn(env.sim.data.qvel.shape,device=env.device,generator=generator)*.002)
            env.sim.forward()
            for i in range(6):
                fast=s.geometry(i);eager=BatchedState.geometry(s,i)
                for key in fast:torch.testing.assert_close(fast[key],eager[key],rtol=2e-5,atol=2e-6)
            for stage in range(4):
                s.index.fill_(stage);s.measurement_graph.replay()
                saved={key:getattr(s,key).clone() for key in ('force','moment','utilization','max_other_load','max_module_depth','max_ground_depth')}
                s.last_substep=None;BatchedState.update(s)
                for key,value in saved.items():torch.testing.assert_close(getattr(s,key),value,rtol=2e-5,atol=2e-6)
            s.index.zero_();s.waiting.zero_();s.safety_margin_violation.zero_();s.wrong_break.zero_();a.torque_failure.zero_()
            ids=torch.tensor(a.native_ids,device=env.device)
            c.phase.fill_(ReferencePhase.LOAD);a.pd_active.zero_();a.jacobian_torque.fill_(1.)
            a.actuation_graph.replay()
            torch.testing.assert_close(env.sim.model.actuator_gainprm[0,ids,0],torch.ones(6,device=env.device))
            torch.testing.assert_close(env.sim.model.actuator_biasprm[0,ids,1],torch.zeros(6,device=env.device))
            c.phase.fill_(ReferencePhase.STABILIZE);a.pd_active.fill_(True);a.actuation_graph.replay()
            torch.testing.assert_close(env.sim.model.actuator_gainprm[0,ids,0],torch.full((6,),400.,device=env.device))
            env.reset();env.step(action)
            assert not s.safety_margin_violation.any() and s.connected().all()
        finally:env.close()

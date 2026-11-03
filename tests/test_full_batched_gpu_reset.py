import pytest
import torch
import warp as wp
from mjlab.envs import ManagerBasedRlEnv
from zbot_rl_mjlab.tasks.reconfiguration.full_batched_env import make_full_batched_env_cfg
from zbot_rl_mjlab.tasks.reconfiguration.full_batched_gpu_reset import FullBatchedGpuReset

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_fixed_mask_reset_matches_selective_mjlab_reset():
    with torch.inference_mode():
        fast=ManagerBasedRlEnv(make_full_batched_env_cfg(4),device='cuda:0')
        reference=ManagerBasedRlEnv(make_full_batched_env_cfg(4),device='cuda:0')
        previous=torch.cuda.current_stream(fast.device);shared=torch.cuda.ExternalStream(wp.get_stream(fast.sim.wp_device).cuda_stream,device=fast.device)
        try:
            torch.cuda.set_stream(shared);fast.reset();reference.reset();reset=FullBatchedGpuReset(fast)
            generator=torch.Generator(device=fast.device).manual_seed(16)
            dq=torch.randn(fast.sim.data.qvel.shape,device=fast.device,generator=generator)*.01
            for e in (fast,reference):
                e.sim.data.qvel.copy_(dq);e._disconnect_state.fsm.phase.fill_(1)
                a=e.action_manager.get_term('joint_torque');a.policy.fill_(.1);a.motion_velocity.fill_(.03)
                e.episode_length_buf.fill_(5);e.sim.forward()
            mask=torch.tensor([False,True,False,True],device=fast.device)
            reset.reset(mask);fast.scene.write_data_to_sim();fast.sim.forward()
            reference.reset(env_ids=torch.tensor([1,3],device=reference.device));torch.cuda.synchronize()
            for field in ('qpos','qvel','eq_active'):
                torch.testing.assert_close(getattr(fast.sim.data,field)._tensor,getattr(reference.sim.data,field)._tensor,atol=2e-6,rtol=2e-6)
            for field in ('phase','target','stage','phase_time','phase_elapsed'):
                torch.testing.assert_close(getattr(fast._disconnect_state.fsm,field),getattr(reference._disconnect_state.fsm,field))
            for field in ('policy','position_target','motion_velocity','torque_failure'):
                torch.testing.assert_close(getattr(fast.action_manager.get_term('joint_torque'),field),getattr(reference.action_manager.get_term('joint_torque'),field))
            torch.testing.assert_close(fast.episode_length_buf,reference.episode_length_buf)
            actions=torch.zeros((4,7),device=fast.device)
            for _ in range(6):fast.step(actions);reference.step(actions)
            torch.cuda.synchronize()
            torch.testing.assert_close(fast.sim.data.qpos._tensor,reference.sim.data.qpos._tensor,atol=3e-5,rtol=3e-5)
            torch.testing.assert_close(fast.sim.data.qvel._tensor,reference.sim.data.qvel._tensor,atol=3e-4,rtol=3e-4)
            assert torch.equal(fast.sim.data.eq_active._tensor,reference.sim.data.eq_active._tensor)
        finally:
            torch.cuda.set_stream(previous);fast.close();reference.close()

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_gpu_mask_environment_auto_reset_matches_mjlab_steps():
    from zbot_rl_mjlab.tasks.reconfiguration.full_batched_gpu_env import FullBatchedGpuEnv
    from zbot_rl_mjlab.tasks.reconfiguration.full_batched_fsm import P
    with torch.inference_mode():
        fast=FullBatchedGpuEnv(make_full_batched_env_cfg(4),device='cuda:0')
        ref=ManagerBasedRlEnv(make_full_batched_env_cfg(4),device='cuda:0')
        previous=torch.cuda.current_stream(fast.device);shared=torch.cuda.ExternalStream(wp.get_stream(fast.sim.wp_device).cuda_stream,device=fast.device)
        try:
            torch.cuda.set_stream(shared);fast.reset();ref.reset();action=torch.zeros((4,7),device=fast.device)
            # Force a timeout in only one world and verify automatic reset and
            # subsequent independent trajectories match the native MJLab path.
            for e in (fast,ref):
                s=e._disconnect_state;s.fsm.phase[1]=P['FINAL_STRAIGHT_HOLD'];s.fsm.phase_elapsed[1]=2.99
                s.contact_violation[3]=True
                e.action_manager.get_term('joint_torque').torque_failure[0]=True
            for _ in range(6):
                left=fast.step(action);right=ref.step(action);torch.cuda.synchronize()
                torch.testing.assert_close(left[1],right[1],atol=2e-5,rtol=2e-5)
                assert torch.equal(left[2],right[2]) and torch.equal(left[3],right[3])
                if _==0:assert left[2].tolist()==[True,True,False,True]
                torch.testing.assert_close(left[0]['actor'],right[0]['actor'],atol=3e-3,rtol=3e-3)
                torch.testing.assert_close(fast.sim.data.qpos._tensor,ref.sim.data.qpos._tensor,atol=2e-5,rtol=2e-5)
                torch.testing.assert_close(fast.sim.data.qvel._tensor,ref.sim.data.qvel._tensor,atol=2e-4,rtol=2e-4)
                assert torch.equal(fast.sim.data.eq_active._tensor,ref.sim.data.eq_active._tensor)
        finally:
            torch.cuda.set_stream(previous);fast.close();ref.close()

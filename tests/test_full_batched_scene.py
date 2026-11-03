import json
import numpy as np
from zbot_rl_mjlab.tasks.reconfiguration.full_batched_scene import INITIAL,configure_straight_initial_scene
from zbot_rl_mjlab.tasks.reconfiguration.mixed_ppo import motion_ppo_env_cfg

def test_scene_config_uses_recorded_straight_pose_and_open_reset():
    data=json.loads(INITIAL.read_text());cfg=configure_straight_initial_scene(motion_ppo_env_cfg())
    for i in range(6):
        init=cfg.scene.entities[f'module_{i}'].init_state
        np.testing.assert_allclose(init.pos,data['roots'][i][:3]);np.testing.assert_allclose(init.rot,data['roots'][i][3:])
        assert init.joint_pos[f'joint{i+1}']==0.
    assert 'reset_headtail_open' in cfg.events

def test_four_gpu_worlds_reset_with_headtail_open():
    import pytest
    import torch
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    from mjlab.envs import ManagerBasedRlEnv
    from zbot_rl_mjlab.tasks.reconfiguration.disconnect.state import state
    cfg=configure_straight_initial_scene(motion_ppo_env_cfg(play=True));cfg.scene.num_envs=4
    with torch.inference_mode():
        env=ManagerBasedRlEnv(cfg,device='cuda:0');env.reset()
        try:
            s=state(env);q=torch.cat([m.data.joint_pos for m in s.modules],-1)
            assert torch.allclose(q,torch.zeros_like(q))
            assert s.connected().tolist()==[[True,True,True,True,True,False]]*4
            assert env.sim.mj_model.pair('open_face_5').id>=0
            env.sim.data.eq_active[0,s.eq_ids[5]]=True
            env.reset(env_ids=torch.tensor([1,3],device=env.device))
            assert s.connected()[0,5] and not s.connected()[1,5] and not s.connected()[3,5]
        finally:env.close()


def test_asynchronously_finished_world_can_reset_while_others_continue():
    import pytest
    import torch
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    from mjlab.envs import ManagerBasedRlEnv
    from zbot_rl_mjlab.tasks.reconfiguration.full_batched_env import make_full_batched_env_cfg
    from zbot_rl_mjlab.tasks.reconfiguration.full_batched_fsm import P
    cfg=make_full_batched_env_cfg(2);cfg.auto_reset=False
    env=ManagerBasedRlEnv(cfg,device='cuda:0')
    try:
        env.reset();s=env._disconnect_state;s.fsm.phase[0]=P['DONE']
        action=torch.zeros((2,7),device=env.device)
        _,_,terminated,truncated,_=env.step(action)
        assert (terminated|truncated).tolist()==[True,False]
        assert env.termination_manager.get_term('success').tolist()==[True,False]
        env.reset(env_ids=torch.tensor([0],device=env.device))
        env.step(action)
        assert s.fsm.phase[1]!=P['FAILED']
    finally:env.close()

"""Independent mixed-clocked walking geometry and inherited recipe."""
import mujoco
import numpy as np
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

from zbot_rl_mjlab.tasks.snake.mixed_clocking import mixed_clocking_spec

TASK = 'Mjlab-Zbot-6dof-Walking-Mixed-270-270-180-90-90'


def test_inherits_walking_recipe_and_has_independent_identity():
    for play in (False, True):
        cfg = load_env_cfg(TASK, play=play)
        old = load_env_cfg('Mjlab-Zbot-6dof-Walking', play=play)
        assert cfg.scene.entities['robot'].spec_fn is mixed_clocking_spec
        assert {k: v.weight for k, v in cfg.rewards.items()} == {k: v.weight for k, v in old.rewards.items()}
        assert tuple(cfg.observations['actor'].terms) == tuple(old.observations['actor'].terms)
        assert cfg.actions['joint_pos'].scale == old.actions['joint_pos'].scale
        assert cfg.scene.entities['robot'].init_state.joint_pos != old.scene.entities['robot'].init_state.joint_pos
    assert load_rl_cfg(TASK).experiment_name == 'zbot_walking_mixed_270_270_180_90_90'


def test_standing_pose_feet_grounded_and_body_clear():
    cfg = load_env_cfg(TASK)
    init = cfg.scene.entities['robot'].init_state
    q = np.array([init.joint_pos[f'joint{i+1}'] for i in range(6)])
    np.testing.assert_allclose(q[:3], -q[:2:-1], atol=1e-10)
    m = mixed_clocking_spec().compile(); d = mujoco.MjData(m)
    d.qpos[:3] = init.pos; d.qpos[3:7] = init.rot
    for i in range(6): d.qpos[m.joint(f'joint{i+1}').qposadr] = init.joint_pos[f'joint{i+1}']
    mujoco.mj_forward(m, d)
    feet = [m.site(n).id for n in ('foot_0_sole', 'foot_1_sole')]
    np.testing.assert_allclose(d.site_xpos[feet[1]] - d.site_xpos[feet[0]], [0., .25, 0.], atol=.001)
    assert np.max(np.abs(d.site_xpos[feet, 2])) < .001
    assert abs(d.site_xmat[feet[1]].reshape(3,3)[2,2]) > .999
    base = m.body('base').id
    assert abs(d.xpos[base,1]) < .001
    base_x = d.xmat[base].reshape(3,3)[:,0]
    np.testing.assert_allclose(base_x[:2] / np.linalg.norm(base_x[:2]), [1., 0.], atol=.001)
    assert d.xpos[base,2] > .23
    assert not any(c.dist < -.001 for c in d.contact)
    for g in range(m.ngeom):
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH: continue
        mesh = m.geom_dataid[g]; start = m.mesh_vertadr[mesh]; count = m.mesh_vertnum[mesh]
        verts = m.mesh_vert[start:start+count] @ d.geom_xmat[g].reshape(3,3).T + d.geom_xpos[g]
        assert verts[:,2].min() > -.001

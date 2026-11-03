"""Opt-in connector shell clipping leaves original module inertials intact."""
import mujoco
import numpy as np
import pytest
from scipy.spatial import ConvexHull

from zbot_mjbatch.shell_collision import ShellCut, add_clipped_collision_shells
from zbot_rl_mjlab.tasks.reconfiguration.disconnect.model import module_spec


def mesh_vertices_in_body(model, geom_name):
    g = model.geom(geom_name).id
    mesh = model.geom_dataid[g]
    start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, model.geom_quat[g])
    return model.mesh_vert[start:start + count] @ rotation.reshape(3, 3).T + model.geom_pos[g]


@pytest.mark.parametrize('index', [0, 5])
def test_real_module_cut_preserves_body_and_collision_shell(index):
    spec = module_spec(index)
    before = spec.compile()
    a = 'a1_geom' if index == 0 else 'a6_geom'
    b = 'b1_geom' if index == 0 else 'b6_geom'
    cuts = [ShellCut(a, 'port_in', +1), ShellCut(b, 'port_out', -1)]
    result = add_clipped_collision_shells(spec, cuts)
    after = spec.compile()
    np.testing.assert_allclose(after.body_mass, before.body_mass, atol=0., rtol=0.)
    np.testing.assert_allclose(after.body_inertia, before.body_inertia, atol=0., rtol=0.)
    np.testing.assert_allclose(after.qpos0, before.qpos0, atol=0., rtol=0.)
    assert after.ngeom == before.ngeom + 2
    for cut, record in zip(cuts, result):
        old_id = after.geom(cut.geom).id
        shell_id = after.geom(record['collision_geom']).id
        assert after.geom_contype[old_id] == after.geom_conaffinity[old_id] == 0
        assert after.geom_contype[shell_id] == before.geom_contype[before.geom(cut.geom).id]
        assert after.geom_conaffinity[shell_id] == before.geom_conaffinity[before.geom(cut.geom).id]
        assert after.geom_bodyid[shell_id] == after.geom_bodyid[old_id]
        sid = after.site(cut.site).id
        rotation = np.empty(9); mujoco.mju_quat2Mat(rotation, after.site_quat[sid])
        normal = cut.direction * rotation.reshape(3, 3)[:, 2]
        origin = after.site_pos[sid]
        shell_vertices = mesh_vertices_in_body(after, record['collision_geom'])
        signed = (shell_vertices - origin) @ normal
        assert signed.min() >= cut.depth_m - 1e-7
        assert signed.max() > .05
        original_vertices = mesh_vertices_in_body(before, cut.geom)
        retained = original_vertices[(original_vertices - origin) @ normal > cut.depth_m + 1e-7]
        assert len(retained) > 0
        hull = ConvexHull(shell_vertices)
        inside = retained @ hull.equations[:, :3].T + hull.equations[:, 3]
        assert inside.max() < 1e-7


def test_invalid_cut_rejected_before_changing_geom():
    spec = module_spec(0)
    with pytest.raises(ValueError, match='direction'):
        add_clipped_collision_shells(spec, [ShellCut('a1_geom', 'port_in', 0)])
    assert spec.geom('a1_geom').contype != 0


def test_clipped_slab_is_open_but_back_shell_still_blocks_probe():
    spec = module_spec(0)
    base = spec.compile()
    data = mujoco.MjData(base); mujoco.mj_forward(base, data)
    site = base.site('port_in').id
    origin = data.site_xpos[site].copy()
    axis = data.site_xmat[site].reshape(3, 3)[:, 2].copy()
    for name, distance in [('cut_probe', .001), ('shell_probe', .006)]:
        spec.worldbody.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_SPHERE,
                                size=(.0002, 0., 0.), pos=origin + axis * distance,
                                contype=0, conaffinity=0, mass=0.)
    add_clipped_collision_shells(spec, [ShellCut('a1_geom', 'port_in', +1)])
    model = spec.compile(); data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    shell = model.geom('a1_geom_clipped_shell').id
    gap = lambda probe: mujoco.mj_geomDistance(model, data, shell, model.geom(probe).id, .1, None)
    assert gap('cut_probe') > .0005
    assert gap('shell_probe') <= 0.

import mujoco
import numpy as np

from zbot_rl_mjlab.robot import ASSET_PATH, DEFAULT_JOINT_POS, JOINT_NAMES


def test_sites_are_at_matching_sole_centers():
    """当前运行模型的双足站点须与实际碰撞网格的底面中心一致。"""
    geom_names = ("a1_geom", "b6_geom")
    model = mujoco.MjModel.from_xml_path(str(ASSET_PATH))
    data = mujoco.MjData(model)
    for name, position in zip(JOINT_NAMES, DEFAULT_JOINT_POS):
        data.qpos[model.jnt_qposadr[model.joint(name).id]] = position
    mujoco.mj_forward(model, data)
    sites = []
    for foot, geom_name in zip(("foot_0", "foot_1"), geom_names):
        geom_id = model.geom(geom_name).id
        mesh_id = model.geom_dataid[geom_id]
        start = model.mesh_vertadr[mesh_id]
        count = model.mesh_vertnum[mesh_id]
        vertices = model.mesh_vert[start : start + count]
        world = vertices @ data.geom_xmat[geom_id].reshape(3, 3).T
        world += data.geom_xpos[geom_id]
        sole = world[world[:, 2] < world[:, 2].min() + 0.0005]
        site = data.site_xpos[model.site(foot + "_sole").id]
        # 确认该点位于脚底平面上，并靠近其水平方向的中心。
        assert abs(site[2] - sole[:, 2].mean()) < 0.0001
        center = (sole[:, :2].min(axis=0) + sole[:, :2].max(axis=0)) / 2
        np.testing.assert_allclose(site[:2], center, atol=0.001)
        sites.append(site.copy())
    assert abs(sites[0][2] - sites[1][2]) < 0.0001

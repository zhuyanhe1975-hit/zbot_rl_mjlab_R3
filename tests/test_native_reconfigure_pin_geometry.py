"""New reconstruction peg geometry and its actual four-pin pose gate."""
import math
import inspect

import mujoco
import numpy as np
import pytest

from zbot_mjbatch.full_square_connector import FullSquareConnector
from zbot_mjbatch.native_batch import make_native_batch_model
from zbot_mjbatch.native_safety import NativeSafetyAudit
from zbot_mjbatch.physical_zbot_geometry import add_physical_zbot_geometry


def _vertices_in_frame(model, data, geom, site):
    mesh=int(model.geom_dataid[geom]); start=int(model.mesh_vertadr[mesh])
    count=int(model.mesh_vertnum[mesh])
    world=(model.mesh_vert[start:start+count] @ data.geom_xmat[geom].reshape(3,3).T
           + data.geom_xpos[geom])
    return (world-data.site_xpos[site]) @ data.site_xmat[site].reshape(3,3)


def test_all_twenty_four_compiled_taper_pins_have_the_requested_dimensions():
    model,data=make_native_batch_model()
    tip_radius=.002-.004*math.tan(math.radians(15.))
    for face in range(6):
        site=model.site(f'module_{face}/physical_out_frame').id
        for pin in range(4):
            geom=model.geom(f'physical_connector_{face}_pin_{pin}').id
            local=_vertices_in_frame(model,data,geom,site)
            z=local[:,2]; top=z.max(); bottom=z.min()
            root=local[np.isclose(z,top,atol=1e-7,rtol=0),:2]
            tip=local[np.isclose(z,bottom,atol=1e-7,rtol=0),:2]
            root_radius=np.linalg.norm(root-root.mean(axis=0),axis=1).max()
            tip_r=np.linalg.norm(tip-tip.mean(axis=0),axis=1).max()
            assert top-bottom == pytest.approx(.004,abs=1e-7)
            assert root_radius == pytest.approx(.002,abs=1e-7)
            assert tip_r == pytest.approx(tip_radius,abs=1e-7)
            assert math.degrees(math.atan((root_radius-tip_r)/(top-bottom))) == pytest.approx(15.,abs=.01)
    audit=NativeSafetyAudit(model,data)
    assert audit.pin_height == pytest.approx(.004,abs=1e-7)
    assert audit.pin_tip_radius == pytest.approx(tip_radius,abs=1e-7)
    with pytest.raises(ValueError,match='do not match'):
        NativeSafetyAudit(model,data,pin_height_m=.002)


def test_socket_remains_two_mm_thick_and_original_prototype_defaults_unchanged():
    model,data=make_native_batch_model()
    for face in range(6):
        site=model.site(f'module_{(face+1)%6}/physical_in_frame').id
        geom=model.geom(f'physical_connector_{face}_socket_piece_0').id
        z=_vertices_in_frame(model,data,geom,site)[:,2]
        assert np.ptp(z) == pytest.approx(.002,abs=1e-7)
    assert FullSquareConnector().pin_height_m == .002
    assert inspect.signature(add_physical_zbot_geometry).parameters['pin_tip_radius_m'].default == .0018


@pytest.mark.parametrize('quarter',range(4))
def test_four_pin_pose_gate_recognizes_all_quarter_turns(quarter):
    model,data=make_native_batch_model()
    audit=NativeSafetyAudit(model,data)
    src,_,offset=audit.frames[0]
    rotation=data.site_xmat[src].reshape(3,3)
    pivot=data.site_xpos[src]+rotation@offset
    axis=rotation[:,2].copy()
    adr=int(model.joint('module_0/module_root').qposadr[0])
    turn=np.empty(4); mat=np.empty(9); quat=np.empty(4)
    mujoco.mju_axisAngle2Quat(turn,axis,quarter*math.pi/2)
    mujoco.mju_quat2Mat(mat,turn)
    data.qpos[adr:adr+3]=pivot+mat.reshape(3,3)@(data.qpos[adr:adr+3]-pivot)
    mujoco.mju_mulQuat(quat,turn,data.qpos[adr+3:adr+7].copy())
    data.qpos[adr+3:adr+7]=quat
    mujoco.mj_forward(model,data)
    audit.sample()
    gap,angle=audit.mating_pose(0)
    assert gap < 1e-8 and angle < 1e-6
    assert audit.four_pin_gate_excess(0) <= 0.

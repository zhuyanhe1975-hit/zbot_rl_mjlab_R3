"""Tests for the isolated, uncalibrated cone-mesh pin prototype."""
import math
import mujoco
import numpy as np
import pytest

from scripts.compare_taper_connector import clearance
from zbot_mjbatch.taper_pin_prototype import tapered_pin_model, tapered_pin_spec


def _body_vertices(model, geom):
    mesh=int(model.geom_dataid[geom]); address=model.mesh_vertadr[mesh]; n=model.mesh_vertnum[mesh]
    rotation=np.empty(9);mujoco.mju_quat2Mat(rotation,model.geom_quat[geom])
    return model.mesh_vert[address:address+n] @ rotation.reshape(3,3).T + model.geom_pos[geom]


@pytest.mark.parametrize('tip', [.0018, .001, 0.])
def test_compiled_shared_mesh_has_requested_base_tip_and_height(tip):
    model=tapered_pin_model(tip_radius_m=tip)
    ids=[model.geom(f'pin_{i}').id for i in range(4)]
    assert len(set(model.geom_dataid[ids].tolist()))==1
    geom=ids[0]; vertices=_body_vertices(model,geom)
    bottom=vertices[np.isclose(vertices[:,2],-.002,atol=1e-8)]
    top=vertices[np.isclose(vertices[:,2],0.,atol=1e-8)]
    radii_bottom=np.linalg.norm(bottom[:,:2]-[.035,0.],axis=1)
    radii_top=np.linalg.norm(top[:,:2]-[.035,0.],axis=1)
    np.testing.assert_allclose(radii_bottom,tip,atol=2e-9)
    np.testing.assert_allclose(radii_top,.002,atol=2e-9)
    assert model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH


@pytest.mark.parametrize('quarter', range(4))
def test_aligned_four_pins_do_not_contact_socket(quarter):
    model=tapered_pin_model(); data=mujoco.MjData(model)
    q=model.jnt_qposadr[model.body_jntadr[model.body('pin_plate').id]]
    angle=quarter*math.pi/2
    data.qpos[q+2]=.0005
    data.qpos[q+3:q+7]=[math.cos(angle/2),0.,0.,math.sin(angle/2)]
    mujoco.mj_forward(model,data)
    assert data.ncon == 0
    # Beware: this does not certify positive closest distance; on 3.11.0 some
    # convex mesh pair queries return a false zero, checked separately below.


def test_lateral_overtravel_produces_pin_contact():
    model=tapered_pin_model(); data=mujoco.MjData(model)
    joint=model.body_jntadr[model.body('pin_plate').id]
    q=model.jnt_qposadr[joint]; dof=model.jnt_dofadr[joint]
    data.qpos[q]=.0003; data.qpos[q+2]=.0005
    mujoco.mj_forward(model,data)
    assert data.ncon>0 and data.qfrc_constraint[dof]<0
    mujoco.mj_step(model,data,nstep=10)
    assert not np.any(data.warning.number) and np.isfinite(data.qpos).all()


def test_detect_inconsistent_nearest_point_results():
    model=tapered_pin_model()
    result=clearance(model,.0005)
    # The idealized coaxial geometry has nonzero clearance. A false zero from
    # mj_geomDistance must never be silently presented as validated geometry.
    for value in result:
        if value['raw_min_mm'] < .1:
            assert value['unreliable']


def test_xml_roundtrip(tmp_path):
    spec=tapered_pin_spec();spec.compile();path=tmp_path/'taper.xml';path.write_text(spec.to_xml())
    model=mujoco.MjModel.from_xml_path(str(path))
    assert len({int(model.geom_dataid[model.geom(f'pin_{i}').id]) for i in range(4)})==1

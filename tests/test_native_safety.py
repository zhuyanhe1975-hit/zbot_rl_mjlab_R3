"""Regression tests for the native contact audit, not hardware validation."""
import math

import mujoco
import numpy as np
import pytest

from zbot_mjbatch.native_batch import make_native_batch_model
from zbot_mjbatch.native_safety import NativeSafetyAudit, classify_contact


@pytest.fixture(scope='module')
def model():
    model,_ = make_native_batch_model()
    model.geom_solref[:, 0] = .004
    return model


def data_and_audit(model):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return data, NativeSafetyAudit(model, data)


def test_classification_uses_pair_and_interface():
    assert classify_contact('physical_connector_0_adhesion_source_0', 'physical_connector_0_adhesion_target_3') == ('magnetic_proxy',0)
    assert classify_contact('physical_connector_0_adhesion_source_0', 'physical_connector_1_adhesion_target_0') == ('other',None)
    assert classify_contact('physical_connector_0_pin_plate_disc','physical_connector_0_socket_piece_3') == ('mating_face',0)
    assert classify_contact('physical_connector_0_pin_2','physical_connector_0_socket_piece_3') == ('pin_wall',0)
    assert classify_contact('physical_connector_0_pin_2','module_1/a2_geom_clipped_shell') == ('pin_backing',0)
    assert classify_contact('physical_connector_0_pin_2','module_2/a3_geom_clipped_shell') == ('other',None)
    assert classify_contact('terrain','module_0/a1_geom_clipped_shell') == ('ground',None)


def test_initial_all_six_poses_and_proxy_contact(model):
    data,audit=data_and_audit(model)
    assert not audit.sample()
    for face in range(6):
        gap, angle = audit.mating_pose(face)
        assert gap < 1e-8 and angle < 1e-6
    report=audit.report()
    assert report['contact_counts']['magnetic_proxy'] == 24
    assert report['samples'] == 1
    assert report['max_penetration_m'].get('pin_backing',0.) < 1e-8


def test_post_step_configuration_recomputed_independently(model):
    data,audit=data_and_audit(model)
    assert not audit.sample()
    # Change integrated pose without calling mj_forward: data.contact remains at
    # the original state, while the independent audit records the new overlap.
    old_contacts = data.ncon
    a=model.joint('module_0/module_root').qposadr[0]
    src,_,_=audit.frames[0]
    rotation=data.site_xmat[src].reshape(3,3).copy()
    data.qpos[a:a+3] += rotation[:,0]*.00025 - rotation[:,2]*.0005
    data.time=.001
    issues=audit.sample()
    assert data.ncon == old_contacts
    assert not any(x['kind'] in ('collision','ground_penetration') for x in issues)
    assert audit.max_penetration['magnetic_proxy'] > .0001
    assert audit.worst_contact['magnetic_proxy']['time_s'] == .001


def test_actual_joint_limits_and_time_samples(model):
    data,audit=data_and_audit(model)
    audit.sample()
    data.qpos[audit.qadr[0]]=audit.position_limits[1]+.01
    data.qvel[audit.dadr[1]]=2*math.pi+.01
    data.time=.002
    kinds={x['kind'] for x in audit.sample()}
    assert {'sample_interval','joint_position','joint_speed'} <= kinds


def test_overload_consecutive_and_reset(model):
    data,audit=data_and_audit(model)
    audit.sample()
    data.actuator_force[:]=0.
    data.actuator_force[0]=15.
    # A one-second consecutive loading period is allowed; greater is rejected.
    for step in range(1,1002):
        data.time=step*.001
        problems=audit.sample()
        overload=any(p['kind']=='joint_continuous_overload' for p in problems)
        assert overload == (step>1000)
    data.actuator_force[0]=0.
    data.time=1.002
    assert not any(p['kind']=='joint_continuous_overload' for p in audit.sample())
    assert audit.overload_s[0] == 0.


def test_warning_is_a_violation(model):
    data,audit=data_and_audit(model)
    data.warning.number[0]=1
    assert any(x['kind']=='mujoco_warning' for x in audit.sample())


def test_four_pin_gate_all_registered_orientations_and_offset(model):
    for quarter in range(4):
        data,audit=data_and_audit(model)
        # Rotate the source module about the true connector-plane center.
        src,dst,offset=audit.frames[0]
        rotation=data.site_xmat[src].reshape(3,3)
        pivot=data.site_xpos[src]+rotation@offset
        axis=rotation[:,2].copy()
        adr=int(model.joint('module_0/module_root').qposadr[0])
        quat=np.empty(4); mat=np.empty(9); out=np.empty(4)
        mujoco.mju_axisAngle2Quat(quat,axis,quarter*math.pi/2)
        mujoco.mju_quat2Mat(mat,quat)
        data.qpos[adr:adr+3]=pivot+mat.reshape(3,3)@(data.qpos[adr:adr+3]-pivot)
        mujoco.mju_mulQuat(out,quat,data.qpos[adr+3:adr+7].copy())
        data.qpos[adr+3:adr+7]=out
        mujoco.mj_forward(model,data)
        audit.sample()
        gap,angle=audit.mating_pose(0)
        assert gap<1e-8 and angle<1e-6
        assert audit.four_pin_clearance_error(0) < 1e-8
        # 0.2 mm is below the broad 1 mm pose tolerance but too much for the
        # assumed 0.1 mm square-hole clearance.
        data.qpos[adr:adr+3] += rotation[:,0]*.0002
        data.time=.001
        audit.sample()
        assert audit.mating_pose(0)[0] < .001
        assert audit.four_pin_clearance_error(0) > .0001


def test_ground_penetration_is_diagnostic_only(model):
    data,audit=data_and_audit(model)
    audit.sample()
    adr=int(model.joint('module_0/module_root').qposadr[0])
    data.qpos[adr+2] -= .02
    data.time=.001
    issues=audit.sample()
    assert not any(x['kind']=='ground_penetration' for x in issues)
    assert audit.max_penetration['ground'] > .003
    assert 'terrain' in (audit.worst_contact['ground']['geom1'],audit.worst_contact['ground']['geom2'])


def test_negative_magnetic_proxy_contact_is_diagnostic_only(model):
    data,audit=data_and_audit(model)
    audit.sample()
    src,dst,offset=audit.frames[0]
    r=data.site_xmat[src].reshape(3,3)
    adr=int(model.joint('module_0/module_root').qposadr[0])
    # 0.2 mm lateral offset fails pin clearance while axial overlap causes
    # penetration of the magnetic spheres. Positive-gap attraction is allowed.
    data.qpos[adr:adr+3] += r[:,0]*.00025 - r[:,2]*.0005
    data.time=.001
    issues=audit.sample()
    assert not any(x['kind']=='collision' for x in issues)
    assert audit.four_pin_gate_excess(0) > 0
    assert audit.max_penetration['magnetic_proxy'] > 0


def test_disabled_geometry_or_contact_cannot_be_certified(model):
    data=mujoco.MjData(model)
    geom=model.geom('physical_connector_0_socket_piece_5').id
    contype=int(model.geom_contype[geom]); affinity=int(model.geom_conaffinity[geom])
    try:
        model.geom_contype[geom]=0
        model.geom_conaffinity[geom]=0
        with pytest.raises(ValueError,match='collision mask'):
            NativeSafetyAudit(model,data)
    finally:
        model.geom_contype[geom]=contype
        model.geom_conaffinity[geom]=affinity
    flags=int(model.opt.disableflags)
    try:
        model.opt.disableflags=flags | int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        with pytest.raises(ValueError,match='contacts enabled'):
            NativeSafetyAudit(model,data)
    finally:
        model.opt.disableflags=flags


def test_connector_only_filter_rejects_robot_shell_or_other_geom(model):
    data=mujoco.MjData(model)
    shell=model.geom('module_0/a1_geom_clipped_shell').id
    visual=model.geom('module_0/a1_geom').id
    for geom in (shell,visual):
        assert model.geom_contype[geom] == 0 and model.geom_conaffinity[geom] == 0
        try:
            model.geom_contype[geom] = 1
            model.geom_conaffinity[geom] = 1
            with pytest.raises(ValueError,match='shell collision|non-connector collision'):
                NativeSafetyAudit(model,data)
        finally:
            model.geom_contype[geom] = 0
            model.geom_conaffinity[geom] = 0


def test_taper_gate_uses_allowed_contact_and_tip_radius(model):
    data,audit=data_and_audit(model)
    src,_,offset=audit.frames[0]
    rotation=data.site_xmat[src].reshape(3,3).copy()
    pivot=data.site_xpos[src]+rotation@offset
    adr=int(model.joint('module_0/module_root').qposadr[0])
    quat=np.empty(4); mat=np.empty(9); out=np.empty(4)
    # Tilt 1.7 degrees about the socket Y axis, then move the root 0.18 mm.
    # The tip's projected offset exceeds 0.2 mm but remains within the wider
    # tapered-tip envelope, while the root remains within 0.2 mm including
    # the explicitly assumed 0.1 mm allowable matching contact depth.
    mujoco.mju_axisAngle2Quat(quat,rotation[:,1],-.03)
    mujoco.mju_quat2Mat(mat,quat)
    data.qpos[adr:adr+3]=pivot+mat.reshape(3,3)@(data.qpos[adr:adr+3]-pivot)+rotation[:,0]*.00018
    mujoco.mju_mulQuat(out,quat,data.qpos[adr+3:adr+7].copy())
    data.qpos[adr+3:adr+7]=out
    audit.sample()
    assert audit.mating_pose(0)[1] < math.radians(2)
    assert audit.four_pin_clearance_error(0) > .0002
    assert audit.four_pin_gate_excess(0) < 0
    # Further lateral offset exceeds the assumed root allowance, despite being
    # well within the broad 1 mm face-position threshold.
    data.qpos[adr:adr+3]+=rotation[:,0]*.00015
    data.time=.001
    audit.sample()
    assert audit.mating_pose(0)[0] < .001
    assert audit.four_pin_gate_excess(0) > 0


def test_taper_gate_and_actual_mujoco_contact_depth_are_diagnostic(model):
    data,audit=data_and_audit(model)
    audit.sample()
    src,_,_=audit.frames[0]
    rotation=data.site_xmat[src].reshape(3,3).copy()
    adr=int(model.joint('module_0/module_root').qposadr[0])
    # 0.15 mm offset exceeds the zero-penetration base clearance, but actual
    # observed pin-wall penetration stays within the assumed 0.1 mm limit.
    data.qpos[adr:adr+3] += rotation[:,0]*.00015-rotation[:,2]*.00001
    data.time=.001
    issues=audit.sample()
    assert audit.four_pin_clearance_error(0) > audit.aperture_clearance
    assert audit.four_pin_gate_excess(0) < 0
    assert 0 < audit.max_penetration['pin_wall'] < audit.mating_penetration
    assert not any(x['kind']=='collision' for x in issues)
    # Deeper insertion keeps the geometric gate satisfied; pin-wall contact
    # exceeds the old reference value but is still diagnostic only.
    data.qpos[adr:adr+3] -= rotation[:,2]*.00029
    data.time=.002
    issues=audit.sample()
    assert audit.four_pin_gate_excess(0) < 0
    assert audit.max_penetration['pin_wall'] > audit.mating_penetration
    assert audit.worst_contact['pin_wall']['distance_m'] < -audit.mating_penetration
    assert not any(x['kind'] in ('collision','ground_penetration') for x in issues)

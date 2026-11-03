import math
from types import SimpleNamespace

import mujoco
import numpy as np

from scripts.planned_docking_feedback import enable_normal_guard
from scripts.polish_safety import CollisionAudit, source_joint_limits
from scripts.run_planned_mjbatch import configure_servo
from scripts.view_headtail_mjbatch_refined import create_refined
from zbot_rl_mjlab.tasks.reconfiguration.reference_trajectory import TimedCurve


def test_poststep_normal_guard_reads_integrated_configuration():
    model = mujoco.MjModel.from_xml_string('''
    <mujoco><option timestep="0.001" gravity="0 0 0"/>
      <worldbody>
        <site name="target"/>
        <body><joint name="slide" type="slide" axis="0 0 1"/>
          <geom type="sphere" size="0.01" mass="1"/>
          <site name="source" pos="0 0 -0.001"/>
        </body>
      </worldbody>
      <equality><weld name="dock" site1="source" site2="target" active="false"/></equality>
    </mujoco>''')
    data = mujoco.MjData(model)
    data.qvel[0] = 2.
    mujoco.mj_forward(model, data)
    mujoco.mj_step(model, data)
    assert data.site_xpos[model.site('source').id, 2] < 0
    state = SimpleNamespace(model=model, data=data, phase='CURL', old=np.zeros(6, dtype=int),
                            target=5, measure=lambda: None, failure='')
    state.stop = lambda reason: setattr(state, 'failure', reason)
    enable_normal_guard(state, current_geometry=True)
    state.measure()
    assert 'normal plane crossed' in state.failure


def test_speed_only_curve_and_asset_limits():
    points=np.array([[0.], [1.]])
    speed_only=TimedCurve(points, speed=1., acceleration=None, dt=.001)
    acceleration_limited=TimedCurve(points, speed=1., acceleration=.1, dt=.001)
    assert speed_only.duration < acceleration_limited.duration
    assert np.max(np.abs(speed_only.v)) <= 1.0 + 1e-8
    assert np.allclose(source_joint_limits(), [-6.28, 6.28])


def test_actual_actuator_force_saturates_and_source_position_limit_is_checked():
    f=create_refined('forward')
    try:
        configure_servo(f, 15., 1.5)
        f.cfg.raw_torque_limit_enabled=False
        f.phase='FINAL_STRAIGHT_HOLD'
        f.curve=None
        f.qref=f.data.qpos[f.qadr].copy()+2.
        f.apply()
        assert f.phase != 'FAILED'
        mujoco.mj_step(f.model,f.data)
        assert np.max(np.abs(f.data.qfrc_actuator[f.dofs])) <= 20. + 1e-9
        f.data.qpos[f.qadr[0]]=6.29
        CollisionAudit(f,enforce_source_limits=True).sample()
        assert 'source joint position limit' in f.failure
    finally:
        f.close()

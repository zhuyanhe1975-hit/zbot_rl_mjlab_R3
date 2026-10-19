import math

import mujoco
import pytest
from mjlab.actuator import BuiltinPositionActuatorCfg

from zbot_rl_mjlab import robot
from zbot_rl_mjlab.experiments import addons
from zbot_rl_mjlab.robot import available_torque, robot_cfg

RPM = 2 * math.pi / 60


def test_datasheet_constants():
    assert robot.RS00_PEAK_TORQUE == 14.0
    assert robot.RS00_RATED_TORQUE == 5.0
    assert math.isclose(robot.RS00_NO_LOAD_SPEED, 315 * RPM)
    assert robot.RS00_KP_RANGE == (0.0, 500.0)
    assert robot.RS00_KD_RANGE == (0.0, 5.0)


def test_pd_gains_and_their_randomization_stay_inside_the_mit_range():
    for gain, (low, high) in (
        (robot.STIFFNESS, robot.RS00_KP_RANGE),
        (robot.DAMPING, robot.RS00_KD_RANGE),
    ):
        assert low <= gain * min(addons.GAIN_SCALE)
        assert gain * max(addons.GAIN_SCALE) <= high


def test_actuator_is_the_one_every_existing_checkpoint_was_trained_with():
    # 修改任意一项都会改变所有归档模型使用的物理设置；
    # 此前切换到另一执行器类曾导致这些模型完全无法回放。
    (actuator,) = robot_cfg().articulation.actuators
    assert type(actuator) is BuiltinPositionActuatorCfg
    assert (actuator.stiffness, actuator.damping) == (15.0, 1.5)
    assert actuator.effort_limit == robot.SIM_EFFORT_LIMIT == 20.0
    assert actuator.armature == 0.001


def test_simulated_torque_limit_exceeds_the_hardware_peak():
    # 已记录的差距：仿真器允许的输出仍超过 RS00 的能力。
    assert robot.SIM_EFFORT_LIMIT > robot.RS00_PEAK_TORQUE


def test_actuator_has_one_control_input_per_joint():
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="link")
    body.add_joint(name="j", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1])
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.05, 0, 0], mass=1.0)
    (cfg,) = robot_cfg().articulation.actuators
    cfg.build(None, [0], ["j"]).edit_spec(spec, ["j"])
    assert spec.compile().nu == 1  # 12 维 ctrl 曾导致 play.py 的张量形状不匹配


def test_torque_speed_reference_curve():
    # 在 100 rpm 以内保持峰值，此后采用数据手册给出的点，空载转速处为零。
    for rpm in (0, 50, 100):
        assert available_torque(rpm * RPM) == 14.0
    for rpm, expected in robot.RS00_TN_CURVE:
        assert math.isclose(available_torque(rpm * RPM), expected, abs_tol=1e-6), rpm
    assert available_torque(315 * RPM) == pytest.approx(0.5)
    assert available_torque(320 * RPM) == 0.0
    assert available_torque(-200 * RPM) == available_torque(200 * RPM)  # 对称
    limits = [available_torque(r * RPM) for r in range(100, 316, 5)]
    assert limits == sorted(limits, reverse=True)  # 不随转速升高而增大


def test_rated_torque_is_below_the_curve_at_the_rated_speed():
    assert available_torque(100 * RPM) > robot.RS00_RATED_TORQUE


def test_overload_table_is_ordered():
    seconds = [s for s, _ in robot.RS00_OVERLOAD]
    torques = [t for _, t in robot.RS00_OVERLOAD]
    assert seconds == sorted(seconds)
    assert torques == sorted(torques, reverse=True)
    assert torques[0] == robot.RS00_PEAK_TORQUE
    assert torques[-1] > robot.RS00_RATED_TORQUE

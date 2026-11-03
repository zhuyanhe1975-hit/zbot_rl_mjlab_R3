"""Independent finite-difference checks of floating, moving-frame site Jacobians."""

import importlib.util
from pathlib import Path

import mujoco
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("four_site_jacobian", Path(__file__).resolve().parents[1] / "scripts/four_site_jacobian.py")
four_site = importlib.util.module_from_spec(spec)
spec.loader.exec_module(four_site)


def _model():
    sites = ''.join(f'<site name="{{name}}{i}" pos="{x} {y} 0"/>' for i, (x, y) in enumerate(((.05, 0), (0, .05), (-.05, 0), (0, -.05))))
    return mujoco.MjModel.from_xml_string(f'''<mujoco>
      <option gravity="0 0 0" jacobian="dense"/>
      <worldbody>
        <body name="a" pos=".3 -.1 .2" euler=".3 -.4 .2"><freejoint/><geom type="sphere" size=".03" mass="1"/>{sites.format(name="a")}</body>
        <body name="b" pos="-.1 .2 .3" euler="-.5 .2 -.3"><freejoint/><geom type="sphere" size=".03" mass="1"/>{sites.format(name="b")}</body>
      </worldbody><equality><weld name="lock" body1="a" body2="b"/></equality>
    </mujoco>''')


def _ids(model, name):
    return [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f'{name}{i}') for i in range(4)]


@pytest.mark.parametrize('closed', [False, True])
@pytest.mark.parametrize('offset', [0, 1, 2, 3])
def test_four_site_moving_frame_jacobian_matches_finite_difference(closed, offset):
    model = _model()
    data = mujoco.MjData(model)
    data.eq_active[0] = closed
    mujoco.mj_forward(model, data)
    source, target = _ids(model, 'a'), _ids(model, 'b')
    target = target[offset:] + target[:offset]
    errors, jacobian = four_site.four_site_kinematics(model, data, source, target, target[0])
    rng = np.random.default_rng(91)
    velocity = rng.normal(size=model.nv)
    epsilon = 1e-7
    values = []
    for sign in (-1, 1):
        scratch = mujoco.MjData(model)
        scratch.qpos[:] = data.qpos
        mujoco.mj_integratePos(model, scratch.qpos, velocity, sign * epsilon)
        mujoco.mj_forward(model, scratch)
        values.append(four_site.four_site_kinematics(model, scratch, source, target, target[0])[0])
    numerical = (values[1] - values[0]) / (2 * epsilon)
    np.testing.assert_allclose(jacobian @ velocity, numerical.ravel(), atol=2e-9, rtol=2e-8)
    assert np.linalg.norm(errors) > 0


def test_closed_equality_nullspace_excludes_weld_violating_velocities():
    model = _model()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    constraint = four_site.equality_constraint_jacobian(model, data)
    assert constraint.shape == (6, model.nv)
    nullspace = four_site.equality_nullspace(constraint)
    assert nullspace.shape == (model.nv, 6)
    np.testing.assert_allclose(constraint @ nullspace, 0, atol=1e-12)
    np.testing.assert_allclose(nullspace.T @ nullspace, np.eye(6), atol=1e-12)
    data.eq_active[0] = 0
    mujoco.mj_forward(model, data)
    assert four_site.equality_nullspace(four_site.equality_constraint_jacobian(model, data)).shape == (model.nv, model.nv)

"""Compare the isolated native-adhesion interface in Batch and MuJoCo.

Run with ./scripts/python_mjbatch.sh -m pytest -q tests/test_native_adhesion_mjbatch.py.
This checks API compatibility, not full-robot safety or magnet calibration.
"""

import math

import mujoco
import numpy as np
import pytest

mjbatch = pytest.importorskip("mjbatch")

from scripts.diagnose_native_magnet_adhesion import make_model


def test_isolated_native_adhesion_batch_matches_serial():
    model = make_model(timestep=0.001)
    model.opt.gravity[:] = 0
    batch = mjbatch.Batch(model, num_sims=4, num_threads=2, forward=True)
    qpos = batch.bind("qpos")
    qvel = batch.bind("qvel")
    socket = model.body("socket").id
    pin = model.body("pin_plate").id
    socket_qadr = int(model.jnt_qposadr[model.body_jntadr[socket]])
    pin_qadr = int(model.jnt_qposadr[model.body_jntadr[pin]])

    qpos[:] = model.qpos0
    # Aligned in range, aligned beyond range, 90-degree rotation, lateral offset.
    qpos[:, pin_qadr + 2] = [0.002, 0.004, 0.002, 0.002]
    qpos[2, pin_qadr + 3 : pin_qadr + 7] = [math.sqrt(0.5), 0, 0, math.sqrt(0.5)]
    qpos[3, pin_qadr] = 0.006
    qvel[:] = 0
    initial = qpos.copy()

    adhesion = batch.bind("qfrc_adhesion")
    batch.forward()
    np.testing.assert_allclose(
        np.linalg.norm(adhesion, axis=1), [20 * math.sqrt(2), 0, 20 * math.sqrt(2), 0], atol=1e-8
    )

    serial = []
    for row in initial:
        data = mujoco.MjData(model)
        data.qpos[:] = row
        for _ in range(100):
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        serial.append(data)
    batch.step(nstep=100)

    np.testing.assert_allclose(qpos, np.stack([d.qpos for d in serial]), rtol=0, atol=1e-12)
    np.testing.assert_allclose(qvel, np.stack([d.qvel for d in serial]), rtol=0, atol=1e-12)
    assert np.count_nonzero(batch.bind("warning")[:, :, 1]) == 0
    assert all(np.count_nonzero(data.warning.number) == 0 for data in serial)
    # In-range aligned pairs attract; the out-of-range pair stays separated.
    relative_z = qpos[:, pin_qadr + 2] - qpos[:, socket_qadr + 2]
    assert relative_z[0] < 0.0001 and relative_z[2] < 0.0001
    assert relative_z[1] == pytest.approx(0.004)

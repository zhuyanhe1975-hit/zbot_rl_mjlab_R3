"""The viewer must replay the actual states and model from the same run."""
import hashlib
import json
import subprocess
import sys

import mujoco
import numpy as np


def test_run_records_integrated_states_and_exact_model(tmp_path):
    report = tmp_path / 'result.json'
    completed = subprocess.run(
        [sys.executable, '-m', 'scripts.run_native_reconfiguration',
         '--duration-s', '.005', '--output', str(report), '--no-index'],
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 1, completed.stderr  # intentional timeout
    result = json.loads(report.read_text())
    assert result['timeout'] and not result['success']
    assert result['geom_adhesion_capacity_N_per_contact_assumed'] == 35.
    model_path = tmp_path / result['model_file']
    trajectory_path = tmp_path / result['trajectory_file']
    assert hashlib.sha256(model_path.read_bytes()).hexdigest() == result['model_sha256']
    assert hashlib.sha256(trajectory_path.read_bytes()).hexdigest() == result['trajectory_sha256']
    model = mujoco.MjModel.from_binary_path(str(model_path))
    assert model.geom_adhesion[model.geom('physical_connector_0_adhesion_source_0').id] == 35.
    with np.load(trajectory_path, allow_pickle=False) as data:
        assert data['qpos'].shape == (6, model.nq)
        assert data['qvel'].shape == (6, model.nv)
        assert data['ctrl'].shape == (6, model.nu)
        np.testing.assert_allclose(data['time'], np.arange(6) * .001, atol=1e-12)
        assert data['phase'][0] == 'INITIAL_CHECK'
        assert result['trajectory_samples'] == len(data['time'])

"""Replay must show recorded states, including failed runs, without resimulation."""
import json

import mujoco
import numpy as np
import pytest

from scripts.view_native_reconfiguration import latest_recording, load_recording, replay, set_frame


def recorded_run(tmp_path):
    model = mujoco.MjModel.from_xml_string('''
        <mujoco><worldbody><body><freejoint/>
        <geom type="sphere" size=".01" mass="1"/></body></worldbody></mujoco>
    ''')
    mujoco.mj_saveModel(model, str(tmp_path / 'model.mjb'))
    data = mujoco.MjData(model)
    qpos = np.tile(data.qpos, (3, 1))
    qpos[:, 0] = [0., .1, .2]
    qvel = np.zeros((3, model.nv))
    ctrl = np.zeros((3, model.nu))
    np.savez(tmp_path / 'states.npz', time=np.array([0., .001, .002]),
             qpos=qpos, qvel=qvel, ctrl=ctrl,
             phase=np.array(['INITIAL_HOLD', 'CURL', 'FAILED']),
             target_interface=np.array([6, 6, 6]))
    report = {'success': False, 'failure': 'test',
              'trajectory_file': 'states.npz', 'model_file': 'model.mjb'}
    path = tmp_path / 'result.json'
    path.write_text(json.dumps(report))
    return path, model, qpos


def test_failed_run_replays_exact_recorded_frame(tmp_path, capsys):
    path, _, qpos = recorded_run(tmp_path)
    _, report, model, arrays = load_recording(path)
    assert not report['success']
    data = mujoco.MjData(model)
    set_frame(model, data, arrays, 2)
    np.testing.assert_array_equal(data.qpos, qpos[2])
    assert data.time == .002
    assert replay(path, headless=True) == 0
    assert 'phase=FAILED' in capsys.readouterr().out


def test_historical_summary_cannot_be_claimed_as_a_replay(tmp_path):
    path = tmp_path / 'result.json'
    path.write_text(json.dumps({'success': False, 'events': []}))
    with pytest.raises(ValueError, match='no recorded trajectory and model'):
        load_recording(path)


def test_wrong_model_dimensions_are_rejected(tmp_path):
    path, _, _ = recorded_run(tmp_path)
    with np.load(tmp_path / 'states.npz', allow_pickle=False) as saved:
        arrays = {name: np.array(saved[name]) for name in saved.files}
    arrays['qpos'] = arrays['qpos'][:, :-1]
    np.savez(tmp_path / 'states.npz', **arrays)
    with pytest.raises(ValueError, match='invalid qpos'):
        load_recording(path)


def test_latest_custom_path_requires_matching_run_id(tmp_path):
    path, _, _ = recorded_run(tmp_path)
    report = json.loads(path.read_text())
    report['run_id'] = 'one'
    path.write_text(json.dumps(report))
    index = tmp_path / 'latest.json'
    index.write_text(json.dumps({'result_path': str(path), 'run_id': 'one'}))
    assert latest_recording(index, tmp_path / 'default.json') == path
    index.write_text(json.dumps({'result_path': str(path), 'run_id': 'another'}))
    with pytest.raises(ValueError, match='does not match'):
        latest_recording(index, tmp_path / 'default.json')


def test_mismatched_model_checksum_is_rejected(tmp_path):
    path, _, _ = recorded_run(tmp_path)
    report = json.loads(path.read_text())
    report['model_sha256'] = '0' * 64
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='checksum mismatch'):
        load_recording(path)

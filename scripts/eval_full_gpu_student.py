"""Play a full-FSM GPU student checkpoint with deterministic policy actions."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import argparse
import json
import time
from pathlib import Path

import torch
import warp as wp
from mjlab.envs import ManagerBasedRlEnv
from train_full_batched_student import ActorCritic, REFINED_MODE, reference_identity
from zbot_rl_mjlab.tasks.reconfiguration.full_batched_env import make_full_batched_env_cfg
from zbot_rl_mjlab.tasks.reconfiguration.full_cpu_student import PHASES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--max-time', type=float, default=120.)
    parser.add_argument('--output', type=Path, default=Path('outputs/reconfiguration/full_gpu_student/evaluation.json'))
    parser.add_argument('--viewer', action='store_true')
    parser.add_argument('--headless', dest='viewer', action='store_false')
    parser.set_defaults(viewer=False)
    args = parser.parse_args()
    if args.max_time <= 0:
        parser.error('--max-time must be positive')
    torch.set_num_threads(1)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    mode,reference_sha256=reference_identity(checkpoint['reference'])
    if checkpoint.get('mode') != mode or (mode==REFINED_MODE and checkpoint.get('reference_sha256')!=reference_sha256):
        raise ValueError('Incompatible GPU student mode or changed teacher reference.')
    cfg = make_full_batched_env_cfg(1, checkpoint['reference'], checkpoint['velocity_scale'], checkpoint['torque_scale'])
    cfg.auto_reset = False
    env = ManagerBasedRlEnv(cfg, device='cuda:0')
    previous_stream = torch.cuda.current_stream(env.device)
    torch.cuda.set_stream(torch.cuda.ExternalStream(wp.get_stream(env.sim.wp_device).cuda_stream, device=env.device))
    start = time.perf_counter()
    sim_time = 0.
    episode_return = 0.
    max_action = 0.
    terminal = None
    try:
        with torch.no_grad():
            obs, _ = env.reset()
            dim = obs['actor'].shape[-1]
            if dim != checkpoint['obs_dim']:
                raise ValueError(f'Checkpoint observation dimension {checkpoint["obs_dim"]} does not match environment {dim}.')
            net = ActorCritic(dim).to(env.device)
            net.load_state_dict(checkpoint['model'])
            net.eval()
            print(f'GPU student: {args.checkpoint}, iteration={checkpoint.get("iteration")}, observation={dim}', flush=True)

            def step():
                nonlocal obs, sim_time, episode_return, max_action, terminal
                action = torch.tanh(net.actor(obs['actor']))
                max_action = max(max_action, float(action.abs().max()))
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_return += float(reward[0])
                sim_time += env.step_dt
                if bool(terminated[0]) or bool(truncated[0]):
                    terminal = 'success' if bool(env.termination_manager.get_term('success')[0]) else ('timeout' if bool(truncated[0]) else 'failure')

            if args.viewer:
                import copy
                import mujoco
                import mujoco.viewer
                model = copy.copy(env.sim.mj_model)
                data = mujoco.MjData(model)
                with mujoco.viewer.launch_passive(model, data) as viewer:
                    viewer.cam.lookat[:] = [0., 0., .08]
                    viewer.cam.distance = 1.4
                    viewer.cam.elevation = -55
                    anchor = time.perf_counter()
                    while viewer.is_running():
                        desired = time.perf_counter() - anchor
                        deadline = time.perf_counter() + .012
                        while terminal is None and sim_time < min(desired, args.max_time) and time.perf_counter() < deadline:
                            step()
                        data.qpos[:] = env.sim.data.qpos[0].cpu().numpy()
                        data.qvel[:] = env.sim.data.qvel[0].cpu().numpy()
                        data.eq_active[:] = env.sim.data.eq_active[0].cpu().numpy()
                        mujoco.mj_forward(model, data)
                        s = env._disconnect_state
                        loads = (s.utilization[0] * 80.).cpu().tolist()
                        connected = s.connected()[0].cpu().tolist()
                        phase = PHASES[int(s.fsm.phase[0])]
                        labels = 'Time\nPhase\nTarget\nFace loads 1-6 (N)\nConnected 1-6\nStatus'
                        values = (f'{sim_time:.2f}s\n{phase}\n{int(s.fsm.target[0])+1}\n'
                                  + ' '.join(f'{x:.1f}' for x in loads) + '\n'
                                  + ' '.join('1' if x else '0' for x in connected) + '\n'
                                  + (terminal or ('time limit' if sim_time >= args.max_time else 'running')))
                        viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100.value,
                                          mujoco.mjtGridPos.mjGRID_TOPRIGHT.value, labels, values))
                        viewer.sync()
                        time.sleep(.002)
            else:
                while terminal is None and sim_time < args.max_time:
                    step()
            s = env._disconnect_state
            report = {'checkpoint': str(args.checkpoint), 'iteration': checkpoint.get('iteration'),
                      'mode': mode, 'obs_dim': dim, 'success': terminal == 'success',
                      'terminal': terminal or ('time_limit' if sim_time >= args.max_time else 'viewer_closed'),
                      'phase': PHASES[int(s.fsm.phase[0])], 'failure_code': int(s.fsm.failure_code[0]),
                      'time_s': sim_time, 'episode_return': episode_return,
                      'max_abs_policy_action': max_action, 'wall_time_s': time.perf_counter() - start}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + '\n')
            print('EVALUATION', json.dumps(report), flush=True)
            print(f'Report: {args.output}', flush=True)
            return 1 if terminal in ('failure', 'timeout') else 0
    finally:
        torch.cuda.set_stream(previous_stream)
        env.close()


if __name__ == '__main__':
    raise SystemExit(main())

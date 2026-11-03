"""Run the mixed closed-loop reconfiguration with real dynamics; no learning."""
import argparse
import json
from pathlib import Path

import mujoco
import torch
import zbot_rl_mjlab  # register tasks
from mjlab.envs import ManagerBasedRlEnv
from zbot_rl_mjlab.tasks.reconfiguration.mixed_closed_dynamic import EXPERIMENT_ID, mixed_dynamic_env_cfg
from zbot_rl_mjlab.tasks.reconfiguration.state_machine import Phase
from zbot_rl_mjlab.tasks.reconfiguration.disconnect.state import state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--viewer', action='store_true')
    p.add_argument('--headless', dest='viewer', action='store_false')
    p.set_defaults(viewer=False)
    p.add_argument('--max-time', type=float, default=180.)
    p.add_argument('--output', type=Path, default=Path('outputs/reconfiguration/mixed_closed_dynamic/latest.json'))
    args = p.parse_args()
    if args.max_time <= 0: p.error('--max-time must be positive')
    env = ManagerBasedRlEnv(mixed_dynamic_env_cfg(), device='cuda:0')
    try:
        with torch.inference_mode():
            env.reset()
            a = env.action_manager.get_term('joint_torque'); c = a.distance_controller; s = state(env)
            zero = torch.zeros(1, 6, device=env.device)
            previous = None; trace = []
            def sample(t):
                pair = s.worst_pair.tolist()
                names = [mujoco.mj_id2name(env.sim.mj_model, mujoco.mjtObj.mjOBJ_GEOM, i) if i >= 0 else None for i in pair]
                return {'t': round(t, 5), 'sequence_index': c.index, 'target_connector': env.cfg.target_connector + 1,
                        'phase': Phase(c.phase.item()).name, 'phase_elapsed_s': c.phase_time.item(),
                        'gap_m': s.geometry()['distance'].item(), 'orientation_error_deg': float(torch.rad2deg(c.angle_error)[0]),
                        'joint_angle_rad': torch.cat([m.data.joint_pos for m in a.modules], -1)[0].tolist(),
                        'actual_torque_Nm': a.torque[0].tolist(), 'loads_N': (s.utilization[0] * env.cfg.break_force).tolist(),
                        'connected': s.connected()[0].tolist(), 'registered_new_angle': env.sim.data.eq_active[0, s.variant_ids].bool().tolist(),
                        'failure_code': c.failure_code.item(), 'safety_violation': s.safety_margin_violation.item(),
                        'wrong_break': s.wrong_break.item(), 'torque_failure': a.torque_failure.item(),
                        'peak_other_load_N': s.max_other_load.item(), 'loads_at_break_N': s.loads_at_break[0].tolist(),
                        'module_penetration_max_m': s.max_module_depth.item(), 'worst_geom_pair': names,
                        'ground_penetration_max_m': s.max_ground_depth.item(), 'open_face_excluded_penetration_max_m': s.max_excluded_depth,
                        'penetration_violation': s.contact_violation.item(), 'loading': a.loading.last_report}
            def record(t, force=False):
                nonlocal previous
                key = (c.index, c.phase.item())
                if force or key != previous:
                    row = sample(t); trace.append(row); print(json.dumps(row), flush=True); previous = key
            if args.viewer:
                from mjlab.viewer import NativeMujocoViewer
                def policy(obs):
                    record(env._sim_step_counter * env.physics_dt)
                    return zero
                NativeMujocoViewer(env, policy).run()
            else:
                for k in range(int(args.max_time / env.step_dt)):
                    env.step(zero)
                    record((k+1)*env.step_dt, k % 25 == 0)
                    if c.phase.item() in (Phase.DONE, Phase.FAILED): break
            terminal = sample(env._sim_step_counter * env.physics_dt)
            success = c.index == 4 and c.phase.item() == Phase.DONE
            report = {'experiment': EXPERIMENT_ID, 'target_angles_deg': [270,270,180,90,90],
                      'success': success, 'terminal': terminal, 'completed_connectors': c.history, 'trace': trace,
                      'penetration_stop_m': {'modules': .002, 'ground': .003, 'open_excluded_face': .002},
                      'note': 'Excluded open face is distance-monitored, not collision-resolved by the solver.'}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2))
            print('REPORT', args.output, 'success=', success, 'phase=', terminal['phase'], flush=True)
            return 0 if success else 1
    finally:
        env.close()

if __name__ == '__main__': raise SystemExit(main())

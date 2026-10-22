"""Compare sensor-history students on independent signed and combined commands."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

import zbot_rl_mjlab  # noqa: F401
from zbot_rl_mjlab.student_ppo import UniversalStudentPPORunner
from zbot_rl_mjlab.student_ppo_evaluation import StudentPPOEvaluator

TASK = 'Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Student-PPO'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoints', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, default=Path('outputs/student_ppo_evaluation.json'))
    parser.add_argument('--num-envs', type=int, default=80)
    parser.add_argument('--seconds', type=float, default=12.)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    report = {'protocol': {'requested_num_envs': args.num_envs, 'seconds': args.seconds,
                            'seed': 42, 'randomized': False, 'sensor_noise': False,
                            'completion': 'student_ppo_v2_hysteresis'}, 'checkpoints': {}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for checkpoint in args.checkpoints:
        evaluator = StudentPPOEvaluator(device=args.device, num_envs=args.num_envs,
                                        seconds=args.seconds, env_cfg=load_env_cfg(TASK, play=True), randomized=False)
        try:
            evaluator._build()
            assert evaluator.wrapper is not None
            runner = UniversalStudentPPORunner(evaluator.wrapper, asdict(load_rl_cfg(TASK)), device=args.device)
            runner.load(str(checkpoint), load_cfg={'actor': True}, map_location=args.device)
            values = evaluator(runner.alg.actor)
            report['checkpoints'][str(checkpoint)] = {'num_envs': evaluator.num_envs, 'metrics': values}
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False))
            print(f"{checkpoint}: biped={values['biped_moving_success']:.0%} snake={values['snake_moving_success']:.0%} "
                  f"down/up handoff={values['descent_handoff_fraction']:.0%}/{values['ascent_handoff_fraction']:.0%}", flush=True)
        finally:
            evaluator.close()


if __name__ == '__main__':
    main()

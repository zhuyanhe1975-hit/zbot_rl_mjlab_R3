"""Qualify an explicit site-frame snake checkpoint; failed acceptance exits nonzero."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

TASK_ID = 'Mjlab-Zbot-6dof-Snake-Site-Omni-Teacher'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--num-envs', type=int, default=72)
    parser.add_argument('--seconds', type=float, default=10.)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--randomized', action='store_true')
    args = parser.parse_args()
    if args.seconds < 10.:
        parser.error('qualification requires at least 10 seconds')
    if not args.checkpoint.is_file():
        parser.error(f'checkpoint does not exist: {args.checkpoint}')
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    import zbot_rl_mjlab  # noqa: F401
    from zbot_rl_mjlab.site_snake_evaluation import SiteSnakeEvaluator

    cfg = load_env_cfg(TASK_ID, play=True)
    runner_cls = load_runner_cls(TASK_ID)
    if runner_cls is None:
        raise TypeError('site snake task must register its privileged teacher runner')
    runner_cls.prepare_playback_env_cfg(cfg, args.checkpoint)
    evaluator = SiteSnakeEvaluator(args.device, args.num_envs, args.seconds, cfg, args.randomized)
    try:
        evaluator._build()
        agent = load_rl_cfg(TASK_ID)
        runner = runner_cls(evaluator.wrapper, asdict(agent), device=args.device)
        runner.load(str(args.checkpoint), load_cfg={'actor': True}, strict=True, map_location=args.device)
        metrics = evaluator(runner.alg.actor)
        report = {'checkpoint': str(args.checkpoint.resolve()), 'task': TASK_ID,
                  'randomized': args.randomized, 'num_envs': evaluator.num_envs,
                  'seconds': args.seconds, 'metrics': metrics, **evaluator.last_report}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        print(json.dumps(metrics, ensure_ascii=False))
        return 0 if report['passed'] else 1
    finally:
        evaluator.close()


if __name__ == '__main__':
    sys.exit(main())

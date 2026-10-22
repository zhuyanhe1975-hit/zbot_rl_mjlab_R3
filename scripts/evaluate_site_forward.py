"""Measure fixed-command forward motion and collateral costs without automatic resets."""
import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True, choices=(
        'Mjlab-Zbot-6dof-Snake-Site-Forward-Teacher',
        'Mjlab-Zbot-6dof-Snake-Site-Forward-Relaxed-Teacher'))
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    import torch
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    import zbot_rl_mjlab  # noqa: F401
    from zbot_rl_mjlab.site_snake_evaluation import SiteSnakeEvaluator
    from zbot_rl_mjlab.tasks.snake import mdp as snake_mdp
    from zbot_rl_mjlab.tasks.snake import site_forward, site_omni
    from zbot_rl_mjlab.tasks.transformation.mdp import body_masses
    from zbot_rl_mjlab.tasks.universal import mdp
    from zbot_rl_mjlab.tasks.universal.omni import motion_site_index, site_heading
    from zbot_rl_mjlab.tasks.universal.omni import (
        tracking_velocity as com_tracking_velocity,
    )

    cfg = load_env_cfg(args.task, play=True)
    runner_cls = load_runner_cls(args.task)
    if runner_cls is None:
        raise TypeError('forward task requires its own runner')
    runner_cls.prepare_playback_env_cfg(cfg, args.checkpoint)
    evaluator = SiteSnakeEvaluator(args.device, 72, 10., cfg)
    try:
        evaluator._build()
        env, wrapper = evaluator.env, evaluator.wrapper
        assert env is not None and wrapper is not None
        runner = runner_cls(wrapper, asdict(load_rl_cfg(args.task)), device=args.device)
        runner.load(str(args.checkpoint), load_cfg={'actor': True}, map_location=args.device)
        actor = runner.alg.actor
        actor.eval()
        wrapper.reset()
        term = mdp.command_term(env)
        term.set_velocity(0., 0., 0.)
        obs = wrapper.get_observations()
        samples, failures = [], torch.zeros(env.num_envs, device=args.device, dtype=torch.bool)
        with torch.inference_mode():
            for _ in range(round(1./env.step_dt)):
                obs, _, _, _ = wrapper.step(actor(obs, stochastic_output=False))
            heading = site_heading(env).clone()
            data = env.scene['robot'].data
            masses = body_masses(env)

            def com():
                return (data.body_com_pos_w*masses[..., None]).sum(-2)/masses.sum(-1, keepdim=True)

            initial_com = com().clone()
            initial_site = data.site_pos_w[:, motion_site_index(env)].clone()
            term.set_velocity(.1, 0., 0.)
            for _ in range(round(10./env.step_dt)):
                obs, _, _, _ = wrapper.step(actor(obs, stochastic_output=False))
                failures |= (site_omni.site_tilt_limit_reached(env) | snake_mdp.self_collision(env)
                             | ~torch.isfinite(obs['actor']).all(-1))
                components = site_omni._velocity_tracking_reward(env)
                samples.append(torch.cat((
                    mdp.tracking_velocity(env),
                    site_forward.target_hard_limit_fraction(env)[:, None],
                    site_forward.mean_abs_joint_velocity(env)[:, None],
                    components[:, :1]*4., components[:, 1:2]*4., components[:, 2:3],
                    com_tracking_velocity(env)[:, :2],
                ), -1))
            values = torch.stack(samples)
            steady = values[len(values)//2:]
            keys = ('head_vx', 'head_vy', 'head_wz', 'target_hard_limit_fraction',
                    'mean_abs_joint_velocity', 'forward_reward_rate',
                    'baseline_lateral_reward_rate', 'baseline_yaw_reward_rate')
            keys += ('com_vx_head_frame', 'com_vy_head_frame')
            displacement = ((com()-initial_com)[:, :2]*heading).sum(-1)
            site_displacement = ((data.site_pos_w[:, motion_site_index(env)]-initial_site)[:, :2]*heading).sum(-1)
            report = {
                'task': args.task, 'checkpoint': str(args.checkpoint.resolve()),
                'command': [.1, 0., 0.], 'seconds': 10., 'settle_seconds': 1.,
                'measurement': 'last 5 seconds; displacement covers all 10 commanded seconds',
                'initial_yaws_deg': [0, 90, 180, 270], 'num_envs': env.num_envs,
                'stochastic': False, 'failure_fraction': failures.float().mean().item(),
                'steady': dict(zip(keys, steady.mean((0, 1)).tolist())),
                'sustained_vx_at_least_half_command_fraction': (steady[..., 0]>=.05).float().mean().item(),
                'com_displacement_initial_heading_m': displacement.mean().item(),
                'com_net_displacement_xy_m': (com()-initial_com)[:, :2].norm(dim=-1).mean().item(),
                'head_displacement_initial_heading_m': site_displacement.mean().item(),
                'head_mean_abs_vy': steady[..., 1].abs().mean().item(),
                'head_mean_abs_wz': steady[..., 2].abs().mean().item(),
                'com_mean_speed_xy': steady[..., 8:10].norm(dim=-1).mean().item(),
                'per_heading': [],
            }
            for i, yaw in enumerate((0, 90, 180, 270)):
                report['per_heading'].append({'yaw_deg': yaw,
                    **dict(zip(keys, steady[:, i::4].mean((0, 1)).tolist())),
                    'failure_fraction': failures[i::4].float().mean().item()})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        evaluator.close()


if __name__ == '__main__':
    main()

"""Compact multi-teacher progress without changing the upstream metric writer."""

from __future__ import annotations

import inspect
import statistics
import sys
from contextlib import redirect_stdout
from datetime import timedelta
from functools import wraps
from typing import Any, TextIO

import torch


class _WithoutTrainingBanner:
    """Forward diagnostics, suppressing only the upstream logger's single print."""

    def __init__(self, stream: TextIO, width: int):
        self.stream = stream
        self.prefix = "#" * width + "\n\033[1m"
        self.skip_newline = False

    def write(self, value: str) -> int:
        if value.startswith(self.prefix):
            self.skip_newline = True
        elif self.skip_newline and value == "\n":
            self.skip_newline = False
        else:
            self.skip_newline = False
            self.stream.write(value)
        return len(value)

    def flush(self) -> None:
        self.stream.flush()


def install_compact_logging(logger: Any, *, rewards_only: bool = False) -> None:
    """Replace this logger's console banner; retain its metric/video/save behavior.

    The stdout redirect is scoped to ``log`` and forwards unrelated output. Like
    the upstream logger, this hook is intended for the runner's single log thread.
    """
    original = logger.log
    signature = inspect.signature(original)

    @wraps(original)
    def compact_log(*args: Any, **kwargs: Any) -> Any:
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        # The original handles disabled writers and remains responsible for all
        # mutations and side effects, including exceptions and clearing extras.
        with redirect_stdout(_WithoutTrainingBanner(sys.stdout, values.get("width", 80))):
            result = original(*args, **kwargs)
        if logger.writer is None:
            return result

        losses = values["loss_dict"]
        if rewards_only:
            width, pad = values.get('width', 80), values.get('pad', 40)
            lines = ['#' * width, f" Learning iteration {values['it']}/{values['total_it']} ".center(width), '']
            collect, learn = values['collect_time'], values['learn_time']
            size = logger.cfg['num_steps_per_env'] * logger.num_envs * logger.gpu_world_size
            lines.extend([
                f"{'Total steps:':>{pad}} {logger.tot_timesteps}",
                f"{'Steps per second:':>{pad}} {int(size / (collect + learn))}",
                f"{'Collection time:':>{pad}} {collect:.3f}s",
                f"{'Learning time:':>{pad}} {learn:.3f}s",
                f"{'Mean action std:':>{pad}} {values['action_std'].mean().item():.3f}",
            ])
            reward = f'{statistics.mean(logger.rewbuffer):.2f}' if logger.rewbuffer else '--'
            length = f'{statistics.mean(logger.lenbuffer):.2f}' if logger.lenbuffer else '--'
            lines.extend([
                f"{'Mean reward:':>{pad}} {reward}",
                f"{'Mean episode length:':>{pad}} {length}",
                f"{'Current episode length (steps):':>{pad}} {logger.cur_episode_length.mean().item():.2f}",
            ])
            for key, label, precision in (
                ('rollout_episode_age_s', 'Current episode age (s)', 2),
                ('rollout_reset_count', 'Resets (rollout)', 0),
                ('rollout_termination_count', 'Terminations (no timeout)', 0),
                ('rollout_timeout_count', 'Timeouts (rollout)', 0),
            ):
                if key in losses:
                    lines.append(f"{label + ':':>{pad}} {losses[key]:.{precision}f}")
            if 'rollout_reward_rate' in losses:
                lines.append(f"{'Mean reward rate (/s):':>{pad}} {losses['rollout_reward_rate']:.4f}")
                for name, value in losses.items():
                    if name.startswith('rollout_reward_') and name != 'rollout_reward_rate':
                        label = f"Reward/{name.removeprefix('rollout_reward_')} (/s):"
                        lines.append(f"{label:>{pad}} {value:.4f}")
            for axis in ('vx', 'vy', 'wz'):
                key = f'velocity_weight_used_{axis}'
                if key in losses:
                    label = f'Adaptive {axis} weight (used -> next):'
                    lines.append(f"{label:>{pad}} {losses[key]:.3f} -> "
                                 f"{losses[f'velocity_weight_next_{axis}']:.3f} | "
                                 f"score EMA {losses[f'velocity_score_ema_{axis}']:.3f}")
            if 'velocity_learning_stage' in losses:
                stage = int(losses['velocity_learning_stage'])
                description = 'planar first' if stage == 1 else 'yaw learning'
                label = 'Velocity learning stage:'
                lines.append(f"{label:>{pad}} {stage} ({description}) | "
                             f"positive {int(losses['velocity_planar_positive_updates'])}/20 | "
                             f"regression {int(losses['velocity_planar_regression_updates'])}/10")
            if 'velocity_reward_ratio_target' in losses:
                label = 'Velocity reward planar / yaw (/s):'
                lines.append(f"{label:>{pad}} {losses['velocity_reward_planar_mean']:.4f} / "
                             f"{losses['velocity_reward_yaw_mean']:.4f}")
                if 'velocity_reward_forward_mean' in losses:
                    label = 'Velocity reward vx / vy (/s):'
                    lines.append(f"{label:>{pad}} {losses['velocity_reward_forward_mean']:.4f} / "
                                 f"{losses['velocity_reward_lateral_mean']:.4f}")
                    xy_ratio = (f"{losses['velocity_forward_to_lateral_reward_ratio']:.3f}"
                                if losses['velocity_planar_reward_ratio_valid'] else '--')
                    xy_limited = ' | weight bound active' if losses['velocity_planar_target_limited'] else ''
                    label = 'Vx/vy reward (actual / target):'
                    lines.append(f"{label:>{pad}} {xy_ratio} / 1.000{xy_limited}")
                ratio = (f"{losses['velocity_yaw_to_planar_reward_ratio']:.3f}"
                         if losses['velocity_reward_ratio_valid'] else '--')
                limited = ' | weight bound active' if losses['velocity_yaw_target_limited'] else ''
                label = 'Yaw/planar reward (actual / target):'
                lines.append(f"{label:>{pad}} {ratio} / {losses['velocity_reward_ratio_target']:.3f}{limited}")
            done = values['it'] + 1 - values['start_it']
            remaining = values['total_it'] - values['start_it'] - done
            lines.extend([
                '-' * width,
                f"{'Iteration time:':>{pad}} {collect + learn:.2f}s",
                f"{'Time elapsed:':>{pad}} {timedelta(seconds=int(logger.tot_time))}",
                f"{'ETA:':>{pad}} {timedelta(seconds=int(logger.tot_time / done * remaining))}",
            ])
            print('\n'.join(lines))
            return result
        collect, learn = values["collect_time"], values["learn_time"]
        size = logger.cfg["num_steps_per_env"] * logger.num_envs * logger.gpu_world_size
        fps = int(size / (collect + learn))
        run = logger.cfg.get("run_name") or "training"
        lines = [
            (f"[{run}] {values['it']}/{values['total_it']} | "
             f"{fps:,} steps/s | collect {collect:.3f}s learn {learn:.3f}s")
        ]
        if 'student_ppo_critic_warmup' in losses:
            phase = 'Critic warmup' if losses['student_ppo_critic_warmup'] else 'PPO'
            lines.append(f"student phase {phase} | updates {losses['student_ppo_completed_updates']:.0f}")
            if 'student_ppo_reference_coefficient' in losses:
                lines.append(f"student reference coef {losses['student_ppo_reference_coefficient']:.4f} | "
                             f"KL biped {losses.get('student_ppo_reference_task0_kl', 0):.4f} "
                             f"snake {losses.get('student_ppo_reference_task2_kl', 0):.4f}")
        if "distillation_warmup" in losses:
            phase = "warmup" if losses["distillation_warmup"] else "PPO"
            lines.append(
                f"phase {phase} | coef {losses.get('distillation_coefficient', 0):.4f} | "
                f"imitation {losses.get('distillation', 0):.4f}"
            )
        if 'guidance' in losses:
            biped = f"biped={losses['guidance_biped_motion_cached']:.0f} " if 'guidance_biped_motion_cached' in losses else ''
            counts = ' '.join(
                f"{label}={losses.get(f'guidance_{task}_cached', 0):.0f}"
                for task, label in (('descent', 'down'), ('snake_motion', 'snake'), ('ascent', 'up'))
            )
            lines.append(f"guide {losses['guidance']:.4f} | labels {biped}{counts}")
        if 'dagger_teacher_beta' in losses:
            lines.append(
                f"DAgger teacher {losses['dagger_teacher_fraction']:.0%} "
                f"(beta {losses['dagger_teacher_beta']:.2f}) | "
                f"updates {losses['dagger_optimizer_steps']:.0f} | "
                f"rows {losses['dagger_supervised_rows']:.0f} | "
                f"anchor {losses['dagger_anchor_samples']:.0f}"
            )
            for task in ('biped', 'descending', 'snake', 'ascending'):
                prefix = f'task_{task}_'
                rate = losses.get(prefix + 'reward_rate')
                duration = losses.get(prefix + 'mean_episode_s')
                rate_text = '--' if rate is None else f'{rate:.3f}'
                duration_text = '--' if duration is None else f'{duration:.2f}s'
                lines.append(
                    f"{task}: mse {losses[f'dagger_{task}_mse']:.4f} | "
                    f"reward/s {rate_text} | episode {duration_text} | "
                    f"success {losses.get(prefix + 'success_count', 0):.0f} | "
                    f"falls {losses.get(prefix + 'biped_fallen_count', 0):.0f} | "
                    f"unsafe {losses.get(prefix + 'unsafe_count', 0):.0f} | "
                    f"timeout {losses.get(prefix + 'time_out_count', 0):.0f}"
                )
            print('\n'.join(lines))
            return result
        reward = f"{statistics.mean(logger.rewbuffer):.2f}" if logger.rewbuffer else "--"
        length = f"{statistics.mean(logger.lenbuffer):.1f}" if logger.lenbuffer else "--"
        if 'rollout_reward_rate' in losses:
            lines.append(
                f"reward_rate {losses['rollout_reward_rate']:.3f}/s | "
                f"episode_age {losses['rollout_episode_age_s']:.2f}s | "
                f"resets {losses['rollout_reset_count']:.0f} | "
                f"timeouts {losses.get('rollout_timeout_count', 0):.0f} | "
                f"policy_std {values['action_std'].mean().item():.3f}"
            )
            lines.append(' | '.join(
                f"{name.removeprefix('rollout_reward_')} {value:.3f}/s"
                for name, value in losses.items()
                if name.startswith('rollout_reward_') and name != 'rollout_reward_rate'
            ))
            if all(f'rollout_snake_{axis}_rmse' in losses for axis in ('vx', 'vy', 'wz')):
                lines.append(
                    f"velocity_rmse vx {losses['rollout_snake_vx_rmse']:.3f} m/s | "
                    f"vy {losses['rollout_snake_vy_rmse']:.3f} m/s | "
                    f"wz {losses['rollout_snake_wz_rmse']:.3f} rad/s"
                )
        else:
            lines.append(f"reward {reward} | ep_len {length} | std {values['action_std'].mean().item():.3f}")
        counts = [f"{key[8:-11]}={value:.0f}" for key, value in losses.items()
                  if key.startswith("teacher_") and key.endswith("_candidates")]
        if counts:
            lines.append("teachers " + " ".join(counts))
        blocked = [key.removeprefix("warmup_blocked_") for key, value in losses.items()
                   if key.startswith("warmup_blocked_") and value]
        if "warmup_validation_passes" in losses:
            lines.append(f"gate blocked={','.join(blocked) or 'none'} | "
                         f"passes {losses['warmup_validation_passes']:.0f}")
        if 'snake_omni_scenarios_passed' in losses:
            lines.append(f"snake omni {losses['snake_omni_scenarios_passed']:.0f}/9 | qualified={bool(losses.get('snake_omni_passed', 0))}")
        elif 'biped_stop_success' in losses and 'snake_stop_success' in losses:
            skills = ('stop', 'forward', 'backward', 'left', 'right', 'yaw_left', 'yaw_right',
                      'mixed_positive', 'mixed_negative')
            groups = []
            for mode in ('biped', 'snake'):
                passed = sum(
                    losses.get(f'{mode}_{skill}_success', 0.) >= .75
                    and all(losses.get(f'{mode}_{skill}_yaw_{yaw}_success', 0.) >= .75 for yaw in (0, 90, 180, 270))
                    for skill in skills
                )
                groups.append(f'{mode}={passed}/9')
            transitions = sum(
                losses.get(f'{name}_success', 0.) >= .75
                and all(losses.get(f'{name}_yaw_{yaw}_success', 0.) >= .75 for yaw in (0, 90, 180, 270))
                for name in ('ascent', 'descent')
            )
            lines.append('validation ' + ' '.join(groups) + f' transform={transitions}/2')
        else:
            if 'validation_biped_moving_success' in losses:
                lines.append(f"student validation biped {losses['validation_biped_moving_success']:.0%} "
                             f"snake {losses['validation_snake_moving_success']:.0%} | "
                             f"down/up handoff {losses['validation_descent_handoff_fraction']:.0%}/"
                             f"{losses['validation_ascent_handoff_fraction']:.0%}")
            successes = [f"{key.removesuffix('_success')}={value:.0%}" for key, value in losses.items()
                         if key.endswith("_success") and not key.startswith(("robust_", "validation_"))]
            if successes:
                lines.append("validation " + " ".join(successes))
        print("\n".join(lines))
        return result

    logger.log = compact_log


def install_site_snake_rollout_logging(logger: Any, env: Any) -> None:
    """Measure each collected step, so statistics do not require a collision reset.

    MJLab's _step_reward contains weighted rates before dt scaling, even after
    episode sums are cleared. Keep reductions on-device until the iteration log.
    """
    process = logger.process_env_step
    original_log = logger.log
    signature = inspect.signature(original_log)
    rates = counts = None
    steps = 0

    @wraps(process)
    def process_step(*args: Any, **kwargs: Any) -> Any:
        nonlocal rates, counts, steps
        result = process(*args, **kwargs)
        if logger.writer is None:
            return result
        # process_env_step accepts rewards, dones, extras, optional intrinsic rewards.
        dones = args[1] if len(args) > 1 else kwargs['dones']
        extras = args[2] if len(args) > 2 else kwargs['extras']
        current = env.reward_manager._step_reward.detach().mean(0)
        if rates is None:
            rates = torch.zeros_like(current)
            counts = current.new_zeros(2)  # All resets, then timeout resets.
        assert rates is not None and counts is not None
        rates.add_(current)
        ended = dones > 0
        time_outs = extras.get('time_outs')
        if time_outs is None:
            time_outs = getattr(getattr(env, 'termination_manager', None), 'time_outs', None)
        timed_out = (ended & time_outs.to(device=ended.device).reshape_as(ended).bool()
                     if time_outs is not None else torch.zeros_like(ended))
        counts.add_(torch.stack((ended.sum(), timed_out.sum())))
        steps += 1
        return result

    @wraps(original_log)
    @torch.inference_mode()
    def log_rollout(*args: Any, **kwargs: Any) -> Any:
        nonlocal steps
        if not steps or logger.writer is None:
            return original_log(*args, **kwargs)
        assert rates is not None and counts is not None
        bound = signature.bind(*args, **kwargs)
        metrics = dict(bound.arguments['loss_dict'])
        values = torch.cat((rates / steps, counts,
                            (logger.cur_episode_length.mean() * env.step_dt).reshape(1))).cpu().tolist()
        for name, value in zip(env.reward_manager.active_terms, values[:-3], strict=True):
            metrics[f'rollout_reward_{name}'] = value
        metrics['rollout_reward_rate'] = sum(values[:-3])
        metrics['rollout_reset_count'] = values[-3]
        metrics['rollout_timeout_count'] = values[-2]
        metrics['rollout_termination_count'] = values[-3] - values[-2]
        metrics['rollout_episode_age_s'] = values[-1]
        bound.arguments['loss_dict'] = metrics
        result = original_log(*bound.args, **bound.kwargs)
        rates.zero_()
        counts.zero_()
        steps = 0
        return result

    logger.process_env_step = process_step
    logger.log = log_rollout

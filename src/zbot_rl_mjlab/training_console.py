"""Compact multi-teacher progress without changing the upstream metric writer."""

from __future__ import annotations

import inspect
import statistics
import sys
from contextlib import redirect_stdout
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
            if 'rollout_reward_rate' in losses:
                lines.append(f"{'Mean reward rate (/s):':>{pad}} {losses['rollout_reward_rate']:.4f}")
                for name, value in losses.items():
                    if name.startswith('rollout_reward_') and name != 'rollout_reward_rate':
                        label = f"Reward/{name.removeprefix('rollout_reward_')} (/s):"
                        lines.append(f"{label:>{pad}} {value:.4f}")
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
        reward = f"{statistics.mean(logger.rewbuffer):.2f}" if logger.rewbuffer else "--"
        length = f"{statistics.mean(logger.lenbuffer):.1f}" if logger.lenbuffer else "--"
        if 'rollout_reward_rate' in losses:
            lines.append(
                f"reward_rate {losses['rollout_reward_rate']:.3f}/s | "
                f"episode_age {losses['rollout_episode_age_s']:.2f}s | "
                f"resets {losses['rollout_reset_count']:.0f} | "
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
            successes = [f"{key.removesuffix('_success')}={value:.0%}" for key, value in losses.items()
                         if key.endswith("_success") and not key.startswith("robust_")]
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
    rates = collisions = None
    steps = 0

    @wraps(process)
    def process_step(*args: Any, **kwargs: Any) -> Any:
        nonlocal rates, collisions, steps
        result = process(*args, **kwargs)
        if logger.writer is None:
            return result
        # process_env_step accepts rewards, dones, extras, optional intrinsic rewards.
        dones = args[1] if len(args) > 1 else kwargs['dones']
        current = env.reward_manager._step_reward.detach().mean(0)
        if rates is None:
            rates = torch.zeros_like(current)
            collisions = current.new_zeros(())
        assert rates is not None and collisions is not None
        rates.add_(current)
        collisions.add_((dones > 0).sum())
        steps += 1
        return result

    @wraps(original_log)
    @torch.inference_mode()
    def log_rollout(*args: Any, **kwargs: Any) -> Any:
        nonlocal steps
        if not steps or logger.writer is None:
            return original_log(*args, **kwargs)
        assert rates is not None and collisions is not None
        bound = signature.bind(*args, **kwargs)
        metrics = dict(bound.arguments['loss_dict'])
        values = torch.cat((rates / steps, collisions.reshape(1),
                            (logger.cur_episode_length.mean() * env.step_dt).reshape(1))).cpu().tolist()
        for name, value in zip(env.reward_manager.active_terms, values[:-2], strict=True):
            metrics[f'rollout_reward_{name}'] = value
        metrics['rollout_reward_rate'] = sum(values[:-2])
        metrics['rollout_reset_count'] = values[-2]
        metrics['rollout_episode_age_s'] = values[-1]
        bound.arguments['loss_dict'] = metrics
        result = original_log(*bound.args, **bound.kwargs)
        rates.zero_()
        collisions.zero_()
        steps = 0
        return result

    logger.process_env_step = process_step
    logger.log = log_rollout

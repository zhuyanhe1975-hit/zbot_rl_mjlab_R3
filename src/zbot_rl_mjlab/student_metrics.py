"""Whole-rollout student diagnostics; no last-100-episode sampling bias.

Frame reward rates follow the task before each action. Completed episodes and
termination causes follow their first observed task, even if the FSM switches
tasks before completion. Flush clears reporting totals, retaining live episodes.
These are descriptive environment rewards, not the DAgger optimization loss.
"""

import math

import torch

TASK_NAMES = ('biped', 'descending', 'snake', 'ascending')
FAILURE_REASONS = ('biped_fallen', 'unsafe', 'transition_failure')


class StudentTaskMetrics:
    def __init__(self, num_envs: int, device: str | torch.device, step_dt: float):
        if num_envs <= 0 or not math.isfinite(step_dt) or step_dt <= 0:
            raise ValueError('student metrics require positive environment count and finite positive step_dt')
        self.num_envs, self.step_dt = num_envs, step_dt
        # frames, reward, completed episodes, episode return, episode seconds,
        # early failures, successes; all per initial task except frames/reward.
        self.totals = torch.zeros(7, 4, device=device, dtype=torch.float64)
        self.episode_steps = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.episode_returns = torch.zeros(num_envs, device=device, dtype=torch.float64)
        self.initial_tasks = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.reasons = {name: torch.zeros(4, device=device, dtype=torch.float64)
                        for name in (*FAILURE_REASONS, 'skill_complete', 'time_out')}

    @torch.no_grad()
    def observe(self, pre_teacher56: torch.Tensor, rewards: torch.Tensor,
                dones: torch.Tensor, termination_masks: dict[str, torch.Tensor]):
        """Consume the pre-action task and this step's reward/done/reason masks.

        Read termination masks after stepping and before the next environment
        step. Do not pass auto-reset observations as ``pre_teacher56``.
        Overlapping causes count individually, but each early failure counts
        once. Skill completion and timeout alone never count as early failure.
        """
        tasks = pre_teacher56[:, 39].round().long()
        reward = rewards.reshape(self.num_envs).to(self.totals.dtype)
        done = dones.reshape(self.num_envs).bool()
        self.initial_tasks.copy_(torch.where(self.episode_steps == 0, tasks, self.initial_tasks))
        self.episode_steps += 1
        self.episode_returns += reward
        ones = torch.ones_like(reward)
        self.totals[0].scatter_add_(0, tasks, ones)
        self.totals[1].scatter_add_(0, tasks, reward)
        completed = done.to(self.totals.dtype)
        seconds = self.episode_steps.to(self.totals.dtype) * self.step_dt
        self.totals[2].scatter_add_(0, self.initial_tasks, completed)
        self.totals[3].scatter_add_(0, self.initial_tasks, self.episode_returns * completed)
        self.totals[4].scatter_add_(0, self.initial_tasks, seconds * completed)
        failure = torch.zeros_like(done)
        for name, mask in termination_masks.items():
            mask = mask.reshape(self.num_envs).bool() & done
            if name not in self.reasons:
                # Collection runs under inference_mode; counters survive into
                # ordinary optimizer/logging context where flush resets them.
                with torch.inference_mode(False):
                    self.reasons[name] = torch.zeros_like(self.totals[0])
            self.reasons[name].scatter_add_(0, self.initial_tasks, mask.to(self.totals.dtype))
            if name in FAILURE_REASONS:
                failure |= mask
            if name == 'skill_complete':
                self.totals[6].scatter_add_(0, self.initial_tasks, mask.to(self.totals.dtype))
        early = failure & (seconds < 2.)
        self.totals[5].scatter_add_(0, self.initial_tasks, early.to(self.totals.dtype))
        self.episode_steps.masked_fill_(done, 0)
        self.episode_returns.masked_fill_(done, 0.)

    @torch.no_grad()
    def flush(self) -> dict[str, float]:
        """Report all environments, copying reductions to the CPU only once.

        Means without a denominator are omitted; explicit count fields remain
        zero. Live episodes cross rollout boundaries without being truncated.
        """
        live = self.episode_steps > 0
        ongoing = torch.zeros(2, 4, device=self.totals.device, dtype=self.totals.dtype)
        ongoing[0].scatter_add_(0, self.initial_tasks, live.to(self.totals.dtype))
        ongoing[1].scatter_add_(0, self.initial_tasks, self.episode_steps.to(self.totals.dtype) * self.step_dt)
        reason_names = tuple(self.reasons)
        values = torch.cat((self.totals, ongoing, torch.stack(tuple(self.reasons.values())))).cpu().tolist()
        result: dict[str, float] = {}
        total_frames = sum(values[0])
        for task, name in enumerate(TASK_NAMES):
            prefix = f'task_{name}_'
            frames, reward, completed, returns, seconds, early, success, live_count, live_age = (
                row[task] for row in values[:9]
            )
            result.update({prefix + 'frames': frames,
                           prefix + 'completed_episodes': completed,
                           prefix + 'early_failure_count': early,
                           prefix + 'success_count': success,
                           prefix + 'ongoing_episodes': live_count})
            if total_frames:
                result[prefix + 'frame_fraction'] = frames / total_frames
            if frames:
                result[prefix + 'reward_rate'] = reward / (frames * self.step_dt)
            if completed:
                result[prefix + 'mean_episode_return'] = returns / completed
                result[prefix + 'mean_episode_s'] = seconds / completed
            if live_count:
                result[prefix + 'ongoing_mean_age_s'] = live_age / live_count
            for index, reason in enumerate(reason_names, 9):
                result[prefix + reason + '_count'] = values[index][task]
        self.totals.zero_()
        for value in self.reasons.values():
            value.zero_()
        return result

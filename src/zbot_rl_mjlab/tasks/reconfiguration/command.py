"""Episode goal: requested head-tail twist about the docking-interface normal."""
from dataclasses import dataclass

import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

DEGREES = (0, 90, 180, 270)


@dataclass(kw_only=True)
class DockingCommandCfg(CommandTermCfg):
    resampling_time_range: tuple[float, float] = (1e9, 1e9)
    target_yaw_deg: float = 180.
    sample_on_reset: bool = True
    angles_deg: tuple[int, ...] = DEGREES

    def build(self, env):
        return DockingCommand(self, env)


class DockingCommand(CommandTerm):
    def __init__(self, cfg, env):
        if not cfg.angles_deg or len(set(cfg.angles_deg)) != len(cfg.angles_deg) or any(v not in DEGREES for v in cfg.angles_deg):
            raise ValueError('angles_deg must contain unique choices from 0,90,180,270')
        if cfg.target_yaw_deg not in cfg.angles_deg:
            raise ValueError('target_yaw_deg must be an allowed angle')
        if any(int(v/90) not in env.cfg.allowed_dock_quarters for v in cfg.angles_deg):
            raise ValueError('command angles must be physically allowed registrations')
        super().__init__(cfg, env)
        self.quarter = torch.full((env.num_envs,), int(cfg.target_yaw_deg/90), device=env.device, dtype=torch.long)
        self.manual_quarter = torch.full_like(self.quarter, -1)
        self.choices = torch.tensor(cfg.angles_deg, device=env.device, dtype=torch.long) // 90

    @property
    def command(self):
        return torch.nn.functional.one_hot(self.quarter, 4).float()

    def _resample_command(self, env_ids):
        if self.cfg.sample_on_reset:
            self.quarter[env_ids] = self.choices[torch.randint(len(self.choices), (len(env_ids),), device=self.device)]
        else:
            self.quarter[env_ids] = int(self.cfg.target_yaw_deg/90)
        self.quarter[env_ids] = torch.where(self.manual_quarter[env_ids] >= 0, self.manual_quarter[env_ids], self.quarter[env_ids])

    def _update_metrics(self):
        pass

    def _update_command(self, env_ids):
        pass

    def set_target(self, angle_deg, env_ids=None):
        if angle_deg not in self.cfg.angles_deg:
            raise ValueError('target angle must be one of the configured angles_deg')
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        self.quarter[ids] = int(angle_deg/90)
        self.manual_quarter[ids] = int(angle_deg/90)
        # Changing a goal does not change physical bonds or body poses.
        dock = getattr(self._env, '_magnetic_dock', None)
        if dock is not None and getattr(self._env.cfg, 'docking_mode', 'registered_weld') != 'normal_pose':
            for name in ('capture_time', 'hold_time', 'succeeded', 'just_succeeded'):
                getattr(dock, name)[ids] = 0


def target_quarter(env):
    return env.command_manager.get_term('dock').quarter


def target_yaw_deg(env):
    return target_quarter(env).float() * 90

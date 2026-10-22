"""独立蛇形教师：沿蛇形前向推进及原有安全复位边界。"""

import math
from dataclasses import dataclass
from typing import Literal

from ..transformation.env_cfg import TransformationEnvCfg

TASK_ID = "Mjlab-Zbot-6dof-Snake-Omni-Teacher"
SNAKE_JOINT_LIMIT = math.pi / 2
SNAKE_MAX_TARGET_SPEED = 2 * math.pi


@dataclass
class SnakeEnvCfg(TransformationEnvCfg):
    allow_resume_joint_limit_change: bool = False
    observation_reference: Literal['world_x', 'command'] = 'world_x'
    motion_frame: Literal['body_gravity', 'legacy_head'] = 'body_gravity'
    head_gyro_frame: Literal['head', 'world'] = 'head'
    head_body: str = "foot_0"
    motion_site: str = "foot_0_sole"
    tail_site: str = "foot_1_sole"
    head_forward_axis: tuple[float, float, float] = (0., 0., -1.)
    head_up_axis: tuple[float, float, float] = (1., 0., 0.)
    head_tilt_limit: float = math.pi / 3
    head_yaw_limit: float = math.pi / 3
    body_yaw_limit: float = math.pi / 6
    body_yaw_integral_time_scale: float = 1.
    head_yaw_integral_time_scale: float = 1.
    forward_speed_cap: float = .2
    motion_penalty_min_scale: float = 0.25
    motion_penalty_speed_gain: float = 5.
    workspace_radius: float = 15.
    max_contact_bodyweights: float = 8.
    stall_min_speed: float = .01
    stall_progress_fraction: float = .20
    stall_grace_s: float = 1.
    stall_filter_s: float = .5

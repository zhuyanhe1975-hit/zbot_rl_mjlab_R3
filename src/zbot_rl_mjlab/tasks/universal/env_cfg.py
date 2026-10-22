"""统一物理场景下的双足、蛇形与双向变形任务。"""
from dataclasses import dataclass
from typing import Literal

from ...robot import DEFAULT_JOINT_POS
from ..transformation.env_cfg import TransformationEnvCfg

TASK_ID = "Mjlab-Zbot-6dof-Universal-Locomotion"


@dataclass
class UniversalEnvCfg(TransformationEnvCfg):
    omni_site_frame: bool = False
    omni_frame_backend: Literal['eager', 'cuda_graph'] = 'eager'
    motion_site: str = 'foot_0_sole'
    snake_teacher_enabled: bool = False
    straighten_before_stand: bool = False
    standing_joint_pos: tuple[float, ...] = DEFAULT_JOINT_POS
    standing_base_height: float = .264
    workspace_radius: float = 15.

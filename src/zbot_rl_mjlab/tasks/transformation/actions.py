"""六个独立策略动作控制有界关节目标速度。"""

import math
from dataclasses import dataclass, field

import torch
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

from ...robot import JOINT_NAMES


def bounded_target(current_target, action, limits, dt, max_speed=1.0):
    rate = torch.nan_to_num(action).clamp(-1.0, 1.0) * max_speed
    return (current_target + rate * dt).clamp(min=limits[..., 0], max=limits[..., 1])


@dataclass(kw_only=True)
class JointTargetRateActionCfg(JointPositionActionCfg):
    max_target_speed: float = 1.0
    target_limit: float | None = None
    joint_target_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)

    def build(self, env):
        return JointTargetRateAction(self, env)


class JointTargetRateAction(JointPositionAction):
    def __init__(self, cfg: JointTargetRateActionCfg, env):
        super().__init__(cfg, env)
        self.rate_cfg = cfg
        self._named_target_bounds = None
        if cfg.joint_target_bounds:
            self._initialize_target_bounds()
        self._processed_actions.copy_(
            self._entity.data.default_joint_pos[:, self._target_ids]
        )
        if self._named_target_bounds is not None or cfg.target_limit is not None:
            self._processed_actions.copy_(self._clamp_target(self._processed_actions))

    def _initialize_target_bounds(self):
        bounds = self.rate_cfg.joint_target_bounds
        missing = bounds.keys() - set(self._target_names)
        if missing:
            raise ValueError(f"Unknown commanded target joints: {sorted(missing)}")
        named_bounds = self._processed_actions.new_tensor(
            [[-math.inf, math.inf]] * len(self._target_names)
        )
        for name, (low, high) in bounds.items():
            if not math.isfinite(low) or not math.isfinite(high) or low > high:
                raise ValueError(
                    f"Invalid commanded target bounds for {name}: {(low, high)}"
                )
            named_bounds[self._target_names.index(name)] = named_bounds.new_tensor(
                [low, high]
            )
        self._named_target_bounds = named_bounds
        limits = self._target_limits()
        if (limits[..., 0] > limits[..., 1]).any():
            raise ValueError(
                "Commanded target bounds do not intersect joint/target limits"
            )

    def _target_limits(self):
        limits = self._entity.data.joint_pos_limits[:, self._target_ids]
        named_bounds = getattr(self, "_named_target_bounds", None)
        if self.rate_cfg.target_limit is not None:
            if named_bounds is None:
                limits = limits.clamp(
                    -self.rate_cfg.target_limit, self.rate_cfg.target_limit
                )
            else:
                limits = torch.stack(
                    (
                        limits[..., 0].clamp(min=-self.rate_cfg.target_limit),
                        limits[..., 1].clamp(max=self.rate_cfg.target_limit),
                    ),
                    dim=-1,
                )
        if named_bounds is not None:
            limits = torch.stack(
                (
                    torch.maximum(limits[..., 0], named_bounds[:, 0]),
                    torch.minimum(limits[..., 1], named_bounds[:, 1]),
                ),
                dim=-1,
            )
        return limits

    def _clamp_target(self, target, env_ids: slice | torch.Tensor = slice(None)):
        limits = self._target_limits()[env_ids]
        return target.clamp(min=limits[..., 0], max=limits[..., 1])

    @property
    def commanded_target(self):
        """跨控制步保存的目标状态；每环境每关节独立，内容始终是裁剪结果。"""
        return self._processed_actions

    def process_actions(self, actions):
        self._raw_actions.copy_(torch.nan_to_num(actions).clamp(-1.0, 1.0))
        limits = self._target_limits()
        # 积分状态与实际下发目标共用这一份持久缓冲，不保留越界累计量。
        # 下次积分直接从裁剪结果开始，反向请求无需先抵消隐藏积分。
        self._processed_actions.copy_(
            bounded_target(
                self._processed_actions,
                self._raw_actions,
                limits,
                self._env.step_dt,
                self.rate_cfg.max_target_speed,
            )
        )

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        super().reset(ids)
        self._processed_actions[ids] = self._entity.data.joint_pos[ids][
            :, self._target_ids
        ]
        if getattr(self, "_named_target_bounds", None) is not None or self.rate_cfg.target_limit is not None:
            self._processed_actions[ids] = self._clamp_target(
                self._processed_actions[ids], ids
            )


def expand_mirror(values):
    return torch.cat((values, -values.flip(-1)), dim=-1)


def project_mirror(values):
    return (values[..., :3] - values[..., 3:].flip(-1)) * 0.5


def mirrored_last_action(env):
    # 保留预训练的 30 维观测结构，包括六个上一动作槽位。
    return expand_mirror(env.action_manager.get_term("joint_pos").raw_action)


@dataclass(kw_only=True)
class MirroredJointTargetRateActionCfg(JointTargetRateActionCfg):
    def build(self, env):
        return MirroredJointTargetRateAction(self, env)


class MirroredJointTargetRateAction(JointTargetRateAction):
    def __init__(self, cfg, env):
        self._mirror_target_limits = None
        # 先让父类分配六个执行器目标槽位，再减少原始动作数。
        super().__init__(cfg, env)
        if tuple(self._target_names) != JOINT_NAMES:
            raise ValueError("Mirror actions require joint1..joint6 in model order")
        self._action_dim = 3
        self._raw_actions = torch.zeros((env.num_envs, 3), device=env.device)
        self._paired_limits(validate=True)
        self._processed_actions.copy_(self._clamp_target(self._processed_actions))
        self._previous_commanded_target = self.commanded_target.clone()

    @property
    def commanded_target_velocity(self):
        return (self.commanded_target - self._previous_commanded_target) / self._env.step_dt

    def _paired_limits(self, validate=False):
        # 限制在构造时配置，采样过程中不随机化。
        if self._mirror_target_limits is None:
            limits = self._target_limits()
            left, right = limits[:, :3], limits[:, 3:].flip(1)
            self._mirror_target_limits = torch.stack((torch.maximum(left[..., 0], -right[..., 1]), torch.minimum(left[..., 1], -right[..., 0])), dim=-1)
        paired = self._mirror_target_limits
        if validate and (paired[..., 0] > paired[..., 1]).any():
            raise ValueError("Joint limits have no common mirrored target interval")
        return paired

    def _clamp_target(self, target, env_ids=slice(None)):
        limits = self._paired_limits()[env_ids]
        coordinates = project_mirror(target).clamp(min=limits[..., 0], max=limits[..., 1])
        return expand_mirror(coordinates)

    def process_actions(self, actions):
        if actions.shape[-1] != 3:
            raise ValueError("Mirrored policy must provide exactly three actions")
        self._previous_commanded_target.copy_(self.commanded_target)
        self._raw_actions.copy_(torch.nan_to_num(actions).clamp(-1, 1))
        coordinates = bounded_target(project_mirror(self.commanded_target), self._raw_actions, self._paired_limits(), self._env.step_dt, self.rate_cfg.max_target_speed)
        self._processed_actions.copy_(expand_mirror(coordinates))

    def reset(self, env_ids=None):
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self._processed_actions[ids] = self._clamp_target(self._processed_actions[ids], ids)
        self._previous_commanded_target[ids] = self.commanded_target[ids]


def migrate_beta_mirror_head(actor_state):
    """将原生六动作 Beta 输出头合并为三个镜像 Beta 变量。

    对原始 alpha-left / beta-right 和 beta-left / alpha-right 行取平均。
    这是线性输出头投影，不能精确保留原有非线性 Beta 均值或其非对称行为。
    主干网络和归一化保持不变。
    """
    key = "mlp.4.weight"
    bias = "mlp.4.bias"
    if key not in actor_state or bias not in actor_state:
        raise ValueError("Expected the pretrained 256/128 MLP Beta action head")
    weight, offset = actor_state[key], actor_state[bias]
    if weight.shape[0] == 6 and offset.shape[0] == 6:
        return actor_state, False
    if weight.shape[0] != 12 or offset.shape[0] != 12:
        raise ValueError("Expected a six-action or three-action Beta checkpoint")
    left = torch.tensor([0, 1, 2, 6, 7, 8], device=weight.device)
    opposite = torch.tensor([11, 10, 9, 5, 4, 3], device=weight.device)
    migrated = dict(actor_state)
    migrated[key] = (weight[left] + weight[opposite]) * 0.5
    migrated[bias] = (offset[left] + offset[opposite]) * 0.5
    return migrated, True

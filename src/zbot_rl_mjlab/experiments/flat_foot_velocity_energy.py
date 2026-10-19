"""采用速度与能耗奖励、着地及站立时保持全足底支撑的行走实验。

替换旧 foot_nonparallel_ground 项，其 1-|nz| 形式约为 theta^4，在倾角小于 10 度时
几乎没有梯度；新代价基于 sin^2，在承重、触地及触地前的三个阶段生效。

三个权重按课程变化。若从第 0 次迭代就使用完整权重，总奖励为负，策略会学会快速
倒下（回合约 30 步）且无法恢复；固定小权重（-0.1 / -0.4 / -0.1）虽能行走，
足底却仍像此前一样倾斜。因此先采用小权重，形成步态后再逐渐增加。
"""

from mjlab.envs.mdp import reward_curriculum
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.reward_manager import RewardTermCfg

from ..tasks.walking import mdp
from .base import RewardEdit
from .walking import VelocityEnergyTracking

FLAT_FOOT_VELOCITY_ENERGY_TASK_ID = "Mjlab-Zbot-6dof-Flat-Foot-Velocity-Energy-Walking"

# 足底倾斜按 sin^2(倾角) / SIGMA^2 惩罚；0.05 约对应 3 度。
FLAT_SIGMA = 0.05
PRE_LANDING_HEIGHT = 0.03  # 摆动足低于此足底高度（m）时必须摆平

# env.common_step_counter 每个向量环境步递增一次，每次 PPO 迭代有
# num_steps_per_env 个这样的步，因此“从第 N 次迭代起”的阶段始于 N * 此值。
STEPS_PER_ITERATION = 24
# （起始迭代、支撑期权重、触地权重、着地前权重）。最后一行即
# 从第 0 次迭代直接启用时导致训练崩溃的权重。
WEIGHT_STAGES = (
    (0, -0.05, -0.2, -0.05),
    (150, -0.15, -0.6, -0.15),
    (300, -0.3, -1.2, -0.3),
    (400, -0.5, -2.0, -0.5),
)
_TERMS = ("foot_flat_stance", "foot_flat_touchdown", "foot_flat_pre_landing")


class FlatFootVelocityEnergy(VelocityEnergyTracking):
    task_id = FLAT_FOOT_VELOCITY_ENERGY_TASK_ID
    experiment_name = "zbot_flat_foot_velocity_energy"
    weight_stages = WEIGHT_STAGES
    """课程各行依次为（起始迭代、支撑期权重、触地权重、着地前权重）。"""

    def reward_edit(self):
        first = self.weight_stages[0]
        params = {"sigma": FLAT_SIGMA}
        return RewardEdit(
            remove=("foot_nonparallel_ground",),
            terms={
                "foot_flat_stance": RewardTermCfg(
                    func=mdp.stance_flat_penalty, weight=first[1], params=params
                ),
                "foot_flat_touchdown": RewardTermCfg(
                    func=mdp.touchdown_flat_penalty, weight=first[2], params=params
                ),
                "foot_flat_pre_landing": RewardTermCfg(
                    func=mdp.pre_landing_flat_penalty,
                    weight=first[3],
                    params={**params, "height": PRE_LANDING_HEIGHT},
                ),
            },
        )

    def edit(self, cfg, play):
        for column, name in enumerate(_TERMS, start=1):
            cfg.curriculum[f"{name}_weight"] = CurriculumTermCfg(
                func=reward_curriculum,
                params={
                    "reward_name": name,
                    "stages": [
                        {"step": row[0] * STEPS_PER_ITERATION, "weight": row[column]}
                        for row in self.weight_stages
                    ],
                },
            )

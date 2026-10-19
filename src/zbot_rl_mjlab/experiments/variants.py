"""行走实验的扰动与实机部署变体。

部署变体限制策略观测，加入噪声、控制延迟和物理参数随机化；平足变体沿用
父类的奖励课程。回放时关闭观测噪声和控制延迟。
"""

from ..tasks.walking.env_cfg import DISTURBED_WALKING_TASK_ID
from .addons import DEPLOYMENT, DISTURBED
from .flat_foot_velocity_energy import FlatFootVelocityEnergy
from .walking import FlatWalking, VelocityEnergyTracking

VELOCITY_ENERGY_DISTURBED_TASK_ID = "Mjlab-Zbot-6dof-Velocity-Energy-Disturbed-Walking"
FLAT_FOOT_DISTURBED_TASK_ID = "Mjlab-Zbot-6dof-Flat-Foot-Velocity-Energy-Disturbed-Walking"


class DisturbedWalking(FlatWalking):
    task_id = DISTURBED_WALKING_TASK_ID
    experiment_name = "zbot_disturbed_walking"
    addons = DISTURBED


class VelocityEnergyDisturbed(VelocityEnergyTracking):
    task_id = VELOCITY_ENERGY_DISTURBED_TASK_ID
    experiment_name = "zbot_velocity_energy_disturbed"
    addons = DISTURBED


class FlatFootDisturbed(FlatFootVelocityEnergy):
    task_id = FLAT_FOOT_DISTURBED_TASK_ID
    experiment_name = "zbot_flat_foot_velocity_energy_disturbed"
    addons = DISTURBED


DEPLOY_WALKING_TASK_ID = "Mjlab-Zbot-6dof-Deploy-Walking"
DEPLOY_FLAT_FOOT_TASK_ID = "Mjlab-Zbot-6dof-Deploy-Flat-Foot-Walking"

# 将平足惩罚课程拉伸至 1500 次迭代（迭代、支撑期、触地、着地前权重）。
# 最后一行为完整惩罚。
DEPLOY_WEIGHT_STAGES = (
    (0, -0.05, -0.2, -0.05),
    (300, -0.15, -0.6, -0.15),
    (600, -0.3, -1.2, -0.3),
    (800, -0.5, -2.0, -0.5),
)
MAX_ITERATIONS = 1500


class DeployWalking(VelocityEnergyTracking):
    task_id = DEPLOY_WALKING_TASK_ID
    experiment_name = "zbot_deploy_walking"
    max_iterations = MAX_ITERATIONS
    addons = DEPLOYMENT


class DeployFlatFoot(FlatFootVelocityEnergy):
    task_id = DEPLOY_FLAT_FOOT_TASK_ID
    experiment_name = "zbot_deploy_flat_foot"
    max_iterations = MAX_ITERATIONS
    addons = DEPLOYMENT
    weight_stages = DEPLOY_WEIGHT_STAGES

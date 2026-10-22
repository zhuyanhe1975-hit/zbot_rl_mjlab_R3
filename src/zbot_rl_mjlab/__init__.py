"""mjlab 的 Zbot 任务；导入本包会注册全部实验。

每个实验都是 ``experiments/`` 中继承父实验的类，仅声明自身改动
（参见 ``experiments/base.py``）。添加实验时，定义该类并在此处列出。
"""

from .experiments.bidirectional import SnakeBipedBidirectional
from .experiments.flat_foot_velocity_energy import FlatFootVelocityEnergy
from .experiments.omni_teacher import (
    SnakeLegOmniGroupedTeacher,
    SnakeLegOmniGuidedTeacher,
    SnakeLegOmniTaskRoutedTeacher,
    SnakeLegOmniTaskTeacher,
    SnakeLegOmniTeacher,
    SnakeLegOmniWideBipedTeacher,
)
from .experiments.site_snake import (
    SnakeBaseAdaptiveTeacher,
    SnakeSiteAdaptiveTeacher,
    SnakeSiteForwardRelaxedTeacher,
    SnakeSiteForwardTeacher,
    SnakeSiteOmniTeacher,
)
from .experiments.snake import SnakeOmniTeacher
from .experiments.transformation import (
    BipedToSnake,
    SnakeToBiped,
    SnakeToBipedArc,
    SnakeToBipedBeta,
    SnakeToBipedRobust,
    SnakeToBipedTeacher,
)
from .experiments.universal import (
    DeployUniversalLocomotion,
    SnakeLegDeploy,
    SnakeLegOmniDeploy,
    UniversalLocomotion,
)
from .experiments.variants import (
    DeployFlatFoot,
    DeployWalking,
    DisturbedWalking,
    FlatFootDisturbed,
    VelocityEnergyDisturbed,
)
from .experiments.walking import (
    FlatWalking,
    OmniVelocityEnergy,
    PeriodicStepping,
    VelocityEnergyTracking,
    VelocityTracking,
)
from .experiments.walking_tuning import WalkingTuning

EXPERIMENTS = (
    PeriodicStepping,
    FlatWalking,
    DisturbedWalking,
    VelocityTracking,
    VelocityEnergyTracking,
    VelocityEnergyDisturbed,
    OmniVelocityEnergy,
    FlatFootVelocityEnergy,
    FlatFootDisturbed,
    DeployWalking,
    DeployFlatFoot,
    WalkingTuning,
    BipedToSnake,
    SnakeToBiped,
    SnakeToBipedBeta,
    SnakeToBipedArc,
    SnakeToBipedRobust,
    SnakeToBipedTeacher,
    SnakeBipedBidirectional,
    UniversalLocomotion,
    DeployUniversalLocomotion,
    SnakeLegDeploy,
    SnakeLegOmniDeploy,
    SnakeLegOmniTeacher,
    SnakeLegOmniWideBipedTeacher,
    SnakeLegOmniGroupedTeacher,
    SnakeLegOmniTaskTeacher,
    SnakeLegOmniTaskRoutedTeacher,
    SnakeLegOmniGuidedTeacher,
    SnakeOmniTeacher,
    SnakeSiteOmniTeacher,
    SnakeSiteAdaptiveTeacher,
    SnakeBaseAdaptiveTeacher,
    SnakeSiteForwardTeacher,
    SnakeSiteForwardRelaxedTeacher,
)

for experiment in EXPERIMENTS:
    experiment.card_name = experiment.experiment_name.removeprefix('zbot_')
    experiment().register()

# 仅兼容原始归档 YAML 中的三个项目回调模块名，不恢复旧实验导入接口。
# 不修改训练配置原件，也不在任务层注入实验类。
import sys as _sys

from .tasks.walking import mdp as _walking_mdp

_sys.modules.setdefault(f"{__name__}.mdp", _walking_mdp)
_sys.modules.setdefault(f"{__name__}.experiments.velocity_tracking", _walking_mdp)
_sys.modules.setdefault(f"{__name__}.experiments.omni_velocity_energy", _walking_mdp)
del _walking_mdp, _sys

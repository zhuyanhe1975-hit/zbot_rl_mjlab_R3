"""中性实验模板：创建任务配置、依次应用配方与扩展，再统一注册。

每个类仅通过自身的 reward_edit/edit 声明增量，基类按 MRO 自动执行一次，
钩子内部不再调用 super().edit()。任务工厂负责环境行为，实验负责训练方案。
"""

from dataclasses import dataclass, field

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.registry import register_mjlab_task


@dataclass
class RewardEdit:
    """实验对继承奖励所作的改动，按以下顺序应用。"""

    remove: tuple[str, ...] = ()
    """需要删除的奖励名称。"""
    weights: dict[str, float] = field(default_factory=dict)
    """继承奖励的新权重。"""
    params: dict[str, dict] = field(default_factory=dict)
    """合并到继承奖励参数中的关键字参数。"""
    terms: dict[str, RewardTermCfg] = field(default_factory=dict)
    """需要添加的奖励；名称已存在时原位替换。"""

    def apply(self, rewards):
        for name in self.remove:
            self._require(rewards, name, "remove")
            del rewards[name]
        for name, weight in self.weights.items():
            self._require(rewards, name, "re-weight")
            rewards[name].weight = weight
        for name, params in self.params.items():
            self._require(rewards, name, "re-parameterize")
            rewards[name].params.update(params)
        rewards.update(self.terms)

    @staticmethod
    def _require(rewards, name, action):
        if name not in rewards:
            raise KeyError(
                f"cannot {action} reward {name!r}: not inherited (have {sorted(rewards)})"
            )


class Experiment:
    """一个已注册训练任务：公共 Zbot 配置叠加子类改动。"""

    task_id: str
    experiment_name: str
    train_envs: int | None = None
    """训练并行环境数（回放始终为 1）；None 表示沿用 env_cfg 的设置。"""
    max_iterations: int | None = None
    """PPO 迭代数；None 表示沿用公共默认值。"""
    runner_cls: type | None = None
    card_name: str | None = None
    """注册时选用的训练运行器；None 表示使用 MJLab 默认值。"""
    addons: tuple = ()
    """本类在父类基础上添加的可选扩展（``addons.py``）。"""

    def collected_addons(self):
        """本类及父类的扩展，每种类型保留一个，并按固定顺序排列。

        子类声明与父类同类型的扩展时，将替换父类的对应扩展。
        """
        chosen = {}
        for cls in reversed(type(self).__mro__):
            for addon in cls.__dict__.get("addons", ()):
                chosen[type(addon)] = addon
        return sorted(chosen.values(), key=lambda addon: addon.order)

    def create_env(self, play=False) -> ManagerBasedRlEnvCfg:
        """由任务族提供正确配置类型的基础环境。"""
        raise NotImplementedError("实验必须提供任务环境工厂")

    def configuration_card(self):
        """Read the registered card afresh; dynamic CLI variants inherit its values."""
        if self.card_name is None:
            return None
        from .cards import load_card

        card = load_card(self.card_name)
        owner = next(cls for cls in type(self).__mro__ if 'card_name' in cls.__dict__)
        if card.get('task_id') != owner.task_id:
            raise ValueError(f'{self.card_name}: card task_id does not match {owner.task_id}')
        return card

    def build_env(self, play=False) -> ManagerBasedRlEnvCfg:
        card = self.configuration_card()
        cfg = self.create_env(play=play)
        if self.train_envs is not None and not play:
            cfg.scene.num_envs = self.train_envs
        # 先应用基类：每个类仅执行自身钩子，不执行继承钩子。
        for cls in reversed(type(self).__mro__):
            reward_edit = cls.__dict__.get("reward_edit")
            if reward_edit is not None:
                reward_edit(self).apply(cfg.rewards)
            edit = cls.__dict__.get("edit")
            if edit is not None:
                edit(self, cfg, play)
        # 最后应用扩展，使其作用于完整构建的实验。
        addons = self.collected_addons()
        if card is not None:
            from .cards import configure_addons

            overrides = ()
            if 'card_name' not in type(self).__dict__:
                overrides = tuple(type(addon) for addon in type(self).__dict__.get('addons', ()))
            addons = configure_addons(addons, card, overrides=overrides)
        for addon in addons:
            addon.apply(cfg, play)
        if card is not None:
            from .cards import apply_environment

            apply_environment(cfg, card, play=play)
        if not play and 'train_envs' in self.__dict__ and self.train_envs is not None:
            cfg.scene.num_envs = self.train_envs
        return cfg

    def build_ppo(self):
        cfg = self._build_ppo()
        card = self.configuration_card()
        if card is not None:
            from .cards import apply_agent

            apply_agent(cfg, card)
        if 'max_iterations' in self.__dict__ and self.max_iterations is not None:
            cfg.max_iterations = self.max_iterations
        return cfg

    def _build_ppo(self):
        from zbot_rl_mjlab.experiments.common import ppo_cfg

        cfg = ppo_cfg()
        cfg.experiment_name = self.experiment_name
        if self.max_iterations is not None:
            cfg.max_iterations = self.max_iterations
        return cfg

    def register(self):
        register_mjlab_task(
            task_id=self.task_id,
            env_cfg=self.build_env(),
            play_env_cfg=self.build_env(play=True),
            rl_cfg=self.build_ppo(),
            runner_cls=self.runner_cls,
        )


class WalkingExperiment(Experiment):
    """行走任务族仅负责创建基础环境和设置指令步频。"""

    step_frequency: tuple[float, float] | None = None

    def create_env(self, play=False):
        from zbot_rl_mjlab.experiments.walking import env_cfg

        cfg = env_cfg(play=play)
        if self.step_frequency is not None:
            cfg.step_frequency_min, cfg.step_frequency_max = self.step_frequency
        return cfg


def variant(base, task_id, experiment_name, *addons, **attributes):
    """新实验：在 ``base`` 上叠加扩展，并使用独立的任务 ID 和实验名。

    无需新建类或文件，即可将任意实验改为受扰动或面向部署的实验；
    ``attributes`` 可覆盖 ``max_iterations`` 等类属性。

        variant(OmniVelocityEnergy, "Mjlab-Zbot-6dof-Omni-Disturbed-Walking",
                "zbot_omni_disturbed", *DISTURBED)
    """
    parts = "+".join(type(addon).__name__ for addon in addons)
    return type(
        f"{base.__name__}+{parts}" if parts else base.__name__,
        (base,),
        {
            "task_id": task_id,
            "experiment_name": experiment_name,
            "addons": tuple(addons),
            **attributes,
        },
    )

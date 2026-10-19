"""共用 PPO 默认值、奖励热更新与检查点恢复运行器。"""

import hashlib
import json
import math
import tempfile
from pathlib import Path

import torch
from mjlab.rl import (
    MjlabOnPolicyRunner,
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
    RslRlVecEnvWrapper,
)

from .tasks.transformation.actions import migrate_beta_mirror_head
from .tasks.transformation.robust import (
    RobustSnakeToBipedEnvCfg,
    apply_fixed_standing_weights,
)
from .tasks.transformation.std_projection import ActionStdProjection


def ppo_cfg():
    actor = RslRlModelCfg(
        hidden_dims=(256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    )
    critic = RslRlModelCfg(
        hidden_dims=(256, 128), activation="elu", obs_normalization=True
    )
    return RslRlOnPolicyRunnerCfg(
        actor=actor,
        critic=critic,
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="zbot_periodic_stepping",
        num_steps_per_env=24,
        max_iterations=1000,
        save_interval=50,
        logger="tensorboard",
    )


CONFIG_SIGNS = {
    "first_stage_initial_pose_weight": -1,
    "first_stage_joint_weight": 1,
    "first_stage_prepared_joint_weight": 1,
    "standing_stage_joint_weight": 1,
    "standing_stage_height_weight": 1,
    "standing_stage_velocity_weight": 1,
    "base_support_distance_weight": -1,
    "base_gravity_alignment_weight": -1,
    "survival_weight": 1,
    "end_joint_flip_weight": -1,
    "first_stage_roll_rate_weight": -1,
    "feet_contact_weight": 1,
}
REWARD_SIGNS = {
    "sole_ground_alignment": 1,
    "sole_height": -1,
    "joint_symmetry": -1,
    "base_z_roll": -1,
    "reset_failure": -1,
}
CHECKPOINT_KEY = "snake_to_biped_hot_rewards"


class HotRewardController:
    """完整校验旁路配置文件后，再原子应用实时奖励权重。"""

    def __init__(self, env, log_dir=None):
        self.env = env
        self.log_path = Path(log_dir) / "hot_reward_updates.jsonl" if log_dir else None
        self._last_file = None

    def _signs(self):
        if getattr(self.env.cfg, "compact_standing_rewards", False):
            return ({name: CONFIG_SIGNS[name] for name in ("standing_stage_joint_weight", "standing_stage_height_weight", "standing_stage_velocity_weight")}, {"reset_failure": -1, "sole_ground_alignment": 1})
        return CONFIG_SIGNS, REWARD_SIGNS

    def snapshot(self):
        env = self.env
        stage = getattr(env, "_standing_curriculum_stage", None)
        start = getattr(env, "_standing_stage_start_step", env.common_step_counter)
        config_signs, reward_signs = self._signs()
        return {
            "config": {name: getattr(env.cfg, name) for name in config_signs},
            "rewards": {
                name: env.reward_manager.get_term_cfg(name).weight
                for name in reward_signs
            },
            "effective_weights": {
                name: env.reward_manager.get_term_cfg(name).weight
                for name in env.reward_manager.active_terms
            },
            "stage": stage,
            "pose_prepared": bool(
                getattr(env, "_standing_pose_prepared", False)
                or (stage == 2 and getattr(env.cfg, "curriculum_prepare_pose", False))
            ),
            "ramp_elapsed_steps": max(0, env.common_step_counter - start),
        }

    def _validate(self, payload):
        if not isinstance(payload, dict) or payload.keys() - {"config", "rewards"}:
            raise ValueError("Only config and rewards mappings are hot-editable")
        config_signs, reward_signs = self._signs()
        for section, signs in (("config", config_signs), ("rewards", reward_signs)):
            values = payload.get(section, {})
            if not isinstance(values, dict) or values.keys() - signs.keys():
                raise ValueError(f"Unknown fields in {section}")
            for name, value in values.items():
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value * signs[name] < 0
                ):
                    raise ValueError(f"Invalid finite/sign weight: {name}={value!r}")
        if getattr(self.env.cfg, "compact_standing_rewards", False):
            return
        alignment = payload.get("rewards", {}).get("sole_ground_alignment", self.env.reward_manager.get_term_cfg("sole_ground_alignment").weight)
        alignment_required = (
            getattr(self.env.cfg, "curriculum_start_stage", 1) == 1
            or getattr(self.env.cfg, "curriculum_prepare_pose", False)
        )
        if alignment_required and not 0 <= self.env.cfg.sole_alignment_stage_threshold < alignment:
            raise ValueError("sole alignment stage threshold must remain reachable")

    def _apply(self, payload):
        from .tasks.transformation.snake_to_biped import apply_standing_stage_weights

        for name, value in payload.get("config", {}).items():
            setattr(self.env.cfg, name, float(value))
        for name, value in payload.get("rewards", {}).items():
            self.env.reward_manager.get_term_cfg(name).weight = float(value)
        if getattr(self.env.cfg, "compact_standing_rewards", False):
            from .tasks.transformation.robust import apply_fixed_standing_weights

            apply_fixed_standing_weights(self.env)
        elif hasattr(self.env, "_standing_curriculum_stage"):
            apply_standing_stage_weights(self.env)
        if "both_feet_contact" in self.env.reward_manager.active_terms:
            self.env.reward_manager.get_term_cfg("both_feet_contact").weight = self.env.cfg.feet_contact_weight

    def _log(self, status, iteration, **details):
        record = {
            "status": status,
            "iteration": iteration,
            **details,
            **self.snapshot(),
        }
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
        print(
            f"[hot rewards] {status} at iteration {iteration}"
            + (f": {details['error']}" if "error" in details else "")
        )

    def _transaction(self, payload):
        self._validate(payload)
        previous = self.snapshot()
        missing = object()
        previous_changed_step = getattr(
            self.env, "_standing_alignment_hot_changed_step", missing
        )
        try:
            self._apply(payload)
            alignment = self.env.reward_manager.get_term_cfg("sole_ground_alignment").weight if "sole_ground_alignment" in self.env.reward_manager.active_terms else None
            if alignment is not None and alignment != previous["rewards"]["sole_ground_alignment"]:
                # 全部进行中的回合结束前，现有回合累计量混合了旧权重和新权重得分。
                # 不要据此提升阶段。
                self.env._standing_alignment_hot_changed_step = (
                    self.env.common_step_counter
                )
        except Exception:
            for name, value in previous["config"].items():
                setattr(self.env.cfg, name, value)
            for name, value in previous["effective_weights"].items():
                self.env.reward_manager.get_term_cfg(name).weight = value
            if previous_changed_step is missing:
                if hasattr(self.env, "_standing_alignment_hot_changed_step"):
                    delattr(self.env, "_standing_alignment_hot_changed_step")
            else:
                self.env._standing_alignment_hot_changed_step = previous_changed_step
            raise

    def poll(self, iteration):
        """文件内容变更后仅应用一次；写入不完整时保留旧权重。"""
        path = getattr(self.env.cfg, "reward_hot_reload_path", None)
        if path is None:
            return False
        try:
            contents = Path(path).read_bytes()
            fingerprint = (str(path), contents)
        except OSError as error:
            fingerprint = (str(path), type(error).__name__, str(error))
            if fingerprint != self._last_file:
                self._last_file = fingerprint
                self._log("rejected", iteration, error=str(error))
            return False
        if fingerprint == self._last_file:
            return False
        self._last_file = fingerprint
        try:
            self._transaction(json.loads(contents))
        except (
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
            OverflowError,
        ) as error:
            self._log("rejected", iteration, error=str(error))
            return False
        self._log("applied", iteration, source=str(path))
        return True

    def restore(self, snapshot, iteration):
        """重新读取旁路配置文件前，恢复已保存权重和课程进度。"""
        self._transaction({name: snapshot[name] for name in ("config", "rewards")})
        stage = snapshot.get("stage")
        self.env._standing_pose_prepared = snapshot.get("pose_prepared", False)
        if stage is not None:
            from .tasks.transformation.snake_to_biped import (
                apply_standing_stage_weights,
            )

            self.env._standing_curriculum_stage = stage
            self.env._standing_stage_start_step = (
                self.env.common_step_counter - snapshot["ramp_elapsed_steps"]
            )
            apply_standing_stage_weights(self.env)
        self._log("restored", iteration)
        self._last_file = None
        self.poll(iteration)


class SnakeToBipedHotRewardRunner(MjlabOnPolicyRunner):
    """保留上游 PPO 实现，每次算法更新成功后重新加载。"""

    def __init__(self, env: RslRlVecEnvWrapper, train_cfg, log_dir=None, device="cpu"):
        super().__init__(env, train_cfg, log_dir, device)
        self._action_std_projection = ActionStdProjection(
            getattr(self.alg.get_policy(), "distribution", None)
        )
        self._action_std_projection.project()
        self._action_std_projection_hook = self._action_std_projection.install(
            self.alg.optimizer
        )
        self.hot_rewards = HotRewardController(env.unwrapped, log_dir)
        self._hot_update_iteration = self.current_learning_iteration
        self.hot_rewards.poll(self._hot_update_iteration)
        update = self.alg.update

        def update_then_reload(*args, **kwargs):
            result = update(*args, **kwargs)
            self._hot_update_iteration += 1
            self.hot_rewards.poll(self._hot_update_iteration)
            return result

        # 有意按实例包装，不替换 PPO 或其优化器。
        self.alg.update = update_then_reload  # ty: ignore[invalid-assignment]

    def save(self, path: str, infos=None) -> None:
        infos = {**(infos or {}), CHECKPOINT_KEY: self.hot_rewards.snapshot()}
        super().save(path, infos)

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict:
        full_load = load_cfg is None or all(
            load_cfg.get(key, False)
            for key in ("actor", "critic", "optimizer", "iteration")
        )
        previous_counter = self.hot_rewards.env.common_step_counter
        infos = super().load(path, load_cfg, strict, map_location)
        if full_load:
            self._action_std_projection.project()
            self._hot_update_iteration = self.current_learning_iteration
            if infos and CHECKPOINT_KEY in infos:
                self.hot_rewards.restore(
                    infos[CHECKPOINT_KEY], self._hot_update_iteration
                )
        else:
            # 上游运行器即使在推理加载时也会恢复计数器。
            self.hot_rewards.env.common_step_counter = previous_counter
        return infos


class RobustSnakeToBipedRunner(SnakeToBipedHotRewardRunner):
    """热启动新训练；常规 --agent.resume 仍恢复完整检查点。"""

    def load(self, path, load_cfg=None, strict=True, map_location=None):
        full_load = load_cfg is None or all(load_cfg.get(key, False) for key in ("actor", "critic", "optimizer", "iteration"))
        env = self.env.unwrapped
        if not isinstance(env.cfg, RobustSnakeToBipedEnvCfg):
            raise TypeError("robust runner requires RobustSnakeToBipedEnvCfg")
        # 恢复旧训练时，不得恢复已被替代的足部优先配方。
        recipe_config = {name: getattr(env.cfg, name) for name in (
            "standing_stage_joint_weight", "standing_stage_height_weight", "standing_stage_velocity_weight",
        )}
        recipe_rewards = {name: env.reward_manager.get_term_cfg(name).weight for name in ("reset_failure", "sole_ground_alignment")}
        payload = torch.load(path, map_location=map_location or self.device, weights_only=False) if Path(path).is_file() else None
        migrated = False
        if payload is not None:
            actor, migrated = migrate_beta_mirror_head(payload["actor_state_dict"])
            if migrated:
                payload["actor_state_dict"] = actor
        if migrated:
            # 六动作训练的优化器张量和迭代计划与新策略不兼容。
            # 仅迁移权重，并以全新训练启动。
            effective_load = {"actor": True, "critic": full_load or bool(load_cfg and load_cfg.get("critic")), "optimizer": False, "iteration": False}
            with tempfile.TemporaryDirectory(prefix="zbot-mirror-") as temporary:
                converted = Path(temporary) / "checkpoint.pt"
                torch.save(payload, converted)
                infos = super().load(str(converted), effective_load, strict, map_location)
            self._mirror_migration = {"source": str(Path(path).resolve()), "old_action_dim": 6, "new_action_dim": 3, "optimizer_and_iteration_restored": False}
            print("[INFO] Migrated six-action Beta head to three mirrored actions; optimizer and iteration remain fresh.")
        else:
            infos = super().load(path, load_cfg, strict, map_location)
        if full_load:
            for name, value in recipe_config.items():
                setattr(env.cfg, name, value)
            for name, weight in recipe_rewards.items():
                env.reward_manager.get_term_cfg(name).weight = weight
            apply_fixed_standing_weights(env)
        return infos

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        super().__init__(env, train_cfg, log_dir, device)
        cfg = env.unwrapped.cfg
        if not isinstance(cfg, RobustSnakeToBipedEnvCfg):
            raise TypeError("robust runner requires RobustSnakeToBipedEnvCfg")
        source = cfg.pretrained_checkpoint
        if log_dir is None or train_cfg.get("resume", False) or source is None:
            return
        path = Path(source).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Pretrained snake-to-biped checkpoint not found: {path}")
        load_cfg = {"actor": True, "critic": True, "optimizer": False, "iteration": False}
        self.load(str(path), load_cfg=load_cfg, strict=True, map_location=device)
        record = {
            "source": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "load_cfg": load_cfg,
            "initial_iteration": self.current_learning_iteration,
            "initial_common_step_counter": env.unwrapped.common_step_counter,
            "learning_rate": self.alg.learning_rate,
            "mirror_migration": getattr(self, "_mirror_migration", None),
        }
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        (Path(log_dir) / "pretrained_initialization.json").write_text(json.dumps(record, indent=2) + "\n")
        print(f"[INFO] Warm-started actor/critic from {path}; optimizer and training counters are fresh.")

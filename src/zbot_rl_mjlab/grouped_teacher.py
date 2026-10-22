"""Isolated reward-group teacher runner; existing teacher archives keep their runner."""

from pathlib import Path
from typing import Any

import torch

from .omni_teacher import OmniTeacherRunner
from .tasks.universal.reward_groups import GROUP_NAMES, install_reward_group_capture


class GroupedTeacherRunner(OmniTeacherRunner):
    """Retain teacher progress/course handling with a distinct checkpoint contract."""

    allowed_algorithm_classes = ("zbot_rl_mjlab.grouped_ppo:GroupedPPO",)

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        install_reward_group_capture(env.unwrapped)
        super().__init__(env, train_cfg, log_dir, device)

    def _schema(self) -> dict[str, Any]:
        schema = super()._schema()
        if (schema["actor_obs_dim"], schema["critic_obs_dim"], schema["action_dim"]) != (56, 56, 6):
            raise ValueError("reward-group teacher requires scalar 56-dimensional observations and 6 actions")
        return schema | {
            "role": "privileged_omni_reward_group_teacher",
            "schema_version": 3,
            "training_algorithm": "reward_grouped_ppo_v1",
            "reward_groups": GROUP_NAMES,
        }

    @staticmethod
    def _validate_metadata(saved: dict) -> None:
        infos = saved.get("infos") or {}
        metadata = infos.get("privileged_omni_teacher")
        expected = {
            "role": "privileged_omni_reward_group_teacher",
            "schema_version": 3,
            "actor_obs_dim": 56,
            "critic_obs_dim": 56,
            "action_dim": 6,
            "training_algorithm": "reward_grouped_ppo_v1",
            "reward_groups": GROUP_NAMES,
        }
        if not isinstance(metadata, dict) or any(
            type(metadata.get(key)) is not type(value) or metadata.get(key) != value
            for key, value in expected.items()
        ):
            raise ValueError("checkpoint reward-group teacher role/schema/algorithm mismatch")

    @staticmethod
    def prepare_playback_env_cfg(env_cfg, checkpoint: str | Path) -> None:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        GroupedTeacherRunner._validate_metadata(saved)
        command = env_cfg.commands.get("transform")
        if command is None or not hasattr(command, "task_observation_encoding"):
            raise ValueError("reward-group teacher playback requires a privileged transform command")
        command.task_observation_encoding = "scalar"

    def load(self, path, load_cfg=None, strict=True, map_location=None):
        # Actor-only playback must not silently load a baseline teacher with another role.
        saved = torch.load(path, map_location="cpu", weights_only=False)
        self._validate_metadata(saved)
        return super().load(path, load_cfg, strict, map_location)

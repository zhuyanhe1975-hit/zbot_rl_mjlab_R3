from mjlab.envs.mdp.curriculums import _apply_stages
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

import zbot_rl_mjlab  # noqa: F401
from zbot_rl_mjlab.experiments import flat_foot_velocity_energy as flat

TASK = flat.FLAT_FOOT_VELOCITY_ENERGY_TASK_ID


def test_stage_boundaries_use_the_ppo_iteration_length():
    assert flat.STEPS_PER_ITERATION == load_rl_cfg(TASK).num_steps_per_env


def test_weights_only_get_stronger_and_end_at_the_full_values():
    iterations = [row[0] for row in flat.WEIGHT_STAGES]
    assert iterations == sorted(iterations) and iterations[0] == 0
    for column in (1, 2, 3):
        weights = [row[column] for row in flat.WEIGHT_STAGES]
        assert all(w < 0 for w in weights)
        assert weights == sorted(weights, reverse=True)  # 数值越来越负
    assert flat.WEIGHT_STAGES[-1][1:] == (-0.5, -2.0, -0.5)


def test_every_penalty_has_a_curriculum_term_in_train_and_play():
    for play in (False, True):
        cfg = load_env_cfg(TASK, play=play)
        for name in flat._TERMS:
            assert name in cfg.rewards
            term = cfg.curriculum[f"{name}_weight"]
            assert term.params["reward_name"] == name
            assert cfg.rewards[name].weight == term.params["stages"][0]["weight"]


def test_stage_applies_at_its_first_step_not_before():
    reward = RewardTermCfg(func=lambda env: 0.0, weight=0.0)
    stages = [
        {"step": row[0] * flat.STEPS_PER_ITERATION, "weight": row[1]}
        for row in flat.WEIGHT_STAGES
    ]
    boundary = stages[1]["step"]
    _apply_stages(reward, boundary - 1, stages)
    assert reward.weight == flat.WEIGHT_STAGES[0][1]
    _apply_stages(reward, boundary, stages)
    assert reward.weight == flat.WEIGHT_STAGES[1][1]
    _apply_stages(reward, 10**9, stages)
    assert reward.weight == flat.WEIGHT_STAGES[-1][1]

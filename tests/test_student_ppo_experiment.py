"""PPO is a separate actor-initialized phase with student-only objectives."""

import copy
import hashlib
import re
from dataclasses import asdict
from pathlib import Path

import pytest
import tyro
from mjlab.scripts.train import TrainConfig
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from test_student_observations import environment
from test_task_contracts import task_contract

from zbot_rl_mjlab.experiments.omni_teacher import SnakeLegOmniGuidedTeacher
from zbot_rl_mjlab.experiments.student import UniversalOmniStudent
from zbot_rl_mjlab.experiments.student_ppo import UniversalOmniStudentPPO
from zbot_rl_mjlab.student_ppo import StudentPPORunnerCfg, UniversalStudentPPORunner
from zbot_rl_mjlab.tasks.universal.student_observations import (
    GuidedStudentEnvCfg,
    student_sensor_state,
)
from zbot_rl_mjlab.tasks.universal.student_rewards import (
    student_touchdown_flat,
    student_velocity_tracking,
)
from zbot_rl_mjlab.tasks.universal.teacher_rewards import teacher_action_rate

ROOT = Path(__file__).resolve().parents[1]
TASK = UniversalOmniStudentPPO.task_id
SOURCE_SHA256 = '63946c079786be56835dcb3b0ad046248d145175b00ed4c7cbed84969b4ef022'


def test_ppo_registration_keeps_history_actor_and_privileged_task_critic():
    cfg = load_rl_cfg(TASK)
    assert isinstance(cfg, StudentPPORunnerCfg)
    assert cfg.obs_groups == {'actor': ('actor',), 'critic': ('teacher',)}
    assert cfg.actor.class_name == 'zbot_rl_mjlab.student_ppo:PPOHistoryGaussianStudentMLP'
    assert cfg.critic.class_name == 'zbot_rl_mjlab.task_models:LocalFrameTaskNormalizedCritics'
    assert cfg.class_name == 'zbot_rl_mjlab.student_ppo:UniversalStudentPPORunner'
    assert load_runner_cls(TASK) is UniversalStudentPPORunner
    distribution = cfg.actor.distribution_cfg
    assert distribution is not None
    assert distribution['class_name'].endswith(':GaussianTaskDistribution')
    assert distribution['learn_std'] is False
    assert distribution['init_std'] == distribution['rate_init_std'] == .1
    env = load_env_cfg(TASK)
    actor, critic = env.observations['actor'], env.observations['teacher']
    assert actor.history_length == 5 and actor.flatten_history_dim
    assert not critic.history_length
    fake, _, _, _ = environment()
    assert student_sensor_state(fake).shape[-1] * actor.history_length == 280


def test_ppo_algorithm_does_not_inherit_dagger_or_teacher_initialization():
    cfg = load_rl_cfg(TASK)
    assert isinstance(cfg, StudentPPORunnerCfg)
    algorithm = asdict(cfg.algorithm)
    assert algorithm['class_name'] == 'zbot_rl_mjlab.student_ppo:StudentPPO'
    assert algorithm['learning_rate'] == 3e-5
    assert algorithm['schedule'] == 'fixed'
    assert algorithm['clip_param'] == .1
    assert algorithm['entropy_coef'] == 0.
    assert algorithm['critic_warmup_iterations'] == 10
    assert not any(name.startswith(('teacher_', 'dagger_', 'guidance_')) for name in algorithm)
    assert 'teacher_checkpoint' not in asdict(cfg)
    assert not cfg.resume


def test_initial_actor_checkpoint_is_the_verified_iteration150_student():
    values = asdict(load_rl_cfg(TASK))
    source = ROOT / values['initial_student_checkpoint']
    assert source.name == 'model_150.pt'
    assert 'rsl_rl_universal_student_v2' in str(source)
    assert values['initial_student_sha256'] == SOURCE_SHA256
    assert source.is_file()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == SOURCE_SHA256


@pytest.mark.parametrize('play', [False, True])
def test_ppo_retains_student_reward_corrections_and_sensor_noise_modes(play):
    cfg = load_env_cfg(TASK, play=play)
    assert isinstance(cfg, GuidedStudentEnvCfg)
    assert cfg.rewards['action_rate'].func is teacher_action_rate
    assert cfg.rewards['velocity_tracking'].func is student_velocity_tracking
    assert cfg.rewards['velocity_tracking'].params['snake_std'] == (.05, .05, .15)
    assert cfg.rewards['foot_flat_touchdown'].func is student_touchdown_flat
    assert cfg.rewards['foot_flat_touchdown'].weight == -.2
    assert cfg.student_observation_noise is (not play)


def test_building_ppo_preserves_teacher_and_dagger_recipes():
    sources = (SnakeLegOmniGuidedTeacher, UniversalOmniStudent)
    before = {source.task_id: task_contract(source) for source in sources}
    experiment = UniversalOmniStudentPPO()
    experiment.build_env()
    experiment.build_env(play=True)
    experiment.build_ppo()
    for source in sources:
        assert task_contract(source) == before[source.task_id]


def test_card_parameters_remain_editable_for_ppo_and_inherited_rewards(monkeypatch):
    experiment = UniversalOmniStudentPPO()
    card = copy.deepcopy(experiment.configuration_card())
    assert card is not None
    card['agent']['algorithm'].update(learning_rate=1e-5, critic_warmup_iterations=3)
    card['rewards']['velocity_tracking']['params']['snake_std'] = [.03, .04, .12]
    card['rewards']['foot_flat_touchdown']['weight'] = -.1
    monkeypatch.setattr(experiment, 'configuration_card', lambda: card)
    agent = asdict(experiment.build_ppo())
    assert agent['algorithm']['learning_rate'] == 1e-5
    assert agent['algorithm']['critic_warmup_iterations'] == 3
    for play in (False, True):
        env = experiment.build_env(play=play)
        assert env.rewards['velocity_tracking'].params['snake_std'] == (.03, .04, .12)
        assert env.rewards['foot_flat_touchdown'].weight == -.1


def test_train_parser_accepts_actor_source_and_critic_warmup_without_starting_env():
    parsed = tyro.cli(
        TrainConfig, default=TrainConfig.from_task(TASK), args=[
            '--agent.initial-student-checkpoint', 'alternate/model_150.pt',
            '--agent.initial-student-sha256', 'a' * 64,
            '--agent.algorithm.critic-warmup-iterations', '2',
            '--agent.algorithm.learning-rate', '0.00001',
        ],
    )
    agent = asdict(parsed.agent)
    assert agent['initial_student_checkpoint'] == 'alternate/model_150.pt'
    assert agent['initial_student_sha256'] == 'a' * 64
    assert agent['algorithm']['critic_warmup_iterations'] == 2
    assert agent['algorithm']['learning_rate'] == 1e-5


@pytest.mark.parametrize('filename,entrypoint', [('train.sh', 'train.py'), ('run.sh', 'play.py')])
def test_r3_frozen_scripts_preserve_ppo_commands(filename, entrypoint):
    text = (ROOT / filename).read_text()
    active = '\n'.join(line for line in text.splitlines() if not line.lstrip().startswith('#'))
    assert not re.search(r'^\s*(?:exec\s+)?python\S*\s', active, re.MULTILINE)
    preserved = (UniversalOmniStudent.task_id, UniversalOmniStudent.task_id + '-Disturbed',
                 TASK, TASK + '-V4')
    if filename == 'train.sh':
        preserved += (TASK + '-V3',)
    for task in preserved:
        assert re.search(rf'^#\s*python {re.escape(entrypoint)} {re.escape(task)}\s*\\', text, re.MULTILINE)
    assert '"$@"' in text

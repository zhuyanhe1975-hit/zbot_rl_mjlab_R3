"""最终部署学生的注册、物理随机化及父任务隔离。"""

from dataclasses import asdict
from types import SimpleNamespace

import pytest
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

from zbot_rl_mjlab.distillation import MultiTeacherRunner
from zbot_rl_mjlab.experiments.addons import (
    BaseDisturbance,
    DomainRandomization,
    GainRandomization,
    MassRandomization,
)
from zbot_rl_mjlab.experiments.universal import (
    DEPLOY_TASK_ID,
    DeployUniversalLocomotion,
    StudentDisturbance,
    UniversalLocomotion,
)
from zbot_rl_mjlab.tasks.common import PartialBaseImpulse
from zbot_rl_mjlab.tasks.universal.deployment import StudentBaseImpulse


def test_registered_student_retains_training_algorithm_with_separate_experiment():
    assert issubclass(DeployUniversalLocomotion, UniversalLocomotion)
    assert DEPLOY_TASK_ID == 'Mjlab-Zbot-6dof-Universal-Locomotion-Deploy'
    assert load_runner_cls(DEPLOY_TASK_ID) is MultiTeacherRunner
    cfg = load_rl_cfg(DEPLOY_TASK_ID)
    assert cfg.experiment_name == 'zbot_universal_locomotion_deploy'
    assert not cfg.algorithm.validation_use_mse
    assert not cfg.validation_randomized
    assert cfg.robust_validation_interval == 100
    assert cfg.algorithm.distillation_enabled


def test_student_only_adds_physical_randomization_and_disturbance():
    train = load_env_cfg(DEPLOY_TASK_ID)
    play = load_env_cfg(DEPLOY_TASK_ID, play=True)
    addons = DeployUniversalLocomotion().collected_addons()
    assert [type(item) for item in addons] == [MassRandomization, GainRandomization, StudentDisturbance, DomainRandomization]
    assert isinstance(addons[2], BaseDisturbance)
    actor = train.observations['actor']
    assert set(actor.terms) == {'joint_pos','joint_vel','commanded_target','actions','projected_gravity','base_ang_vel','command','gait'}
    assert actor.history_length == play.observations['actor'].history_length == 5
    assert not actor.enable_corruption and not play.observations['actor'].enable_corruption
    assert all(term.noise is None for term in actor.terms.values())
    assert all(term.noise is None for term in train.observations['critic'].terms.values())
    assert asdict(train.actions['joint_pos']) == asdict(UniversalLocomotion().build_env().actions['joint_pos'])
    events = train.events
    assert {'base_disturbance', 'randomize_mass_inertia', 'randomize_pd_gains', 'randomize_base_com'} <= events.keys()
    assert 'randomize_encoder_bias' not in events
    assert events['randomize_foot_friction'].params['asset_cfg'].geom_names == ('.*',)
    assert list(events).index('randomize_mass_inertia') < list(events).index('randomize_base_com')
    base = UniversalLocomotion().build_env()
    assert {key:value for key,value in train.rewards.items() if key != 'straightening'} == base.rewards
    assert train.rewards['straightening'].weight == -2.


def test_parent_stays_full_state_and_does_not_enable_robustness():
    cfg = UniversalLocomotion().build_env()
    assert list(cfg.observations['actor'].terms) == ['universal_state']
    assert cfg.observations['actor'].history_length is None
    assert 'base_disturbance' not in cfg.events
    assert UniversalLocomotion().build_ppo().algorithm.validation_use_mse


def test_evaluator_keeps_student_schema_and_separates_clean_and_robust():
    from zbot_rl_mjlab.universal_evaluation import evaluation_env_cfg

    source = load_env_cfg(DEPLOY_TASK_ID)
    clean = evaluation_env_cfg(source, num_envs=16, randomized=False)
    robust = evaluation_env_cfg(source, num_envs=16, randomized=True)
    assert source.scene.num_envs == 8192 and clean.scene.num_envs == 16
    assert clean.observations['actor'].history_length == 5
    assert set(clean.observations['actor'].terms) == set(source.observations['actor'].terms)
    assert not clean.commands['transform'].training and source.commands['transform'].training
    assert not any(name.startswith('randomize_') for name in clean.events)
    assert 'base_disturbance' not in clean.events
    assert {'base_disturbance','randomize_mass_inertia','randomize_pd_gains'} <= robust.events.keys()
    assert set(source.events) == set(robust.events)
    assert robust.events['base_disturbance'].params['disturbed_fraction'] == 1.
    assert source.events['base_disturbance'].params['disturbed_fraction'] == .25


@pytest.mark.parametrize(('pretraining','iteration','training','scale'), [(True,1000,True,.2),(False,0,True,.2),(False,250,True,.6),(False,500,True,1.),(False,2000,True,1.),(True,0,False,1.)])
def test_external_disturbance_follows_actual_student_rl_progress(monkeypatch, pretraining, iteration, training, scale):
    seen = {}
    monkeypatch.setattr(PartialBaseImpulse, '__call__', lambda self, env, env_ids, *args, **kwargs: seen.update(kwargs))
    env = SimpleNamespace(cfg=SimpleNamespace(commands={'transform':SimpleNamespace(training=training)}),_student_pretraining=pretraining,_student_rl_iteration=iteration)
    impulse = object.__new__(StudentBaseImpulse)
    impulse(env, None, force_range=(-5.,5.),torque_range=(-.5,.5))
    assert seen['force_range'] == pytest.approx((-5*scale,5*scale))
    assert seen['torque_range'] == pytest.approx((-.5*scale,.5*scale))


@pytest.mark.parametrize('kwargs',[{'pretraining_scale':-1.},{'pretraining_scale':float('nan')},{'ramp_iterations':0},{'ramp_iterations':True}])
def test_invalid_disturbance_curriculum_is_rejected(kwargs):
    with pytest.raises(ValueError):
        StudentDisturbance(**kwargs)

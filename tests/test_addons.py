import itertools

import pytest
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg

import zbot_rl_mjlab
from zbot_rl_mjlab.experiments import addons as ad
from zbot_rl_mjlab.experiments.base import Experiment, WalkingExperiment, variant
from zbot_rl_mjlab.experiments.variants import (
    DisturbedWalking,
    FlatFootDisturbed,
    VelocityEnergyDisturbed,
)
from zbot_rl_mjlab.experiments.walking import OmniVelocityEnergy, VelocityEnergyTracking
from zbot_rl_mjlab.tasks.walking.actions import DelayedJointPositionActionCfg

BASE = VelocityEnergyTracking
RESET = "reset_scene_to_default"


def build(*addons, base=BASE, play=False):
    return variant(base, "Mjlab-Zbot-Test", "test", *addons)().build_env(play=play)


def event_names(cfg):
    return list(cfg.events)


def test_without_addons_the_experiment_has_no_randomization_noise_or_delay():
    cfg = BASE().build_env()
    assert event_names(cfg) == [RESET]
    actor = cfg.observations["actor"].terms
    assert "base_lin_vel" in actor and "yaw_error" in actor
    assert all(term.noise is None for term in actor.values())
    assert type(cfg.actions["joint_pos"]) is JointPositionActionCfg
    assert BASE.addons == ()


def test_base_disturbance_alone_adds_only_the_push_event_and_is_tunable():
    assert event_names(build(ad.BaseDisturbance())) == [RESET, "base_disturbance"]
    params = build(ad.BaseDisturbance(disturbed_fraction=1.0, force_range=(-9.0, 9.0)))
    params = params.events["base_disturbance"].params
    assert params["disturbed_fraction"] == 1.0
    assert params["force_range"] == (-9.0, 9.0)
    default = build(ad.BaseDisturbance()).events["base_disturbance"].params
    assert default["disturbed_fraction"] == 0.25 and default["force_range"] == (
        -5.0,
        5.0,
    )


def test_domain_randomization_alone_and_tunable():
    names = event_names(build(ad.DomainRandomization()))
    assert names == [
        RESET,
        "randomize_foot_friction",
        "randomize_encoder_bias",
        "randomize_base_com",
    ]
    custom = build(ad.DomainRandomization(friction_range=(0.5, 0.9)))
    assert custom.events["randomize_foot_friction"].params["ranges"] == (0.5, 0.9)


def test_mass_and_gain_randomization_events():
    cfg = build(ad.MassRandomization(alpha_range=(-0.1, 0.1)), ad.GainRandomization())
    assert event_names(cfg) == [RESET, "randomize_mass_inertia", "randomize_pd_gains"]
    assert cfg.events["randomize_mass_inertia"].params["alpha_range"] == (-0.1, 0.1)
    gains = cfg.events["randomize_pd_gains"]
    assert gains.mode == "reset"
    assert gains.params["kp_range"] == gains.params["kd_range"] == ad.GAIN_SCALE


def test_blind_actor_hides_terms_from_the_actor_only():
    cfg = build(ad.BlindActor())
    assert "base_lin_vel" not in cfg.observations["actor"].terms
    assert "yaw_error" not in cfg.observations["actor"].terms
    assert "base_lin_vel" in cfg.observations["critic"].terms
    only = build(ad.BlindActor(hidden=("yaw_error",)))
    assert "base_lin_vel" in only.observations["actor"].terms
    assert "yaw_error" not in only.observations["actor"].terms


def test_observation_noise_goes_on_the_actor_and_is_off_for_play():
    cfg = build(ad.ObservationNoise())
    actor, critic = cfg.observations["actor"], cfg.observations["critic"]
    assert actor.enable_corruption and not critic.enable_corruption
    assert (
        not build(ad.ObservationNoise(), play=True)
        .observations["actor"]
        .enable_corruption
    )
    assert all(term.noise is None for term in critic.terms.values())
    assert isinstance(actor.terms["joint_pos"].noise, GaussianNoiseCfg)
    assert isinstance(actor.terms["base_ang_vel"].noise, NoiseModelWithAdditiveBiasCfg)
    assert actor.terms["base_lin_vel"].noise is None


def test_observation_noise_validates_its_arguments():
    with pytest.raises(KeyError, match="no_such_term"):
        build(ad.ObservationNoise(std={"no_such_term": 0.1}, bias={}))
    with pytest.raises(ValueError):
        build(ad.ObservationNoise(std={"joint_pos": 0.1}, bias={"joint_vel": 0.1}))
    # 被其他附加配置对策略网络隐藏的观测项，仍属于已知观测项。
    build(ad.BlindActor(), ad.ObservationNoise(std={"base_lin_vel": 0.1}, bias={}))


def test_action_delay_applies_to_training_only_and_keeps_scale_and_offset():
    train = build(ad.ActionDelay(max_lag=3)).actions["joint_pos"]
    play = build(ad.ActionDelay(max_lag=3), play=True).actions["joint_pos"]
    base = BASE().build_env().actions["joint_pos"]
    assert isinstance(train, DelayedJointPositionActionCfg)
    assert (train.max_lag, play.max_lag) == (3, 0)
    assert (train.scale, train.use_default_offset) == (base.scale, True)


def all_orders(addons):
    return itertools.permutations(addons)


def test_the_result_does_not_depend_on_how_add_ons_are_listed():
    reference = build(*ad.DEPLOYMENT)
    for order in itertools.islice(all_orders(ad.DEPLOYMENT), 0, 5040, 211):
        cfg = build(*order)
        assert event_names(cfg) == event_names(reference)
        assert list(cfg.observations["actor"].terms) == list(
            reference.observations["actor"].terms
        )
        assert type(cfg.actions["joint_pos"]) is type(reference.actions["joint_pos"])
    events = event_names(reference)
    assert events.index("randomize_mass_inertia") < events.index("randomize_base_com")


def test_a_subclass_replaces_a_parents_add_on_of_the_same_type_and_adds_others():
    class Parent(BASE):
        task_id, experiment_name = "Mjlab-Zbot-Parent", "parent"
        addons = (ad.BaseDisturbance(), ad.DomainRandomization())

    class Child(Parent):
        task_id, experiment_name = "Mjlab-Zbot-Child", "child"
        addons = (ad.BaseDisturbance(disturbed_fraction=1.0), ad.ActionDelay())

    assert [type(a) for a in Parent().collected_addons()] == [
        ad.BaseDisturbance,
        ad.DomainRandomization,
    ]
    cfg = Child().build_env()
    assert cfg.events["base_disturbance"].params["disturbed_fraction"] == 1.0
    assert "randomize_foot_friction" in cfg.events  # 继承的配置
    assert isinstance(cfg.actions["joint_pos"], DelayedJointPositionActionCfg)
    # 子配置不应修改父配置。
    assert (
        Parent().build_env().events["base_disturbance"].params["disturbed_fraction"]
        == 0.25
    )


@pytest.mark.parametrize("play", [False, True])
@pytest.mark.parametrize(
    "experiment",
    [
        experiment
        for experiment in zbot_rl_mjlab.EXPERIMENTS
        if issubclass(experiment, WalkingExperiment)
    ],
)
def test_every_walking_experiment_accepts_every_add_on(experiment, play):
    cfg = build(*ad.DEPLOYMENT, base=experiment, play=play)
    actor = cfg.observations["actor"].terms
    assert not set(ad.PRIVILEGED_TERMS) & set(actor)
    assert {
        "base_disturbance",
        "randomize_foot_friction",
        "randomize_mass_inertia",
        "randomize_pd_gains",
    } <= set(cfg.events)
    assert isinstance(cfg.actions["joint_pos"], DelayedJointPositionActionCfg)


def test_variant_creates_a_new_experiment_without_touching_the_base():
    before = OmniVelocityEnergy().build_env()
    new = variant(
        OmniVelocityEnergy,
        "Mjlab-Zbot-Omni-Disturbed",
        "zbot_omni_disturbed",
        *ad.DISTURBED,
    )
    assert issubclass(new, OmniVelocityEnergy) and issubclass(new, Experiment)
    assert (new.task_id, new.experiment_name) == (
        "Mjlab-Zbot-Omni-Disturbed",
        "zbot_omni_disturbed",
    )
    assert new().build_ppo().experiment_name == "zbot_omni_disturbed"
    assert "base_disturbance" in new().build_env().events
    after = OmniVelocityEnergy().build_env()
    assert event_names(before) == event_names(after) == [RESET]
    assert OmniVelocityEnergy.addons == ()


def test_the_disturbed_and_deployment_tasks_use_the_presets():
    for experiment in (DisturbedWalking, VelocityEnergyDisturbed, FlatFootDisturbed):
        assert experiment().collected_addons() == sorted(
            ad.DISTURBED, key=lambda a: a.order
        )
    deploy_types = [type(a) for a in ad.DEPLOYMENT]
    assert [
        type(a) for a in sorted(ad.DEPLOYMENT, key=lambda a: a.order)
    ] == deploy_types

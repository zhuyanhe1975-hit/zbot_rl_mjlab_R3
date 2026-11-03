"""Population isolation, staged allocation, mutation, and checkpoint fidelity."""
import copy
from dataclasses import asdict
import random

import pytest
import torch
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

from zbot_rl_mjlab.reconfiguration_pbt import (
    PbtCfg, ReconfigurationPbtRunner, apply_parameters, choose_replacement,
    mutate, objective, parameters, validate,
)

TASK = 'Mjlab-Zbot-6dof-Reconfigure-Head-Tail-Docking-PBT'


def test_population_budget_and_default_stages():
    cfg = load_rl_cfg(TASK)
    env_cfg = load_env_cfg(TASK)
    assert cfg.pbt.population_size == 8 and cfg.pbt.focused_population_size == 4
    assert env_cfg.scene.num_envs == 8192
    assert cfg.pbt.warmup_iterations == cfg.pbt.focus_after_iterations == 100
    validate(asdict(cfg.pbt), env_cfg.scene.num_envs)
    assert cfg.max_iterations == 500


@pytest.mark.parametrize('patch', [dict(population_size=3), dict(focused_population_size=3),
    dict(mutation_factor=1.), dict(mutation_probability=2.), dict(gamma_bounds=(.99, 1.)),
    dict(evaluation_episode_s=0.), dict(minimum_score_gap=float('nan'))])
def test_invalid_pbt_configuration(patch):
    cfg = asdict(PbtCfg()) | patch
    with pytest.raises(ValueError):
        validate(cfg, 8192)


def test_selection_does_not_collapse_ties_or_protected_explorer():
    cfg = asdict(PbtCfg())
    assert choose_replacement([0.] * 8, cfg) is None
    assert choose_replacement([1., .8, .7, .6, .5, .4, .3, -10.], cfg) == (6, 0)
    assert choose_replacement([0.] * 7 + [2.], cfg) == (0, 7)


def test_success_dominates_bounded_auxiliary_objective():
    failure = dict(success_rate=0., pose_quality=1., coverage=1., failure_rate=0.)
    success = dict(success_rate=1/16, pose_quality=0., coverage=0., failure_rate=1.)
    assert objective(success) > objective(failure)


def test_mutations_are_bounded_reproducible_and_do_not_change_task():
    cfg = asdict(PbtCfg()) | {'mutation_probability': 1.}
    hp = dict(entropy=.01, learning_rate=.001, gamma=.995, std=.6)
    first = mutate(hp, cfg, random.Random(42))
    assert first == mutate(hp, cfg, random.Random(42)) and first != hp
    assert hp['gamma'] == .995
    for name, bounds in [('entropy','entropy_bounds'), ('learning_rate','learning_rate_bounds'),
                         ('gamma','gamma_bounds'), ('std','std_bounds')]:
        assert cfg[bounds][0] <= first[name] <= cfg[bounds][1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA population integration')
def test_gpu_population_update_focus_replacement_resume_and_play(tmp_path):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
    cfg = load_env_cfg(TASK)
    cfg.scene.num_envs = 16
    agent = asdict(load_rl_cfg(TASK))
    agent['num_steps_per_env'] = 4
    agent['pbt'].update(evaluation_envs=2, evaluation_episode_s=.04,
                        warmup_iterations=100, focus_after_iterations=1)
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, device='cuda:0'), clip_actions=1.)
    try:
        runner = ReconfigurationPbtRunner(env, copy.deepcopy(agent), str(tmp_path), 'cuda:0')
        assert len(runner.members) == 8
        assert runner.members[0].storage.num_envs == 2
        assert len({alg.optimizer for alg in runner.members}) == 8
        assert len({next(alg.get_policy().parameters()).data_ptr() for alg in runner.members}) == 8
        assert len({round(parameters(alg)['std'], 2) for alg in runner.members}) >= 4
        runner.learn(1)
        # Demonstrate a real donor/recipient exchange with populated optimizer state.
        runner.pbt['warmup_iterations'] = 1
        runner.pbt['focus_after_iterations'] = 100
        metrics = [dict(success_rate=0., failure_rate=0., pose_quality=i/8,
                        coverage=.1, descriptor=[i/8]*6, score=i/8) for i in range(8)]
        before_donor = copy.deepcopy(runner.members[7].get_policy().state_dict())
        with torch.no_grad(): runner._select(metrics)
        for name, value in before_donor.items():
            if not name.startswith('distribution.'):
                torch.testing.assert_close(runner.members[0].get_policy().state_dict()[name], value)
        assert runner.members[0].storage.step == 0
        runner.pbt['focus_after_iterations'] = 1
        with torch.no_grad(): runner._select(metrics)
        assert runner.focused and len(runner.members) == 4
        assert all(alg.storage.num_envs == 4 for alg in runner.members)
        assert sum(s.stop-s.start for s in runner.slices) == 16
        # New storage and model tensors must support another backward pass.
        runner.pbt['warmup_iterations'] = 100
        runner.learn(1)
        checkpoint = tmp_path / 'population.pt'
        saved = torch.load(checkpoint, weights_only=False, map_location='cuda:0')
        restored = ReconfigurationPbtRunner(env, copy.deepcopy(agent), None, 'cuda:0')
        restored.load(str(checkpoint))
        assert restored.focused and len(restored.members) == 4
        assert restored.next_iteration == runner.next_iteration == 2
        assert restored.total_env_steps == 128
        for alg, model, hp in zip(restored.members, saved['members'], saved['parameters'], strict=True):
            assert parameters(alg) == pytest.approx(hp)
            for name, value in model['actor_state_dict'].items():
                torch.testing.assert_close(alg.get_policy().state_dict()[name], value)
            assert alg.optimizer.state  # moments preserved during restore
        restored.learn(1)
        assert restored.next_iteration == 3 and restored.total_env_steps == 192
        # Exported best/model files use the ordinary single-policy playback path.
        play_cfg = load_env_cfg(TASK, play=True)
        play = RslRlVecEnvWrapper(ManagerBasedRlEnv(play_cfg, device='cuda:0'), clip_actions=1.)
        try:
            playback = ReconfigurationPbtRunner(play, copy.deepcopy(agent), None, 'cuda:0')
            assert not playback.population_mode
            playback.load(str(tmp_path / 'best.pt'), load_cfg={'actor': True})
            obs = play.get_observations()
            with torch.inference_mode():
                action = playback.get_inference_policy()(obs)
            assert action.shape == (1, 6) and torch.isfinite(action).all()
        finally:
            play.close()
    finally:
        env.close()


def test_final_evaluation_cannot_bypass_warmup():
    runner = object.__new__(ReconfigurationPbtRunner)
    runner.pbt = asdict(PbtCfg())
    runner.next_iteration = 1
    runner.members = [object() for _ in range(8)]
    runner.writer = runner.log_path = None
    runner.best_score = -float('inf')
    runner.focused = False
    events = []
    runner._event = events.append
    metrics = [dict(score=float(i)) for i in range(8)]
    runner._select(metrics)
    assert runner.best_member == 7
    assert not runner.focused
    assert [event['event'] for event in events] == ['evaluation']


def test_basic_copy_has_independent_identity_and_no_population_training():
    from mjlab.tasks.registry import load_runner_cls
    from zbot_rl_mjlab.experiments.cards import load_card
    basic_task = TASK.removesuffix('-PBT') + '-Basic'
    basic, population = load_env_cfg(basic_task), load_env_cfg(TASK)
    basic_agent, population_agent = load_rl_cfg(basic_task), load_rl_cfg(TASK)
    assert load_runner_cls(basic_task) is None
    assert not hasattr(basic_agent, 'pbt')
    assert basic_agent.algorithm.schedule == 'adaptive'
    assert population_agent.pbt.enabled
    assert basic_agent.experiment_name != population_agent.experiment_name
    assert basic.scene.num_envs == population.scene.num_envs == 8192
    assert basic.actions['joint_pos'].target_limit == load_card('reconfiguration_head_tail_docking_basic')['env']['actions']['joint_pos']['target_limit']
    assert basic.allowed_dock_quarters == population.allowed_dock_quarters == (0,1,2,3)
    assert not hasattr(basic.actions['joint_pos'], 'escape_assist')
    assert not hasattr(population.actions['joint_pos'], 'escape_assist')
    assert basic.scene.entities['robot'].init_state == population.scene.entities['robot'].init_state
    assert 'connected' in basic.rewards and 'connected' in population.rewards
    assert not basic.commands and population.docking_mode == 'registered_weld'
    assert basic.rewards['success'].weight == load_card('reconfiguration_head_tail_docking_basic')['rewards']['success']['weight']
    assert 'limit_proximity' in basic.rewards and 'limit_proximity' not in population.rewards
    assert 'body_height' not in basic.rewards and 'body_height' in population.rewards
    card = load_card('reconfiguration_head_tail_docking_basic')
    assert 'pbt' not in card['agent']
    basic.rewards['end_distance'].weight = 123.
    assert population.rewards['end_distance'].weight != 123.

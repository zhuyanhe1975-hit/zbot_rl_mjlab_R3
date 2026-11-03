import random
from dataclasses import asdict

import numpy as np
import pytest
import torch

from zbot_rl_mjlab.closure_policy import ClosureBetaDistribution,control_phase_advantages


def test_beta_actions_match_executed_mean_and_have_finite_likelihood():
    torch.manual_seed(7)
    distribution=ClosureBetaDistribution(3)
    raw=torch.tensor([[[2.,5.,-3.],[5.,2.,4.]]])
    expanded=raw.expand(30000,-1,-1)
    distribution.update(expanded)
    samples=distribution.sample()
    assert (samples.abs()<1.).all()
    torch.testing.assert_close(samples.mean(0),distribution.deterministic_output(raw)[0],atol=.01,rtol=0.)
    assert torch.isfinite(distribution.log_prob(samples)).all()
    assert float(distribution.entropy.mean())==pytest.approx(float(-distribution.log_prob(samples).mean()),abs=.02)
    torch.testing.assert_close(distribution.kl_divergence(distribution.params,distribution.params),torch.zeros(30000))
    exported=distribution.as_deterministic_output_module()
    torch.testing.assert_close(exported(raw),distribution.deterministic_output(raw))
    extreme=torch.tensor([[[10000.,-10000.,0.],[-10000.,10000.,0.]]])
    distribution.update(extreme)
    assert torch.isfinite(distribution.log_prob(distribution.sample())).all()
    assert distribution.std.max()<=1/5**.5


def test_actor_advantage_ignores_uncontrollable_phase_outliers():
    raw=torch.tensor([1.,3.,1e6,-1e6]).reshape(2,2,1)
    active=torch.tensor([True,True,False,False]).reshape_as(raw)
    torch.testing.assert_close(control_phase_advantages(raw,active).flatten(),torch.tensor([-2.,2.,0.,0.]))
    assert not control_phase_advantages(raw,torch.zeros_like(active)).any()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_real_ppo_update_and_independent_validator_preserve_training_state(tmp_path):
    import warp as wp
    from mjlab.rl import RslRlVecEnvWrapper
    from scripts.train_refined_gpu_initial_closure import InitialClosureRunner,make_agent_cfg,make_initial_closure_cfg
    from zbot_rl_mjlab.closure_validation import ClosureValidator
    from zbot_rl_mjlab.tasks.reconfiguration.full_batched_fsm import P
    from zbot_rl_mjlab.tasks.reconfiguration.full_batched_gpu_env import FullBatchedGpuEnv
    torch.set_num_threads(1)
    env=FullBatchedGpuEnv(make_initial_closure_cfg(4),device='cuda:0')
    previous=torch.cuda.current_stream(env.device)
    torch.cuda.set_stream(torch.cuda.ExternalStream(wp.get_stream(env.sim.wp_device).cuda_stream,device=env.device))
    validator=None
    try:
        wrapped=RslRlVecEnvWrapper(env)
        cfg=make_agent_cfg(1);cfg.num_steps_per_env=4;cfg.algorithm.num_learning_epochs=1;cfg.algorithm.num_mini_batches=1
        runner=InitialClosureRunner(wrapped,asdict(cfg),None,device=env.device,
                                    reference_sha256='test',evaluation_interval=0)
        def update(curl):
            wrapped.reset()
            if curl:env._disconnect_state.fsm.phase.fill_(P['CURL'])
            env.observation_manager.compute(update_history=True)
            obs=wrapped.get_observations();runner.alg.train_mode()
            with torch.inference_mode():
                for _ in range(4):
                    action=runner.alg.act(obs)
                    obs,reward,done,extras=wrapped.step(action)
                    runner.alg.process_env_step(obs,reward,done,extras)
                runner.alg.compute_returns(obs)
            return runner.alg.update()
        loss=update(True)
        assert loss['actor_active_fraction']==1.
        assert all(np.isfinite(v) for v in loss.values())
        assert runner.alg.learning_rate==.0003
        assert all(group['lr']==.0003 for group in runner.alg.optimizer.param_groups)
        before=[p.detach().clone() for p in runner.alg.actor.parameters()]
        critic_before=[p.detach().clone() for p in runner.alg.critic.parameters()]
        loss=update(False)
        assert loss['actor_active_fraction']==0.
        for p,old in zip(runner.alg.actor.parameters(),before):torch.testing.assert_close(p,old,rtol=0,atol=0)
        assert any(not torch.equal(p,old) for p,old in zip(runner.alg.critic.parameters(),critic_before))
        runner.logger.log_dir=str(tmp_path)
        checkpoint=tmp_path/'model_0.pt';runner.save(str(checkpoint))
        states={name:getattr(env.sim.data,name)._tensor.clone() for name in ('qpos','qvel','eq_active')}
        normalizer={k:v.clone() for k,v in runner.alg.actor.obs_normalizer.state_dict().items()}
        cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state()
        py_rng=random.getstate();np_rng=np.random.get_state()
        validator=ClosureValidator(runner,make_initial_closure_cfg,make_agent_cfg)
        result=validator.check(checkpoint)
        assert result['successes']+result['failures']==8 and result['normalizer_unchanged']
        for name,old in states.items():torch.testing.assert_close(getattr(env.sim.data,name)._tensor,old,rtol=0,atol=0)
        for k,v in normalizer.items():torch.testing.assert_close(runner.alg.actor.obs_normalizer.state_dict()[k],v,rtol=0,atol=0)
        assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(cuda_rng,torch.cuda.get_rng_state())
        assert py_rng==random.getstate() and np.array_equal(np_rng[1],np.random.get_state()[1])
        assert (tmp_path/'deterministic_evaluation.json').is_file()
        with torch.inference_mode():
            obs,_,_,_=wrapped.step(torch.zeros((4,7),device=env.device))
            assert torch.isfinite(obs['actor']).all()
    finally:
        if validator is not None:validator.close()
        env.close();torch.cuda.set_stream(previous)

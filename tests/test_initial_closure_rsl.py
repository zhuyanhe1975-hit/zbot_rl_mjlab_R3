import pytest
import torch

from scripts.train_refined_gpu_initial_closure import make_agent_cfg,InitialClosureRunner,prepare_fresh_log_dir
from scripts.train_refined_gpu_incremental import make_online_initial_closure_cfg as make_initial_closure_cfg
from scripts.view_initial_closure_rsl import find_checkpoint
from scripts.eval_initial_closure_rsl import evaluate,select_successful_checkpoint


def test_conservative_ppo_and_bounded_policy_config():
    cfg=make_agent_cfg(100)
    assert cfg.num_steps_per_env==24
    assert cfg.actor.hidden_dims==(256,128) and cfg.actor.activation=='elu'
    assert cfg.actor.obs_normalization and cfg.critic.obs_normalization
    assert cfg.actor.distribution_cfg['class_name'].endswith(':ClosureBetaDistribution')
    assert cfg.actor.distribution_cfg['initial_concentration']==6.
    assert cfg.save_interval==10
    assert cfg.algorithm.num_learning_epochs==3 and cfg.algorithm.num_mini_batches==4
    assert cfg.algorithm.learning_rate==.0003 and cfg.algorithm.schedule=='fixed'
    assert cfg.algorithm.clip_param==.1
    assert cfg.algorithm.gamma==.997 and cfg.algorithm.lam==.97
    assert cfg.algorithm.entropy_coef==0.


def test_viewer_selects_highest_numeric_checkpoint(tmp_path):
    for name in ('model_9.pt','model_50.pt','model_100.pt','model_latest.pt'):
        (tmp_path/name).touch()
    assert find_checkpoint(tmp_path).name=='model_100.pt'
    with pytest.raises(FileNotFoundError):find_checkpoint(tmp_path/'missing')


def test_deterministic_selection_rejects_fast_failures_and_verifies_best(tmp_path):
    import hashlib
    import json
    rows=[{'worlds':16,'successes':15,'unfinished':0,'mean_success_time_s':1.,'iteration':10},
          {'worlds':16,'successes':16,'unfinished':0,'mean_success_time_s':6.,'iteration':20},
          {'worlds':16,'successes':16,'unfinished':0,'mean_success_time_s':5.8,'iteration':30}]
    assert select_successful_checkpoint(rows) is rows[2]
    assert select_successful_checkpoint(rows[:1]) is None
    (tmp_path/'model_40.pt').touch()
    best=tmp_path/'best.pt';best.write_bytes(b'checked policy')
    report={'selected':{'sha256':hashlib.sha256(best.read_bytes()).hexdigest()}}
    (tmp_path/'deterministic_evaluation.json').write_text(json.dumps(report))
    assert find_checkpoint(tmp_path)==best
    best.write_bytes(b'replaced, unevaluated policy')
    assert find_checkpoint(tmp_path).name=='model_40.pt'
    (tmp_path/'deterministic_evaluation.json').write_text('{"selected": null}')
    with pytest.raises(FileNotFoundError,match='passed deterministic validation'):
        find_checkpoint(tmp_path)


def test_fresh_run_preserves_previous_logs_and_checkpoints(tmp_path):
    log_dir=tmp_path/'fresh'
    assert prepare_fresh_log_dir(log_dir) is None
    log_dir.mkdir()
    assert prepare_fresh_log_dir(log_dir) is None
    (log_dir/'model_100.pt').write_bytes(b'previous checkpoint')
    (log_dir/'events').write_text('previous metrics')
    archived=prepare_fresh_log_dir(log_dir)
    assert not log_dir.exists()
    assert (archived/'model_100.pt').read_bytes()==b'previous checkpoint'
    assert (archived/'events').read_text()=='previous metrics'
    log_dir.mkdir()
    (log_dir/'events').touch()
    assert prepare_fresh_log_dir(log_dir)!=archived
    assert archived.exists()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_native_rsl_wiring_and_compact_observations(tmp_path):
    import warp as wp
    from dataclasses import asdict
    from mjlab.rl import RslRlVecEnvWrapper
    from zbot_rl_mjlab.tasks.reconfiguration.full_batched_gpu_env import FullBatchedGpuEnv
    env=FullBatchedGpuEnv(make_initial_closure_cfg(2),device='cuda:0')
    previous=torch.cuda.current_stream(env.device)
    torch.cuda.set_stream(torch.cuda.ExternalStream(wp.get_stream(env.sim.wp_device).cuda_stream,device=env.device))
    try:
        with torch.inference_mode():
            wrapped=RslRlVecEnvWrapper(env)
            obs=wrapped.get_observations()
            assert obs['actor'].shape==(2,52) and torch.isfinite(obs['actor']).all()
            assert env.action_manager.get_term('joint_torque').reference is None
            runner=InitialClosureRunner(wrapped,asdict(make_agent_cfg(1)),None,device='cuda:0',reference_sha256='test')
            actor=runner.alg.get_policy()
            torch.testing.assert_close(actor(obs),torch.zeros((2,7),device=env.device))
            actions=runner.alg.act(obs)
            assert (actions.abs()<1.).all()
            torch.testing.assert_close(actor.output_std,torch.full((2,7),1/13**.5,device=env.device))
            checkpoint=tmp_path/'model_0.pt';runner.save(str(checkpoint))
            runner.load(str(checkpoint),map_location=env.device)
            torch.testing.assert_close(actor(obs),torch.zeros((2,7),device=env.device))
            legacy=torch.load(checkpoint,weights_only=False)
            legacy['infos']['task_mode']='initial_closure_rsl_walking_2ms_v1'
            old=tmp_path/'legacy.pt';torch.save(legacy,old)
            with pytest.raises(ValueError,match='online-jacobian v11'):
                runner.load(str(old),map_location=env.device)
            assert actions.shape==(2,7) and torch.isfinite(actions).all()
            next_obs,reward,done,_=wrapped.step(actions)
            assert next_obs['actor'].shape==(2,52) and torch.isfinite(reward).all()
            assert not done.any()
            result=evaluate(runner,wrapped,checkpoint,400)
            # A short evaluation should report unfinished worlds, not failures.
            assert result['successes']==0 and result['failures']==0 and result['unfinished']==2
            assert result['normalizer_unchanged']
            components=result['reward_components']
            assert components['time']==0
            assert components['pose_nearness']>0
            assert components['penalty_failure']==0
            assert components['penalty_elapsed_time']<0
            assert result['mean_return']==pytest.approx(sum(components.values()),abs=1e-4)
    finally:
        env.close();torch.cuda.set_stream(previous)

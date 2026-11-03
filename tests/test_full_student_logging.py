from zbot_rl_mjlab.full_student_logging import format_rsl_log


def sample(done=0):
    return dict(iteration=3,final_iteration=500,total_steps=524288,fps=20000,collection_stream_s=20.,learning_stream_s=.5,
        value_loss=.1,surrogate_loss=.2,entropy=.3,done=done,mean_episode_reward=12.5 if done else None,
        mean_episode_length=45. if done else None,action_std=.135,success=0,failure=done,phase_timeout=done,
        peak_connector_loads_N=[1.,2.,3.,4.,5.,6.],invalid_load_world_steps=0,wall_s=21.,elapsed_s=80.,eta_s=400.)

def test_rsl_format_no_completed_episodes():
    text=format_rsl_log(sample())
    assert 'Learning iteration 3/500' in text and '524,288' in text
    assert 'n/a (no completed episodes)' in text and '1.0 2.0 3.0 4.0 5.0 6.0' in text
    assert 'Collection time (GPU stream)' in text

def test_rsl_format_reports_measured_episode_statistics():
    text=format_rsl_log(sample(done=2))
    assert '12.50' in text and '45.00' in text and '0 / 2' in text
    assert 'Phase timeout (subset of failure)' in text

def test_rsl_format_closure_specific_metrics():
    data=sample(done=2)
    data.update(mean_success_time_s=5.64,mean_pose_reward_rate=.125,
                policy_samples=1234,peak_other_N=43.98)
    text=format_rsl_log(data)
    assert 'Mean successful closure time:' in text and '5.640s' in text
    assert 'Mean pose reward rate:' in text and '0.12500' in text
    assert 'Policy samples (CURL):' in text and '1,234' in text
    assert 'Peak non-target load (N):' in text and '44.0' in text

def test_rsl_format_reference_and_policy_comparison():
    data=sample(done=2)
    data.update(reference_success=1,policy_success=1,mean_reference_time_s=5.7,mean_policy_time_s=5.6,
                mean_pose_progress_rate=.02,mean_abs_curl_residual_rad_s=.003,mean_curl_clock_scale=1.01)
    text=format_rsl_log(data)
    assert 'Reference closure time:' in text and '5.700s (1 episodes)' in text
    assert 'Policy closure time:' in text and '5.600s (1 episodes)' in text
    assert 'Mean pose progress rate:' in text and '0.02000' in text
    assert 'Mean CURL clock scale:' in text and '1.01000' in text

def test_nonfinite_loads_are_counted_without_poisoning_other_values():
    import torch
    from zbot_rl_mjlab.tasks.reconfiguration.full_batched_gpu_env import record_peak_loads
    values=torch.tensor([[.5,float('nan'),.1,0.,0.,0.],[.2,.75,float('inf'),0.,0.,0.]])
    peak=torch.zeros(6);invalid=torch.zeros((),dtype=torch.long)
    record_peak_loads(values,peak,invalid,80.)
    torch.testing.assert_close(peak,torch.tensor([40.,60.,8.,0.,0.,0.]))
    assert invalid.item()==2

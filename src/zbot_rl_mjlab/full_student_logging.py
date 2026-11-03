"""RSL-RL style presentation for the custom full-student PPO metrics."""
from datetime import timedelta

def format_rsl_log(report):
    width=80;pad=40
    def row(label,value):return f'{label:>{pad}} {value}'
    items=['#'*width,f" Learning iteration {report['iteration']}/{report['final_iteration']} ".center(width),'',
           row('Total steps:',f"{report['total_steps']:,}"),row('Steps per second:',f"{report['fps']:.0f}"),
           row('Collection time (GPU stream):',f"{report['collection_stream_s']:.3f}s"),
           row('Learning time (GPU stream):',f"{report['learning_stream_s']:.3f}s"),
           row('Mean value function loss:',f"{report['value_loss']:.4f}"),
           row('Mean surrogate loss:',f"{report['surrogate_loss']:.4f}"),
           row('Mean policy entropy:',f"{report['entropy']:.4f}")]
    if report['done']:
        items.extend((row('Mean reward (completed episodes):',f"{report['mean_episode_reward']:.2f}"),
                      row('Mean episode length:',f"{report['mean_episode_length']:.2f}")))
    else:items.append(row('Mean reward / episode length:','n/a (no completed episodes)'))
    if report.get('mean_success_time_s') is not None:
        items.append(row('Mean successful closure time:',f"{report['mean_success_time_s']:.3f}s"))
    if 'mean_pose_reward_rate' in report:
        items.append(row('Mean pose reward rate:',f"{report['mean_pose_reward_rate']:.5f}"))
    if 'mean_pose_progress_rate' in report:
        items.append(row('Mean pose progress rate:',f"{report['mean_pose_progress_rate']:.5f}"))
    if 'reference_success' in report:
        ref=report.get('mean_reference_time_s');policy=report.get('mean_policy_time_s')
        if 'reference_failure' in report:
            items.append(row('Reference success / failure:',f"{report['reference_success']} / {report['reference_failure']}"))
            items.append(row('Policy success / failure:',f"{report['policy_success']} / {report['policy_failure']}"))
        items.append(row('Reference closure time:',f"{ref:.3f}s ({report['reference_success']} episodes)" if ref is not None else 'n/a'))
        items.append(row('Policy closure time:',f"{policy:.3f}s ({report['policy_success']} episodes)" if policy is not None else 'n/a'))
    if report.get('mean_abs_curl_residual_rad_s') is not None:
        items.append(row('Mean abs residual speed (rad/s):',f"{report['mean_abs_curl_residual_rad_s']:.5f}"))
        items.append(row('Mean CURL clock scale:',f"{report['mean_curl_clock_scale']:.5f}"))
    if 'policy_samples' in report:
        items.append(row('Policy samples (CURL):',f"{report['policy_samples']:,}"))
    if 'peak_other_N' in report:
        items.append(row('Peak non-target load (N):',f"{report['peak_other_N']:.1f}"))
    items.extend((row('Mean action std:',f"{report['action_std']:.3f}"),
                  row('Episodes: success / failure:',f"{report['success']} / {report['failure']}"),
                  row('Phase timeout (subset of failure):',str(report['phase_timeout'])),
                  row('Peak connector loads 1-6 (N):',' '.join(f'{x:.1f}' for x in report['peak_connector_loads_N'])),
                  row('Non-finite load world-steps:',str(report['invalid_load_world_steps'])),
                  '-'*width,row('Iteration time:',f"{report['wall_s']:.2f}s"),
                  row('Time elapsed:',str(timedelta(seconds=int(report['elapsed_s'])))),
                  row('ETA:',str(timedelta(seconds=int(report['eta_s']))))))
    return '\n'.join(items)

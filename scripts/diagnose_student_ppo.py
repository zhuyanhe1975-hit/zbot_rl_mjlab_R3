"""Export every TensorBoard scalar and fixed-command validation without training."""

import argparse
import csv
import json
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def diagnose(run: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    events = EventAccumulator(str(run), size_guidance={'scalars': 0}).Reload()
    series = {tag: events.Scalars(tag) for tag in events.Tags()['scalars']}
    with (output / 'scalars.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('tag', 'step', 'value', 'wall_time'))
        for tag, points in series.items():
            writer.writerows((tag, point.step, point.value, point.wall_time) for point in points)
    windows = ((0, 10), (10, 35), (125, 150), (275, 300))
    with (output / 'curve_windows.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('tag', *(f'{start}:{end}' for start, end in windows)))
        for tag, points in series.items():
            if 'validation' in tag or tag.endswith('/time'):
                continue
            means = []
            for start, end in windows:
                values = [point.value for point in points if start <= point.step < end]
                means.append(sum(values) / len(values) if values else '')
            writer.writerow((tag, *means))
    reports = [json.loads(path.read_text()) for path in run.glob('validation_*.json')
               if path.name != 'validation_best.json']
    reports.sort(key=lambda report: report['completed_updates'])
    keys = sorted({key for report in reports for key in report['metrics']})
    with (output / 'validation.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('completed_updates', 'checkpoint', *keys))
        for report in reports:
            writer.writerow((report['completed_updates'], report['checkpoint'],
                             *(report['metrics'].get(key, '') for key in keys)))
    metadata = {'run': str(run.resolve()), 'scalar_tags': len(series),
                'validation_reports': len(reports),
                'ppo_policy_kl_tags': [tag for tag in series if 'kl' in tag.lower() and 'reference' not in tag],
                'value_prediction_tags': [tag for tag in series if 'value_mean' in tag],
                'note': 'Episode_Reward values aggregate tasks; they are not per-task reward ablations.'}
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    plot_curves(series, reports, output)
    print(json.dumps(metadata, indent=2))


def plot_curves(series, reports, output):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 2, figsize=(13, 12), constrained_layout=True)
    panels = (
        ('Episode reward contributions (all tasks)', ['Episode_Reward/' + name for name in (
            'velocity_tracking', 'joint_standing', 'base_ascent_velocity',
            'transition_settle_target', 'reset_failure')]),
        ('PPO loss / frozen-student KL', ['Loss/value', 'Loss/surrogate',
                                        'Loss/student_ppo_reference_task0_kl',
                                        'Loss/student_ppo_reference_task2_kl']),
        ('Reference correction', ['Loss/student_ppo_reference_coefficient',
                                  'Loss/student_ppo_reference_loss']),
        ('Actual rollout task fractions', ['Loss/task_' + name + '_frame_fraction'
                                           for name in ('biped', 'snake', 'descending', 'ascending')]),
    )
    for axis, (title, tags) in zip(axes.flat, panels):
        for tag in tags:
            if tag in series:
                axis.plot([point.step for point in series[tag]],
                          [point.value for point in series[tag]], label=tag.split('/')[-1])
        axis.set_title(title)
    for index, (title, keys) in enumerate((
        ('Fixed-command success', ('biped_backward_success', 'snake_moving_success',
                                   'ascent_handoff_fraction', 'biped_forward_success')),
        ('Joint speed peaks (rad/s)', ('snake_forward_peak_joint_speed', 'ascent_peak_joint_speed')),
    )):
        axis = axes[2, index]
        for key in keys:
            axis.plot([report['completed_updates'] for report in reports],
                      [report['metrics'][key] for report in reports], marker='.', label=key)
        axis.set_title(title)
    for axis in axes.flat:
        axis.set_xlabel('Update (TensorBoard uses zero-based step)')
        axis.grid(alpha=.25)
        axis.legend(fontsize=8)
    figure.savefig(output / 'curves.png', dpi=150)
    plt.close(figure)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    diagnose(args.run, args.output)

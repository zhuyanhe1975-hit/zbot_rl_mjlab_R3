"""Deployable-sensor student distilled from the frozen universal teacher."""

from dataclasses import asdict

from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg

from ..student import StudentDAggerCfg, StudentRunnerCfg, UniversalStudentRunner
from ..tasks.universal.guided_observations import local_reference_privileged_state
from ..tasks.universal.student_observations import (
    GuidedStudentEnvCfg,
    student_sensor_state,
)
from ..tasks.universal.student_rewards import (
    student_touchdown_flat,
    student_velocity_tracking,
)
from ..tasks.universal.teacher_rewards import teacher_action_rate
from .omni_teacher import SnakeLegOmniGuidedTeacher, guided_teacher_ppo_cfg
from .universal import universal_env_cfg


class UniversalOmniStudent(SnakeLegOmniGuidedTeacher):
    task_id = 'Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Student'
    experiment_name = 'zbot_universal_snake_leg_omni_student'
    runner_cls = UniversalStudentRunner
    max_iterations = 500

    def create_env(self, play=False):
        return universal_env_cfg(play=play, cfg_cls=GuidedStudentEnvCfg)

    def _build_ppo(self) -> StudentRunnerCfg:
        baseline = guided_teacher_ppo_cfg()
        baseline.actor.class_name = 'zbot_rl_mjlab.student_models:HistoryGaussianStudentMLP'
        assert baseline.actor.distribution_cfg is not None
        baseline.actor.distribution_cfg['learn_std'] = False
        algorithm = StudentDAggerCfg(learning_rate=3e-4, max_grad_norm=1.)
        values = asdict(baseline)
        values.update(class_name='zbot_rl_mjlab.student:UniversalStudentRunner',
                      experiment_name=self.experiment_name, max_iterations=500,
                      obs_groups={'actor': ('actor',), 'critic': ('teacher',)})
        # Preserve typed dataclass children for MJLab's CLI and YAML serialization.
        values.update(actor=baseline.actor, critic=baseline.critic, algorithm=algorithm)
        values.pop('console_mode', None)
        values.pop('validation_interval_iterations', None)
        values.pop('robust_validation_interval', None)
        return StudentRunnerCfg(**values)

    def edit(self, cfg, play):
        # Corrections belong to the student environment, never the frozen teacher.
        cfg.rewards['action_rate'].func = teacher_action_rate
        cfg.rewards['velocity_tracking'].func = student_velocity_tracking
        cfg.rewards['velocity_tracking'].params['snake_std'] = (.05, .05, .15)
        cfg.rewards['foot_flat_touchdown'] = RewardTermCfg(
            func=student_touchdown_flat, weight=-.2,
            params={'sigma': .05, 'sensor_name': 'feet_ground_contact'},
        )
        cfg.student_observation_noise = not play
        cfg.observations = {
            'actor': ObservationGroupCfg(
                {'sensors': ObservationTermCfg(func=student_sensor_state)},
                history_length=cfg.student_history_length, flatten_history_dim=True,
                concatenate_terms=True, enable_corruption=False, nan_policy='disabled',
            ),
            'teacher': ObservationGroupCfg(
                {'privileged': ObservationTermCfg(func=local_reference_privileged_state)},
                concatenate_terms=True, enable_corruption=False, nan_policy='disabled',
            ),
        }

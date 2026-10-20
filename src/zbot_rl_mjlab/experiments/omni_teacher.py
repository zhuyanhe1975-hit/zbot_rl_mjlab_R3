"""单个特权教师从头联合学习两种全向运动及双向变形。"""

from dataclasses import asdict

from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from ..grouped_teacher import GroupedTeacherRunner, grouped_teacher_ppo_cfg
from ..guided_teacher import GuidedTeacherRunner, guided_teacher_ppo_cfg
from ..omni_teacher import OmniTeacherRunner, omni_teacher_ppo_cfg
from ..task_teacher import (
    RoutedTaskTeacherRunner,
    TaskTeacherRunner,
    routed_task_teacher_ppo_cfg,
    task_teacher_ppo_cfg,
)
from ..tasks.universal.actions import SnakeCrawlJointTargetRateActionCfg
from ..tasks.universal.env_cfg import universal_env_cfg
from ..tasks.universal.guided_rewards import (
    guided_action_rate,
    guided_joint_energy,
    guided_standstill,
)
from ..tasks.universal.guided_transitions import (
    GuidedOmniCommandCfg,
    GuidedTransitionActionCfg,
    guided_privileged_state,
    guided_transition_failure,
)
from ..tasks.universal.omni import reset_omni
from ..tasks.universal.privileged import (
    PrivilegedOmniCommandCfg,
    privileged_state,
    training_skill_complete,
)
from ..tasks.universal.teacher_rewards import (
    ArchiveOmniTeacherEnvCfg,
    configure_teacher_rewards,
)
from .addons import DomainRandomization
from .base import Experiment
from .universal import StudentDisturbance


class SnakeLegOmniTeacher(Experiment):
    task_id = 'Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Teacher'
    experiment_name = 'zbot_universal_snake_leg_omni_teacher'
    max_iterations = 6000
    train_envs = 8192
    supports_walking_addons = False
    runner_cls = OmniTeacherRunner
    addons = (
        # 成功受扰行走归档：足部摩擦、基座COM、±5N/±0.5Nm。
        # 变形归档无外扰：起步从0渐增，不额外叠加未经该归档验证的质量/PD随机化。
        StudentDisturbance(
            force_range=(-5., 5.), torque_range=(-.5, .5), disturbed_fraction=.25,
            duration_s=(.08, .2), cooldown_s=(.8, 2.), pretraining_scale=0., ramp_iterations=500,
        ),
        DomainRandomization(encoder_bias_range=None),
    )

    def build_ppo(self):
        cfg = omni_teacher_ppo_cfg()
        cfg.experiment_name = self.experiment_name
        if self.max_iterations is not None:
            cfg.max_iterations = self.max_iterations
        return cfg

    def create_env(self, play=False):
        return universal_env_cfg(play=play, cfg_cls=ArchiveOmniTeacherEnvCfg)

    def edit(self, cfg, play):
        if play:
            # Keep the initial morphology until a selected robot receives a key.
            # Automatic resets would otherwise randomly assign a new morphology.
            cfg.auto_reset = False
        cfg.omni_site_frame = True
        cfg.motion_site = 'foot_0_sole'
        action = SnakeCrawlJointTargetRateActionCfg(**asdict(cfg.actions['joint_pos']))
        action.biped_residual_scale = .25
        action.transition_end_joint_radius = 1.
        cfg.actions['joint_pos'] = action
        cfg.commands['transform'] = PrivilegedOmniCommandCfg(
            training=not play, mode='source',
            debug_vis=play,
            resampling_time_range=(6., 12.),
        )
        cfg.events['reset_transition'].func = reset_omni
        cfg.events['reset_transition'].params = {}
        # 两个网络读取同一个actor组，避免每步重复构建特权状态。
        cfg.observations = {
            'actor': ObservationGroupCfg(
                {'privileged': ObservationTermCfg(func=privileged_state)},
                concatenate_terms=True, enable_corruption=False, nan_policy='disabled',
            ),
        }
        configure_teacher_rewards(cfg)
        # Episode-reset averages bias task occupancy toward short failed trials.
        # The runner reports whole-rollout counts and tracking errors instead.
        cfg.metrics = {}
        cfg.terminations['skill_complete'] = TerminationTermCfg(func=training_skill_complete, time_out=True)
        if play:
            # Viewers keep stepping without servicing manual-reset terminal states.
            # Continuous play ends only when the user resets or closes the viewer.
            cfg.terminations.clear()


class SnakeLegOmniWideBipedTeacher(SnakeLegOmniTeacher):
    """Biped-only control before re-enabling joint skill training."""

    task_id = SnakeLegOmniTeacher.task_id + '-Wide-Biped'
    experiment_name = 'zbot_universal_snake_leg_omni_teacher_wide_biped'

    def edit(self, cfg, play):
        # The experiment builder runs base edits before this hook.
        action = cfg.actions['joint_pos']
        assert isinstance(action, SnakeCrawlJointTargetRateActionCfg)
        action.biped_residual_scale = .375
        command = cfg.commands['transform']
        assert isinstance(command, PrivilegedOmniCommandCfg)
        command.task_observation_encoding = 'scalar'
        # Fixed mode bypasses mixed-role allocation and transformation sampling.
        # Resets start standing; resampling changes only vx/vy/wz.
        command.mode = 'stand'


class SnakeLegOmniGroupedTeacher(SnakeLegOmniWideBipedTeacher):
    """Reward-component critics for joint skills, isolated from the biped baseline."""

    task_id = SnakeLegOmniTeacher.task_id + '-Multi-Critic'
    experiment_name = 'zbot_universal_snake_leg_omni_teacher_multi_critic'
    runner_cls = GroupedTeacherRunner

    def build_ppo(self):
        cfg = grouped_teacher_ppo_cfg()
        cfg.experiment_name = self.experiment_name
        if self.max_iterations is not None:
            cfg.max_iterations = self.max_iterations
        return cfg

    def edit(self, cfg, play):
        command = cfg.commands['transform']
        assert isinstance(command, PrivilegedOmniCommandCfg)
        command.mode = 'source'


class SnakeLegOmniTaskTeacher(SnakeLegOmniGroupedTeacher):
    """Independent full-reward task critics; retain each archive's reward balance."""

    task_id = SnakeLegOmniTeacher.task_id + '-Task-Critic'
    experiment_name = 'zbot_universal_snake_leg_omni_teacher_task_critic'
    runner_cls = TaskTeacherRunner

    def build_ppo(self):
        cfg = task_teacher_ppo_cfg()
        cfg.experiment_name = self.experiment_name
        if self.max_iterations is not None:
            cfg.max_iterations = self.max_iterations
        return cfg


class SnakeLegOmniTaskRoutedTeacher(SnakeLegOmniTaskTeacher):
    task_id = SnakeLegOmniTeacher.task_id + '-Task-Routed'
    experiment_name = 'zbot_universal_snake_leg_omni_teacher_task_routed'
    runner_cls = RoutedTaskTeacherRunner

    def build_ppo(self):
        cfg = routed_task_teacher_ppo_cfg()
        cfg.experiment_name = self.experiment_name
        if self.max_iterations is not None:
            cfg.max_iterations = self.max_iterations
        return cfg

    def edit(self, cfg, play):
        command = cfg.commands['transform']
        assert isinstance(command, PrivilegedOmniCommandCfg)
        # Start inside the demonstrated low-speed region and include the .05 play command.
        command.velocity_limits = (.15, .08, .4)
        command.motion_axis_min_fraction = .1
        command.play_random_velocity = play
        # Validated demo speed; .05 remains an unlearned dead zone in the bounded trial.
        command.play_velocity = (.15, 0., 0.)
        command.forward_speed_levels = (.03, .05, .08, .1, .15)
        command.direction_speed_multipliers = (1., 1.5)


class SnakeLegOmniGuidedTeacher(SnakeLegOmniTaskRoutedTeacher):
    """Privileged task PPO and frozen archival guidance, jointly from update zero."""

    task_id = SnakeLegOmniTeacher.task_id + '-Guided'
    experiment_name = 'zbot_universal_snake_leg_omni_teacher_guided'
    runner_cls = GuidedTeacherRunner

    def build_ppo(self):
        cfg = guided_teacher_ppo_cfg()
        cfg.experiment_name = self.experiment_name
        if self.max_iterations is not None:
            cfg.max_iterations = self.max_iterations
        return cfg

    def edit(self, cfg, play):
        command = cfg.commands['transform']
        assert isinstance(command, PrivilegedOmniCommandCfg)
        # Give both transformations the same frame allocation as locomotion.
        command.training_task_fractions = (.25, .25, .25, .25)
        command.transition_timeout_s = 16.
        # The archived transformations used their seven task rewards, without
        # universal posture shaping, power tax, or action-rate regularization.
        cfg.rewards.pop('mode_posture')
        cfg.rewards.pop('transition_progress')
        cfg.rewards['joint_energy'].func = guided_joint_energy
        cfg.rewards['action_rate'].func = guided_action_rate
        cfg.rewards['standstill'].func = guided_standstill
        cfg.commands['transform'] = GuidedOmniCommandCfg(**asdict(command))
        cfg.actions['joint_pos'] = GuidedTransitionActionCfg(**asdict(cfg.actions['joint_pos']))
        cfg.observations['actor'].terms['privileged'].func = guided_privileged_state
        if not play:
            cfg.terminations['transition_failure'].func = guided_transition_failure

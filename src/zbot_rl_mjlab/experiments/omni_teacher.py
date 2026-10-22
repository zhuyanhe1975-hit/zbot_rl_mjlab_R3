"""单个特权教师从头联合学习两种全向运动及双向变形。"""


from dataclasses import asdict, fields
from typing import cast

from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlPpoAlgorithmCfg
from mjlab.sensor import TerrainHeightSensorCfg

from ..grouped_teacher import GroupedTeacherRunner
from ..guided_ppo import GuidedTaskPpoCfg
from ..guided_teacher import GuidedTeacherRunner, GuidedTeacherRunnerCfg
from ..omni_teacher import OmniTeacherRunner, OmniTeacherRunnerCfg
from ..task_teacher import (
    RoutedTaskTeacherRunner,
    TaskTeacherRunner,
)
from ..tasks.universal import teacher_rewards, teacher_snake_rewards, teacher_transition
from ..tasks.universal.actions import SnakeCrawlJointTargetRateActionCfg
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
)
from .addons import DomainRandomization
from .base import Experiment
from .universal import StudentDisturbance, universal_env_cfg
from .walking import env_cfg as walking_env_cfg


def omni_teacher_ppo_cfg() -> OmniTeacherRunnerCfg:
    """PPO及网络容量取成功归档；v2动作与网络须使用新run。"""
    return OmniTeacherRunnerCfg(
        class_name="zbot_rl_mjlab.omni_teacher:OmniTeacherRunner",
        actor=RslRlModelCfg(
            hidden_dims=(256, 128), activation="elu", obs_normalization=True,
            distribution_cfg={"class_name": "BetaDistribution", "action_range": (-1.0, 1.0)},
        ),
        critic=RslRlModelCfg(hidden_dims=(256, 128), activation="elu", obs_normalization=True),
        # bests/transform_snake_biped_bidirectional_2026-10-17_iter599/agent.yaml
        # 与snake_to_biped_teacher_iter200相同；固定字段避免上游默认值漂移。
        algorithm=RslRlPpoAlgorithmCfg(
            class_name='PPO', num_learning_epochs=5, num_mini_batches=4,
            learning_rate=3e-4, schedule='adaptive', gamma=.99, lam=.95,
            entropy_coef=.005, desired_kl=.01, max_grad_norm=1.,
            value_loss_coef=1., use_clipped_value_loss=True, clip_param=.2,
            normalize_advantage_per_mini_batch=False, optimizer='adam', share_cnn_encoders=False,
        ),
        obs_groups={"actor": ("actor",), "critic": ("actor",)},
        experiment_name="zbot_universal_snake_leg_omni_teacher",
        max_iterations=6000, num_steps_per_env=24, save_interval=50, logger="tensorboard",
    )


def grouped_teacher_ppo_cfg() -> OmniTeacherRunnerCfg:
    """Keep the successful PPO recipe while replacing its value estimator."""
    cfg = omni_teacher_ppo_cfg()
    cfg.class_name = "zbot_rl_mjlab.grouped_teacher:GroupedTeacherRunner"
    cfg.algorithm.class_name = "zbot_rl_mjlab.grouped_ppo:GroupedPPO"
    cfg.critic.class_name = "zbot_rl_mjlab.grouped_ppo:IndependentRewardCritics"
    return cfg


def task_teacher_ppo_cfg() -> OmniTeacherRunnerCfg:
    cfg = omni_teacher_ppo_cfg()
    cfg.class_name = 'zbot_rl_mjlab.task_teacher:TaskTeacherRunner'
    cfg.algorithm.class_name = 'zbot_rl_mjlab.task_ppo:TaskPPO'
    cfg.critic.class_name = 'zbot_rl_mjlab.task_ppo:IndependentTaskCritics'
    return cfg


def routed_task_teacher_ppo_cfg() -> OmniTeacherRunnerCfg:
    cfg = task_teacher_ppo_cfg()
    cfg.class_name = 'zbot_rl_mjlab.task_teacher:RoutedTaskTeacherRunner'
    cfg.actor.class_name = 'zbot_rl_mjlab.task_models:RoutedTaskNormalizedMLP'
    cfg.critic.class_name = 'zbot_rl_mjlab.task_models:TaskNormalizedCritics'
    return cfg


def guided_teacher_ppo_cfg() -> GuidedTeacherRunnerCfg:
    baseline = routed_task_teacher_ppo_cfg()
    algorithm = GuidedTaskPpoCfg(**(asdict(baseline.algorithm) | {
        'class_name': 'zbot_rl_mjlab.guided_ppo:GuidedTaskPPO',
    }))
    values = {item.name: getattr(baseline, item.name) for item in fields(baseline) if item.name != 'algorithm'}
    return GuidedTeacherRunnerCfg(**(values | {
        'class_name': 'zbot_rl_mjlab.guided_teacher:GuidedTeacherRunner', 'algorithm': algorithm,
    }))


def configure_teacher_rewards(cfg):
    """Install archival units once, retaining the current joint-task objectives."""
    for name in ('energy', 'contact_impact', 'failure'):
        cfg.rewards.pop(name, None)
    cfg.rewards.update({
        'mode_posture': RewardTermCfg(func=teacher_rewards.transition_posture, weight=2.),
        'transition_progress': RewardTermCfg(func=teacher_rewards.transition_progress, weight=2.),
        'velocity_tracking': RewardTermCfg(func=teacher_rewards.teacher_velocity_tracking, weight=1., params={'std': (.1, .1, .25), 'axis_weights': (2., 1., 1.)}),
        'biped_upright': RewardTermCfg(func=teacher_rewards.biped_upright, weight=.2),
        'joint_limits': RewardTermCfg(func=teacher_rewards.biped_joint_limits, weight=-.1),
        'joint_acceleration': RewardTermCfg(func=teacher_rewards.biped_joint_acceleration, weight=-.0001),
        'soft_landing': RewardTermCfg(func=teacher_rewards.biped_soft_landing, weight=-.02, params={'sensor_name': 'feet_ground_contact'}),
        'swing_clearance': RewardTermCfg(func=teacher_rewards.biped_swing_clearance, weight=.5, params={'height_sensor_name': 'foot_height_scan', 'target': .05, 'std': .04}),
        'action_rate': RewardTermCfg(func=teacher_rewards.teacher_action_rate, weight=-.01),
        'joint_standing': RewardTermCfg(func=teacher_rewards.transition_pose, weight=12.),
        'base_height_progress': RewardTermCfg(func=teacher_rewards.transition_height, weight=1.),
        'base_ascent_velocity': RewardTermCfg(func=teacher_rewards.transition_ascent, weight=8., params={'target_speed': .15, 'pose_error_range': (.2, .8)}),
        'sole_ground_alignment': RewardTermCfg(func=teacher_rewards.transition_sole, weight=1., params={'angle_power': 1.}),
        'reset_failure': RewardTermCfg(func=teacher_rewards.transition_failure_cost, weight=-10.),
        'stand_finish_joint_speed': RewardTermCfg(func=teacher_rewards.transition_finish_speed, weight=-.02),
        'descent_ground_impact': RewardTermCfg(func=teacher_rewards.transition_descent_impact, weight=-2., params={'max_body_weights': 2.}),
        'joint_energy': RewardTermCfg(func=teacher_rewards.teacher_joint_energy, weight=-.5, params={'energy_scale': 10.}),
        'snake_vertical_speed': RewardTermCfg(func=teacher_snake_rewards.snake_vertical_speed, weight=-.2),
        'snake_joint_deviation': RewardTermCfg(func=teacher_snake_rewards.snake_joint_deviation, weight=-.02),
        'snake_target_saturation': RewardTermCfg(func=teacher_snake_rewards.snake_target_saturation, weight=-.1),
        'support_foot_slip': RewardTermCfg(func=teacher_rewards.biped_support_foot_slip, weight=-5., params={'sensor_name': 'feet_ground_contact', 'site_names': ('foot_0_sole', 'foot_1_sole')}),
        'foot_nonparallel_ground': RewardTermCfg(func=teacher_rewards.biped_foot_nonparallel, weight=-5., params={'sensor_name': 'feet_ground_contact', 'std': .1}),
    })
    cfg.events['reset_archive_ascent_memory'] = EventTermCfg(func=teacher_rewards.reset_ascent_memory, mode='reset')
    cfg.terminations['biped_fallen'] = TerminationTermCfg(func=teacher_rewards.biped_fallen, params={'height': .22, 'tilt': .75})
    cfg.terminations['transition_failure'] = TerminationTermCfg(func=teacher_transition.transition_failure)
    cfg.events['reset_transition_failure_memory'] = EventTermCfg(func=teacher_transition.reset_transition_failure_memory, mode='reset')
    for sensor in cfg.scene.sensors:
        if sensor.name == 'feet_ground_contact':
            sensor.track_air_time = True
            sensor.history_length = cfg.decimation
    if not any(sensor.name == 'foot_height_scan' for sensor in cfg.scene.sensors):
        height = cast(TerrainHeightSensorCfg, next(sensor for sensor in walking_env_cfg().scene.sensors if sensor.name == 'foot_height_scan'))
        height.debug_vis = False
        cfg.scene.sensors = (*cfg.scene.sensors, height)
    return cfg


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

    def _build_ppo(self):
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

    def _build_ppo(self):
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

    def _build_ppo(self):
        cfg = task_teacher_ppo_cfg()
        cfg.experiment_name = self.experiment_name
        if self.max_iterations is not None:
            cfg.max_iterations = self.max_iterations
        return cfg


class SnakeLegOmniTaskRoutedTeacher(SnakeLegOmniTaskTeacher):
    task_id = SnakeLegOmniTeacher.task_id + '-Task-Routed'
    experiment_name = 'zbot_universal_snake_leg_omni_teacher_task_routed'
    runner_cls = RoutedTaskTeacherRunner

    def _build_ppo(self):
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

    def _build_ppo(self):
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

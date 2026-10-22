"""A separate PPO phase initialized from a supervised sensor-history student."""

from dataclasses import asdict, fields

from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg

from ..student_ppo import (
    StudentPPOCfg,
    StudentPPORunnerCfg,
    UniversalStudentPPORunner,
)
from ..tasks.universal import mdp, student_ppo_rewards
from ..tasks.universal.student_ppo_env import (
    StudentPPOCommandCfg,
    StudentPPOEnvCfg,
    reset_student_ppo_state,
    student_training_skill_complete,
)
from .student import UniversalOmniStudent
from .universal import universal_env_cfg


class UniversalOmniStudentPPO(UniversalOmniStudent):
    task_id = UniversalOmniStudent.task_id + '-PPO'
    experiment_name = 'zbot_universal_snake_leg_omni_student_ppo_v2'
    runner_cls = UniversalStudentPPORunner

    def _build_ppo(self):
        baseline = super()._build_ppo()
        baseline.actor.class_name = 'zbot_rl_mjlab.student_ppo:PPOHistoryGaussianStudentMLP'
        assert baseline.actor.distribution_cfg is not None
        baseline.actor.distribution_cfg.update(init_std=.1, rate_init_std=.1, learn_std=False)
        values = {item.name: getattr(baseline, item.name)
                  for item in fields(StudentPPORunnerCfg) if hasattr(baseline, item.name)}
        values.update(
            class_name='zbot_rl_mjlab.student_ppo:UniversalStudentPPORunner',
            experiment_name=self.experiment_name,
            algorithm=StudentPPOCfg(learning_rate=3e-5, schedule='fixed',
                                    clip_param=.1, entropy_coef=0., critic_warmup_iterations=10),
        )
        return StudentPPORunnerCfg(**values)

    def create_env(self, play=False):
        return universal_env_cfg(play=play, cfg_cls=StudentPPOEnvCfg)

    def edit(self, cfg, play):
        # Student-only gates are identical in training and playback.
        cfg.play_transition_sync = True
        cfg.commands['transform'] = StudentPPOCommandCfg(**asdict(cfg.commands['transform']))
        cfg.events['reset_student_ppo_state'] = EventTermCfg(func=reset_student_ppo_state, mode='reset')
        if not play:
            cfg.terminations['skill_complete'].func = student_training_skill_complete
        cfg.rewards['base_ascent_velocity'].func = student_ppo_rewards.student_transition_ascent
        cfg.rewards['base_ascent_velocity'].params['target_speed'] = .10
        cfg.rewards.pop('descent_ground_impact')
        terms = {
            'transition_joint_overspeed': (student_ppo_rewards.transition_joint_overspeed, -1.,
                                          {'ascent_limit': 3., 'descent_limit': 2.5}),
            'transition_settle_target': (student_ppo_rewards.transition_settle_target, -.5,
                                        {'angle_scale': .15, 'near_ground_height': .12, 'near_goal_error': .35}),
            'transition_settle_speed': (student_ppo_rewards.transition_settle_speed, -.1,
                                       {'speed_scale': .5, 'near_ground_height': .12, 'near_goal_error': .35}),
            'transition_settle_action': (student_ppo_rewards.transition_settle_action, -.1,
                                        {'near_ground_height': .12, 'near_goal_error': .35}),
            'transition_descent_speed': (student_ppo_rewards.transition_descent_speed, -.5,
                                        {'max_down_speed': .10, 'approach_height': .15}),
            'transition_ground_impact': (student_ppo_rewards.transition_ground_impact, -2.,
                                        {'max_body_weights': 2.}),
        }
        for name, (func, weight, params) in terms.items():
            cfg.rewards[name] = RewardTermCfg(func=func, weight=weight, params=params)


class UniversalOmniStudentPPOV3(UniversalOmniStudentPPO):
    """Separate opt-in recipe; the registered v2 task keeps its original defaults."""

    task_id = UniversalOmniStudentPPO.task_id + '-V3'
    experiment_name = 'zbot_universal_snake_leg_omni_student_ppo_v3'

    def edit(self, cfg, play):
        tracking = cfg.rewards['velocity_tracking']
        tracking.func = student_ppo_rewards.student_velocity_progress
        tracking.params = {'std': (.1, .1, .25), 'snake_std': (.05, .05, .15)}


class UniversalOmniStudentPPOV4(UniversalOmniStudentPPOV3):
    """Use whole-body progress with the same initialization and safeguards as v3."""

    task_id = UniversalOmniStudentPPO.task_id + '-V4'
    experiment_name = 'zbot_universal_snake_leg_omni_student_ppo_v4'

    def edit(self, cfg, play):
        command = cfg.commands['transform']
        command.snake_velocity_limits = (.06, .06, .5)
        tracking = cfg.rewards['velocity_tracking']
        tracking.func = mdp.smoothed_body_velocity_tracking
        tracking.params = {
            'limits': {
                'biped': tuple(max(abs(low), abs(high)) for low, high in command.biped_velocity_ranges),
                'snake': command.snake_velocity_limits,
            },
            'time_constant': .75,
            'w_lat': .3,
        }

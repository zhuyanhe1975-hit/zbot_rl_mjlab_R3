"""Reconfiguration task family, separate from fixed-topology locomotion."""
from dataclasses import fields

from .base import Experiment
from ..reconfiguration_pbt import ReconfigurationPbtRunner, ReconfigurationRunnerCfg
from ..tasks.reconfiguration.env_cfg import head_tail_env_cfg
from ..planning_guidance import PlanGuidedRunner
from ..jacobian_ppo import JacobianPpoRunner


class ReconfigurationExperiment(Experiment):
    def create_env(self, play=False):
        return head_tail_env_cfg(play)


class HeadTailDockingBasic(ReconfigurationExperiment):
    task_id = 'Mjlab-Zbot-6dof-Reconfigure-Head-Tail-Docking-Basic'
    experiment_name = 'zbot_reconfiguration_head_tail_docking_basic'
    max_iterations = 500

    def create_env(self, play=False):
        from ..tasks.reconfiguration.basic.env_cfg import head_tail_env_cfg as initial_env_cfg
        return initial_env_cfg(play)


class HeadTailDockingPBT(ReconfigurationExperiment):
    task_id = 'Mjlab-Zbot-6dof-Reconfigure-Head-Tail-Docking-PBT'
    experiment_name = 'zbot_reconfiguration_head_tail_docking_pbt'
    max_iterations = 500
    runner_cls = ReconfigurationPbtRunner

    def _build_ppo(self):
        cfg = super()._build_ppo()
        values = {field.name: getattr(cfg, field.name) for field in fields(cfg)}
        values['class_name'] = 'zbot_rl_mjlab.reconfiguration_pbt:ReconfigurationPbtRunner'
        return ReconfigurationRunnerCfg(**values)


class HeadTailDockingPlanGuided(HeadTailDockingBasic):
    task_id = 'Mjlab-Zbot-6dof-Reconfigure-Head-Tail-Docking-Plan-Guided'
    experiment_name = 'zbot_reconfiguration_head_tail_docking_plan_guided'

    runner_cls = PlanGuidedRunner

    def create_env(self, play=False):
        from ..tasks.reconfiguration.plan_guided import plan_guided_env_cfg
        return plan_guided_env_cfg(play)

    def _build_ppo(self):
        from ..planning_guidance import PlanImitationCfg, PlanGuidedRunnerCfg
        cfg = super()._build_ppo()
        cfg.actor.distribution_cfg = None
        values = {f.name: getattr(cfg, f.name) for f in fields(cfg)}
        values['algorithm'] = PlanImitationCfg()
        values['class_name'] = 'zbot_rl_mjlab.planning_guidance:PlanGuidedRunner'
        return PlanGuidedRunnerCfg(**values)


class DisconnectTorque(ReconfigurationExperiment):
    task_id = 'Mjlab-Zbot-6dof-Reconfigure-Disconnect-Torque'
    experiment_name = 'zbot_reconfiguration_disconnect_torque'
    runner_cls = JacobianPpoRunner
    max_iterations = 500

    def create_env(self, play=False):
        from ..tasks.reconfiguration.disconnect.env_cfg import disconnect_env_cfg
        return disconnect_env_cfg(play,native_pd=True)

    def _build_ppo(self):
        from ..jacobian_ppo import JacobianPpoRunnerCfg,ResidualPpoCfg
        cfg = super()._build_ppo()
        cfg.algorithm=ResidualPpoCfg(**{f.name:getattr(cfg.algorithm,f.name) for f in fields(cfg.algorithm)})
        cfg.algorithm.class_name='zbot_rl_mjlab.jacobian_ppo:ResidualMaskedPPO'
        cfg.actor.distribution_cfg['std_range']=(.0002,.02)
        values={f.name:getattr(cfg,f.name) for f in fields(cfg)}
        values['class_name']='zbot_rl_mjlab.jacobian_ppo:JacobianPpoRunner'
        return JacobianPpoRunnerCfg(**values)

"""Frozen archival guidance for privileged task PPO; legacy adapters stay unchanged.

The optional site snake uses its native 56D Beta contract for signed commands.
Historical world-X snake guidance remains a separate forward-only opt-in.
"""

import math
from types import SimpleNamespace

import torch
from mjlab.utils.lab_api.math import quat_mul
from tensordict import TensorDict

from .tasks.snake import mdp as snake_mdp
from .tasks.transformation.actions import expand_mirror, project_mirror
from .tasks.walking import mdp as walking_mdp
from .teachers import TeacherEnsemble, TeacherTargets, _joint_values, position_to_rate


class _View:
    def __init__(self, source, **overrides):
        self._source = source
        self.__dict__.update(overrides)

    def __getattr__(self, name):
        return getattr(self._source, name)


class _SceneView:
    def __init__(self, scene, robot):
        self._scene, self._robot = scene, robot

    def __getitem__(self, name):
        return self._robot if name == 'robot' else self._scene[name]

    def __getattr__(self, name):
        return getattr(self._scene, name)


def _forward_covered(value):
    return (torch.isfinite(value).all(-1) & (value[:, 0] > .025)
            & (value[:, 1:].abs().amax(-1) < .01))


class GuidedTeacherEnsemble(TeacherEnsemble):
    """Task-matched omni walking, transformation and optional snake guidance."""

    def __init__(self, env, device=None, checkpoints=None, include_omni=True, include_snake=False, include_site_snake=False):
        if include_snake and include_site_snake:
            raise ValueError('cannot enable both historical and site snake teachers')
        names = ((('omni',) if include_omni else ()) + ('transform',)
                 + (('snake',) if include_snake else ())
                 + (('snake_site',) if include_site_snake else ()))
        super().__init__(env, device, checkpoints, enabled_teachers=names)
        if not include_snake:
            return
        self._guided_proxy = SimpleNamespace()
        self._guided_episode = torch.full_like(env.episode_length_buf, -1)
        self._guided_task = torch.full_like(env.episode_length_buf, -1)
        self._guided_supported = torch.zeros(env.num_envs, device=self.device, dtype=torch.bool)
        self._guided_heading = env.scene['robot'].data.joint_pos.new_zeros(env.num_envs, 2)
        self._guided_heading[:, 0] = 1.
        self._guided_stamp = None

    def _site_snake_observation(self):
        """Run the restored 56D recipe on an isolated native snake command view."""
        from .tasks.universal.omni import site_heading
        from .tasks.universal.privileged import privileged_state

        env = self.env
        command = env.command_manager.get_term('transform')
        cfg = self.configs['snake_site']
        if not hasattr(self, '_site_proxy'):
            n, reference = env.num_envs, env.scene['robot'].data.joint_pos
            self._site_command = SimpleNamespace(
                cfg=cfg.commands['transform'], teacher_cfg=cfg.commands['transform'],
                standing_mode=torch.zeros(n, device=self.device, dtype=torch.bool),
                current_morphology=torch.full((n,), -1, device=self.device, dtype=torch.long),
                current_task=torch.full((n,), 2, device=self.device, dtype=torch.long),
                phase=torch.full((n,), 2, device=self.device, dtype=torch.long),
                command=reference.new_full((n, 1), -1.),
                gait_frequency=reference.new_full((n,), cfg.commands['transform'].snake_gait_frequency),
                heading=reference.new_tensor((1., 0.)).expand(n, 2).clone(),
                initial_heading=reference.new_tensor((1., 0.)).expand(n, 2).clone(),
                heading_initialized=torch.zeros(n, device=self.device, dtype=torch.bool),
            )
            self._site_proxy = SimpleNamespace()
            self._site_episode = torch.full_like(env.episode_length_buf, -1)
            self._site_task = torch.full_like(command.current_task, -1)
            self._site_stamp = None
        native, proxy = self._site_command, self._site_proxy
        stamp = (env.common_step_counter, getattr(env, '_observation_generation', 0))
        fresh = ((env.episode_length_buf < self._site_episode) | (self._site_episode < 0)
                 | (command.current_task != self._site_task))
        if stamp != self._site_stamp:
            fresh |= (env.episode_length_buf == 0) & (self._site_episode == 0)
        native.heading_initialized[fresh] = False
        self._site_episode.copy_(env.episode_length_buf)
        self._site_task.copy_(command.current_task)
        self._site_stamp = stamp
        proxy.cfg, proxy.scene, proxy.sim = cfg, env.scene, env.sim
        proxy.device, proxy.num_envs, proxy.step_dt = env.device, env.num_envs, env.step_dt
        proxy.common_step_counter, proxy.episode_length_buf = env.common_step_counter, env.episode_length_buf
        proxy.command_manager = SimpleNamespace(get_term=lambda name: native)
        action = env.action_manager.get_term('joint_pos')
        native_action = SimpleNamespace(
            commanded_target=action.commanded_target,
            raw_action=(action.raw_action * action.rate_cfg.max_target_speed
                        / cfg.actions['joint_pos'].max_target_speed).clamp(-1, 1),
        )
        proxy.action_manager = SimpleNamespace(get_term=lambda name: native_action)
        heading = site_heading(proxy)
        source = command.heading / command.heading.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        left = torch.stack((-source[:, 1], source[:, 0]), -1)
        native_left = torch.stack((-heading[:, 1], heading[:, 0]), -1)

        def in_site_frame(value):
            world = source * value[:, :1] + left * value[:, 1:2]
            return torch.stack(((world * heading).sum(-1), (world * native_left).sum(-1), value[:, 2]), -1)

        native.velocity_command = in_site_frame(command.velocity_command)
        native.requested_velocity = in_site_frame(command.requested_velocity)
        return privileged_state(proxy)

    def _convert(self, key, output, env_ids=None):
        if key == 'transform' and getattr(getattr(self.env, 'cfg', None), 'mirrored_transform_actions', False):
            source_speed = self.configs['transform'].actions['joint_pos'].max_target_speed
            target_speed = self.env.action_manager.get_term('joint_pos').rate_cfg.max_target_speed
            return expand_mirror(output * source_speed / target_speed)
        if key != 'omni':
            return super()._convert(key, output, env_ids=env_ids)
        if getattr(getattr(self.env, 'cfg', None), 'native_biped_actions', False):
            # Both policies now use the archived Gaussian position action itself.
            return output
        action = self.env.action_manager.get_term('joint_pos')
        residual_scale = getattr(action.rate_cfg, 'biped_residual_scale', None)
        if residual_scale is None or not math.isfinite(residual_scale) or residual_scale <= 0:
            raise ValueError('omni guidance requires a positive biped_residual_scale')
        data = self.env.scene['robot'].data
        nominal = getattr(data, 'default_joint_pos', None)
        if nominal is None:
            nominal = self.defaults['omni'].expand(self.env.num_envs, -1)
        elif hasattr(action, '_target_ids'):
            nominal = nominal[:, action._target_ids]
        limits = action._target_limits()
        if env_ids is not None:
            nominal = nominal[env_ids]
            if limits.shape[0] != 1:
                limits = limits[env_ids]
        cfg = self.configs['omni'].actions['joint_pos']
        if cfg.clip is not None:
            raise ValueError('omni position teacher clip semantics changed')
        teacher_scale = _joint_values(cfg.scale, nominal)
        offset = self.defaults['omni'] if cfg.use_default_offset else _joint_values(cfg.offset, nominal)
        desired = output * teacher_scale + offset
        desired = desired.clamp(min=nominal - residual_scale, max=nominal + residual_scale)
        desired = desired.clamp(min=limits[..., 0], max=limits[..., 1])
        # Task 0 integrates no velocity: the learner reconstructs this absolute
        # goal from nominal + residual_scale * action, then applies its slew limit.
        return ((desired - nominal) / residual_scale).clamp(-1, 1)

    def _omni_mask(self, tasks, height):
        env = self.env
        command = env.command_manager.get_term('transform')
        action = env.action_manager.get_term('joint_pos')
        data = env.scene['robot'].data
        gravity = walking_mdp.projected_gravity(env)
        upright = ((gravity[:, 2].abs() < .55) & (height > .16)
                   & ((data.joint_pos - self.defaults['omni']).abs().amax(-1) < 1.2)
                   & ((action.commanded_target - self.defaults['omni']).abs().amax(-1) < 1.2))
        velocity = self._walking_velocity_command()
        # Restrict labels to the command range actually used by this archive.
        ranges = self.configs['omni'].commands['target_speed'].ranges
        low = velocity.new_tensor((ranges.lin_vel_x[0], ranges.lin_vel_y[0], ranges.ang_vel_z[0]))
        high = velocity.new_tensor((ranges.lin_vel_x[1], ranges.lin_vel_y[1], ranges.ang_vel_z[1]))
        valid = (torch.isfinite(command.requested_velocity).all(-1)
                 & torch.isfinite(velocity).all(-1)
                 & (velocity >= low - 1e-6).all(-1) & (velocity <= high + 1e-6).all(-1)
                 & torch.isfinite(command.heading).all(-1)
                 & (command.heading.norm(dim=-1) > .1))
        return ((tasks == 0) & command.standing_mode & ~command.transition_mask
                & upright & valid)

    def _canonical_snake_observation(self, tasks, supported):
        env, proxy = self.env, self._guided_proxy
        command = env.command_manager.get_term('transform')
        stamp = (env.common_step_counter, getattr(env, '_observation_generation', 0))
        episode_reset = ((env.episode_length_buf < self._guided_episode)
                         | (self._guided_episode < 0))
        # Two distinct generations at episode step zero can be a partial reset.
        if stamp != self._guided_stamp:
            episode_reset |= (env.episode_length_buf == 0) & (self._guided_episode == 0)
        fresh = episode_reset | (tasks != self._guided_task) | (supported & ~self._guided_supported)
        if fresh.any():
            heading = command.heading
            valid = torch.isfinite(heading).all(-1) & (heading.norm(dim=-1) > .1)
            normalized = heading / heading.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            self._guided_heading.copy_(torch.where((fresh & valid)[:, None], normalized, self._guided_heading))
        self._guided_episode.copy_(env.episode_length_buf)
        self._guided_task.copy_(tasks)
        self._guided_supported.copy_(supported)
        self._guided_stamp = stamp
        yaw = torch.atan2(self._guided_heading[:, 1], self._guided_heading[:, 0])
        rotation = torch.stack(((yaw / 2).cos(), yaw * 0, yaw * 0, -(yaw / 2).sin()), -1)

        def rotate(value):
            c, s = yaw.cos(), yaw.sin()
            while c.ndim < value.ndim - 1:
                c, s = c.unsqueeze(-1), s.unsqueeze(-1)
            return torch.stack((c * value[..., 0] + s * value[..., 1],
                                -s * value[..., 0] + c * value[..., 1], value[..., 2]), -1)

        robot = env.scene['robot']
        data = robot.data
        overrides = {}
        for name in ('body_link_lin_vel_w', 'body_link_ang_vel_w', 'body_com_lin_vel_w', 'site_ang_vel_w'):
            if hasattr(data, name):
                overrides[name] = rotate(getattr(data, name))
        for name in ('body_link_pos_w', 'body_com_pos_w', 'site_pos_w'):
            if hasattr(data, name):
                value = getattr(data, name)
                origin = env.scene.env_origins[:, None, :]
                overrides[name] = rotate(value - origin) + origin
        for name in ('body_link_quat_w', 'site_quat_w'):
            value = getattr(data, name)
            overrides[name] = quat_mul(rotation[:, None, :].expand_as(value), value)
        proxy.cfg = self.configs['snake']
        proxy.scene = _SceneView(env.scene, _View(robot, data=_View(data, **overrides)))
        proxy.sim, proxy.device, proxy.num_envs = env.sim, env.device, env.num_envs
        proxy.step_dt, proxy.common_step_counter = env.step_dt, env.common_step_counter
        # Share the learner-visible gait clock; resetting a private clock on task
        # entry would give contradictory labels for identical policy observations.
        proxy.episode_length_buf = env.episode_length_buf
        action = env.action_manager.get_term('joint_pos')
        native_action = SimpleNamespace(
            commanded_target=action.commanded_target,
            raw_action=(action.raw_action * action.rate_cfg.max_target_speed
                        / proxy.cfg.actions['joint_pos'].max_target_speed).clamp(-1, 1),
        )
        proxy.action_manager = SimpleNamespace(get_term=lambda name: native_action)
        proxy.command_manager = SimpleNamespace(get_term=lambda name: SimpleNamespace(cfg=proxy.cfg.commands['motion']))
        # The archive's reference is task-entry heading, mapped to world +X.
        proxy._snake_initial_yaw = yaw.new_zeros(env.num_envs)
        if fresh.any():
            ids = fresh.nonzero(as_tuple=False).flatten()
            snake_mdp.body_yaw_integral_state(proxy).reset(ids, env.common_step_counter)
            snake_mdp.head_yaw_integral_state(proxy).reset(ids, env.common_step_counter)
        return snake_mdp.actor_state(proxy)

    @torch.no_grad()
    def query(self) -> TeacherTargets:
        env = self.env
        command = env.command_manager.get_term('transform')
        tasks = command.current_task
        action = env.action_manager.get_term('joint_pos')
        data = env.scene['robot'].data
        finite_state = (torch.isfinite(data.joint_pos).all(-1)
                        & torch.isfinite(data.joint_vel).all(-1)
                        & torch.isfinite(action.commanded_target).all(-1)
                        & torch.isfinite(action.raw_action).all(-1))
        mirror_target = expand_mirror(project_mirror(action.commanded_target))
        mirror_joint = expand_mirror(project_mirror(data.joint_pos))
        straightening = getattr(command, 'straightening_mask', None)
        recovering = torch.zeros_like(command.transition_mask)
        if straightening is not None:
            recovering = (tasks == 3) & command.transition_mask & straightening
        transform = (((tasks == 1) | (tasks == 3)) & command.transition_mask
                     & ~recovering
                     & ((action.commanded_target - mirror_target).abs().amax(-1) < .75)
                     & ((data.joint_pos - mirror_joint).abs().amax(-1) < .75))
        from .robot import JOINT_NAMES

        for i, name in enumerate(JOINT_NAMES):
            bounds = self.configs['transform'].actions['joint_pos'].joint_target_bounds.get(name)
            if bounds is not None:
                transform &= ((action.commanded_target[:, i] >= bounds[0] - .5)
                              & (action.commanded_target[:, i] <= bounds[1] + .5))
        base = env.scene['robot'].find_bodies('base')[0][0]
        height = data.body_link_pos_w[:, base, 2] - env.scene.env_origins[:, 2]
        masks = {'transform': transform}
        observation = None
        if 'snake' in self.models:
            supported = ((tasks == 2) & ~command.transition_mask
                         & _forward_covered(command.requested_velocity)
                         & _forward_covered(command.velocity_command)
                         & torch.isfinite(command.heading).all(-1)
                         & (command.heading.norm(dim=-1) > .1))
            observation = self._canonical_snake_observation(tasks, supported)
            proxy, cfg = self._guided_proxy, self.configs['snake']
            limit = min(math.pi / 3, cfg.actions['joint_pos'].target_limit)
            masks['snake'] = (supported & (height < .12) & (snake_mdp.head_height(proxy) < .12)
                             & (snake_mdp.head_tilt(proxy) < cfg.head_tilt_limit)
                             & (snake_mdp.head_forward(proxy)[:, 0] > math.cos(cfg.head_yaw_limit))
                             & (snake_mdp.head_yaw_abs_deviation(proxy) < cfg.head_yaw_limit)
                             & (snake_mdp.body_yaw_abs_deviation(proxy) < cfg.body_yaw_limit)
                             & (data.joint_pos.abs().amax(-1) <= limit + .1)
                             & (action.commanded_target.abs().amax(-1) <= limit))
        site_observation = None
        if 'snake_site' in self.models:
            from .tasks.snake.site_omni import site_height, site_tilt

            site_observation = self._site_snake_observation()
            proxy, cfg = self._site_proxy, self.configs['snake_site']
            limits = data.joint_pos.new_tensor(cfg.commands['transform'].velocity_limits)
            native = self._site_command
            frequency = command.gait_frequency
            valid_command = (torch.isfinite(command.heading).all(-1)
                             & (command.heading.norm(dim=-1) > .1)
                             & torch.isfinite(frequency)
                             & ((frequency - cfg.commands['transform'].snake_gait_frequency).abs() < 1e-6))
            for velocity in (native.velocity_command, native.requested_velocity):
                valid_command &= (torch.isfinite(velocity).all(-1)
                                  & (velocity.abs() <= limits + 1e-6).all(-1))
            limit = cfg.actions['joint_pos'].target_limit
            masks['snake_site'] = ((tasks == 2) & ~command.standing_mode & ~command.transition_mask
                                   & valid_command & (height < .12) & (site_height(proxy) < .12)
                                   & (site_tilt(proxy) <= cfg.site_tilt_limit)
                                   & (data.joint_pos.abs().amax(-1) <= limit + .1)
                                   & (action.commanded_target.abs().amax(-1) <= limit))
        # actor_state sanitizes invalid values; independently reject invalid physics.
        for name in ('body_link_quat_w', 'body_link_pos_w', 'body_link_lin_vel_w',
                     'body_link_ang_vel_w', 'site_pos_w', 'site_quat_w'):
            finite_state &= torch.isfinite(getattr(data, name)).flatten(1).all(-1)
        if 'snake' in self.models:
            finite_state &= torch.isfinite(data.body_com_lin_vel_w).flatten(1).all(-1)
        if 'snake_site' in self.models:
            for name in ('body_com_lin_vel_w', 'body_com_pos_w', 'site_lin_vel_w', 'site_ang_vel_w'):
                finite_state &= torch.isfinite(getattr(data, name)).flatten(1).all(-1)
        result = action.commanded_target.new_zeros(env.num_envs, 6)
        weights = result.new_zeros(env.num_envs)
        ids = torch.full((env.num_envs,), -1, device=self.device, dtype=torch.long)
        counts = {}
        if 'omni' in self.models:
            masks['omni'] = self._omni_mask(tasks, height)
        for teacher_id, name in enumerate(self.teacher_names):
            mask = masks[name]
            selected = (mask & finite_state).nonzero(as_tuple=False).flatten()
            counts[name] = 0
            if not selected.numel():
                continue
            values = (observation if name == 'snake' else site_observation
                      if name == 'snake_site' else self.observation(name))
            assert values is not None
            inputs = TensorDict({'actor': values[selected]}, batch_size=[selected.numel()])
            output = self.models[name](inputs, stochastic_output=False)
            converted = self._convert(name, output, env_ids=selected)
            valid = (torch.isfinite(values[selected]).all(-1) & torch.isfinite(output).all(-1)
                     & torch.isfinite(converted).all(-1))
            accepted = selected[valid]
            result[accepted], weights[accepted], ids[accepted] = converted[valid], 1., teacher_id
            counts[name] = accepted.numel()
        recovering &= finite_state
        if recovering.any():
            corrective_rates = position_to_rate(
                torch.zeros_like(action.commanded_target), action.commanded_target,
                action._target_limits(), env.step_dt, action.rate_cfg.max_target_speed,
            )
            if getattr(getattr(env, 'cfg', None), 'mirrored_transform_actions', False):
                corrective_rates = expand_mirror(project_mirror(corrective_rates))
            recovering &= torch.isfinite(corrective_rates).all(-1)
            result[recovering] = corrective_rates[recovering]
            weights[recovering] = 1.
            ids[recovering] = self.teacher_names.index('transform')
            counts['transform'] += int(recovering.sum())
        return TeacherTargets(result, weights, counts, ids)

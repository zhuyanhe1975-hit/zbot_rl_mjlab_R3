"""Walking experiment on the mixed-clocked six-module assembly."""
from ..tasks.snake.mixed_clocking import mixed_clocking_spec
from .walking import FlatWalking


class MixedClockingWalking(FlatWalking):
    task_id = 'Mjlab-Zbot-6dof-Walking-Mixed-270-270-180-90-90'
    experiment_name = 'zbot_walking_mixed_270_270_180_90_90'

    def edit(self, cfg, play):
        cfg.scene.entities['robot'].spec_fn = mixed_clocking_spec

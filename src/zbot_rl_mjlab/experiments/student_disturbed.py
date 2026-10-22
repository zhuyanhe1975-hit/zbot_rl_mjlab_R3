"""22.d: the original DAgger student learns under nonzero external impulses."""

from .student import UniversalOmniStudent
from .universal import StudentDisturbance


class UniversalOmniStudentDisturbed(UniversalOmniStudent):
    task_id = UniversalOmniStudent.task_id + '-Disturbed'
    experiment_name = 'zbot_universal_snake_leg_omni_student_disturbed'
    max_iterations = 150
    # DAgger has no PPO iteration clock. Full scale also during teacher warmup
    # keeps inherited pretraining gating from silently disabling disturbances.
    addons = (StudentDisturbance(pretraining_scale=1., ramp_iterations=150),)

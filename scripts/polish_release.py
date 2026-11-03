"""Start internal separation on the substep of passive release."""

def enable_immediate_release(f):
    previous_measure = f.measure
    def measure():
        previous_measure()
        if f.phase == 'LOAD' and not f.connected()[f.target]:
            # The original FSM waits until its next 20 ms tick. Here the
            # motion planner starts from the exact post-release state instead.
            try:
                f.prepare_internal()
            except (RuntimeError, ValueError) as exc:
                f.stop(f'immediate release planning failed: {exc}')
    f.measure = measure

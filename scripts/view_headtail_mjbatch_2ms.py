"""2 ms forward/reverse experiment: 1/2 continue, Enter resets current mode.

Uses native online unfolding and no extra 20% load margin. The 100 N passive
rupture rule and the other physical protections remain active.
"""
import scripts.view_headtail_mjbatch_refined as refined

viewer=refined.viewer
original_config=viewer.config
original_create=refined.create_refined


def config_2ms():
    cfg=original_config();cfg.dt=.002
    return cfg


def create_2ms(direction):
    f=original_create(direction)
    try:
        f.cfg.speed_deg_s*=.5;f.cfg.acceleration_deg_s2*=.25
        f.cfg.curl_speed_multiplier*=.5 if direction=='forward' else .7
        f.cfg.stiffness=100.;f.cfg.outer_stiffness=40.
        # Keep the actual 2 cm minimum separation; add tracking allowance to the path.
        f.planner.gap=.026
        f.set_pd()
        return f
    except BaseException:
        f.close();raise


def reverse_from_current_2ms(previous):
    f=refined.reverse_from_current_refined(previous)
    # Model target headroom for the continuous 2 ms state; the physical
    # connector still breaks at 100 N and the ramp is previewed as before.
    f.cfg.loading_model_headroom=15.
    return f


def install():
    refined.NON_TARGET_LOAD_LIMIT=100.
    refined.USE_RECORDED_UNFOLD=False
    viewer.config=config_2ms
    refined.create_refined=create_2ms
    viewer.create=create_2ms
    viewer.reverse_from_current=reverse_from_current_2ms
    viewer.forward_from_current=refined.forward_from_current_refined


if __name__=='__main__':
    install()
    raise SystemExit(viewer.main())

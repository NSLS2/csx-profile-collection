from bluesky.preprocessors import monitor_during_wrapper

monitor_dets = [
    stemp.temp.A.T,
    stemp.temp.B.T,
    epu1.gap.readback,
    pgm.energy.setpoint,
    theta.user_readback,
    sy.readback,
    pgm.mir_pit.user_readback,  # nanopsignal.bx.svolt,
    fccd.stats4.total,
    fccd.stats4.sigma,
    fccd.stats4.centroid.x,
    fccd.stats4.centroid.y,
]


def ct_xpcs(monitor_dets, _extramd_dict=None):
    # TODO - add exception handling in the right way so monitor_dets has correct setting.
    # setup to not crash bsui
    # kind_states = silent_monitor_kind(monitor_dets, set_all_normal=True)

    # the business
    if _extramd_dict is None:
        yield from monitor_xpcs(count([fccd]), monitor_dets)
    elif type(_extramd_dict) == dict:
        yield from monitor_xpcs(count([fccd], md=_extramd_dict), monitor_dets)

    # return to roginal state
    # revert_monitor_kind(monitor_dets, kind_states)

import functools
from typing import Optional, Any
from collections.abc import Sequence, Mapping

import numpy as np
from cycler import cycler
import bluesky.plans as bp
import bluesky.plan_stubs as bps
from bluesky.protocols import Readable, Movable
from bluesky.plan_stubs import TakeReading, trigger_and_read, move_per_step
from bluesky.utils import MsgGenerator


def mv_with_retry(*args, num_retries: int = 3) -> MsgGenerator[None]:
    """
    Move a motor with retries.

    Parameters
    ----------
    *args : Any
        Arguments to pass to the `bps.mv` plan stub.
    num_retries : int, optional
        The number of retries to attempt. Defaults to 3.
    """
    for i in range(num_retries):
        try:
            yield from bps.mv(*args)
        except Exception as e:
            print(f"Error moving with arguments {args}: {e}. Retrying {i + 1} of {num_retries}.")
            if i == num_retries - 1:
                raise e


def one_nd_step_with_retries(
    detectors: Sequence[Readable],
    step: Mapping[Movable, Any],
    pos_cache: Mapping[Movable, Any],
    take_reading: Optional[TakeReading] = None,
    num_retries: int = 3,
) -> MsgGenerator[None]:
    take_reading = trigger_and_read if take_reading is None else take_reading
    motors = step.keys()
    for i in range(num_retries):
        try:
            yield from move_per_step(step, pos_cache)
        except Exception as e:
            print(f"Error moving per step: {e}. Retrying {i + 1} of {num_retries}.")
            if i == num_retries - 1:
                raise e
    yield from take_reading(list(detectors) + list(motors))


def scan_with_retry(scan_plan, *args, num_retries: int = 3, **kwargs) -> MsgGenerator[Any]:
    """
    Scan with retries.

    Parameters
    ----------
    scan_plan : callable
        The Bluesky scan plan to run. It must have a `per_step` keyword argument.
    *args : Any
        Additional arguments to pass to the scan plan.
    num_retries : int, optional
        The number of retries to attempt for each step of the scan. Defaults to 3.
    **kwargs : Any
        Additional keyword arguments to pass to the scan plan.
    """
    per_step = functools.partial(one_nd_step_with_retries, num_retries=num_retries)
    return (yield from scan_plan(*args, per_step=per_step, **kwargs))


def spiral_continuous(detectors,
                      x_motor, y_motor, x_start, y_start, npts,
                      probe_size, overlap=0.8,  *,
                      tilt=0.0, per_step=None, y_over_x_ratio=1.0, md=None):
    '''Continuously increasing radius spiral scan.

    centered around (x_start, y_start) which is generic regarding
    motors and detectors.

    Parameters
    ----------
    x_motor : object
        any 'setable' object (motor, etc.)
    y_motor : object
        any 'setable' object (motor, etc.)
    x_start : float
        x center
    y_start : float
        y center
    npts : integer
        number of points
    probe_size : float
        radius of probe in units of motors
    overlap : float
        fraction of probe overlap
    y_over_x_ratio : float, optional
        the ratio of the y / x distortion. Default: 1.0 (no distortion).

    ----------------------------------------------------------------
    Not implemented yet:
    tilt : float, optional (not yet enabled)
        Tilt angle in radians, default = 0.0

    per_step : callable, optional
        hook for cutomizing action of inner loop (messages per step)
        See docstring of bluesky.plans.one_nd_step (the default) for
        details.
    ----------------------------------------------------------------
    md : dict, optional
        metadata

    '''
    # #TODO clean up pattern args and _md.  Do not remove motors from _md.
    pattern_args = dict(x_motor=x_motor, y_motor=y_motor,
                        x_start=x_start, y_start=y_start, npts=npts,
                        probe_size=probe_size, overlap=overlap,
                        tilt=tilt)

    # cyc = plan_patterns.spiral(**pattern_args)# - leftover from spiral.

    bxs = []
    bzs = []

    bx_init = x_start
    bz_init = y_start

    for i in range(0, npts):
        R = np.sqrt(i/np.pi)
        # this is to get the first point to be the center
        T = 2*i/(R+0.0000001)
        bx = (overlap*probe_size*R * np.cos(T)) + bx_init
        bz = (overlap*probe_size*R * np.sin(T) * y_over_x_ratio) + bz_init
        bxs.append(bx)
        bzs.append(bz)

    motor_vals = [bxs, bzs]
    x_range = max(motor_vals[0]) - min(motor_vals[0])
    y_range = max(motor_vals[1]) - min(motor_vals[1])
    motor_pos = cycler(x_motor, bxs) + cycler(y_motor, bzs)

    # Before including pattern_args in metadata, replace objects with reprs.
    pattern_args['x_motor'] = repr(x_motor)
    pattern_args['y_motor'] = repr(y_motor)
    _md = {'plan_args': {'detectors': list(map(repr, detectors)),
                         'x_motor': repr(x_motor), 'y_motor': repr(y_motor),
                         'x_start': x_start, 'y_start': y_start,
                         'overlap': overlap,  # 'nth': nth,
                         'tilt': tilt,
                         'per_step': repr(per_step)},
           'extents': tuple([[x_start - x_range, x_start + x_range],
                             [y_start - y_range, y_start + y_range]]),
           'plan_name': 'spiral_continuous',
           'plan_pattern': 'spiral_continuous',
           'plan_pattern_args': pattern_args,
           # - leftover from spiral.
           # 'plan_pattern_module': plan_patterns.__name__,
           'hints': {}}

    try:
        dimensions = [(x_motor.hints['fields'], 'primary'),
                      (y_motor.hints['fields'], 'primary')]
    except (AttributeError, KeyError):
        pass
    else:
        _md['hints'].update({'dimensions': dimensions})
    _md.update(md or {})

    cont_sp_plan = bp.scan_nd(detectors, motor_pos, per_step=per_step, md=_md)

    reset_plan = bps.mv(x_motor, x_start, y_motor, y_start)

    def plan_steps():
        yield from cont_sp_plan
        print('Moving back to first point position.')
        yield from reset_plan

    try:
        return (yield from plan_steps())

    except Exception:
        # Catch the exception long enough to clean up.
        print('Moving back to first point position.')
        yield from reset_plan
        raise

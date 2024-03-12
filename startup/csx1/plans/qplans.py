import numpy as np
from cycler import cycler
from ..startup.optics import pgm
from ..startup.endstation import (delta, theta, gamma)
from ..startup.tardis import tardis
import bluesky.plan_stubs as bps
from bluesky.plans import scan_nd


def EfixQapprox(detectors, E_start, E_end, npts, E_shift=0, *,
                per_step=None, md=None):
    '''Approximate fixed Q energy scan based on delta, theta positions

    for CSX-1 mono (pgm.energy)

    Parameters
    ----------
    E_start : float
        starting energy [eV]
    E_stop : float
        stoping energy [eV]
    npts : integer
        number of points
    E_shift : float
        shift in energy calibration relative to orientation matrix
        (i.e, ) - not used currently
    per_step : callable, optional
        hook for cutomizing action of inner loop (messages per step)
        See docstring of bluesky.plans.one_nd_step (the default) for
        details.
    md : dict, optional
        metadata

    '''
    x_motor = pgm.energy  # This is CSX-1's mono motor name for energy
    x_start = E_start
    pattern_args = dict(x_motor=x_motor, x_start=E_start, npts=npts)

    deltas = []
    thetas = []
    deltaCALC = 0
    thetaCALC = 0

    E_init = x_motor.setpoint.get()
    lam_init = 12398/E_init
    delta_init = delta.user_setpoint.get()
    theta_init = theta.user_setpoint.get()
    dsp = lam_init/(2*np.sin(np.radians(delta_init/2)))
    theta_offset = delta_init/2 - theta_init

    E_vals = np.linspace(E_start, E_end, npts)  #
    lam_vals = np.linspace(12398/E_start, 12398/E_end, npts)
    x_range = max(E_vals) - min(E_vals)

    for lam_val in lam_vals:
        deltaCALC = np.degrees(np.arcsin(lam_val/2/dsp))*2
        thetaCALC = deltaCALC/2 - theta_offset
        deltas.append(deltaCALC)
        thetas.append(thetaCALC)

    motor_pos = (cycler(delta, deltas)
                 + cycler(theta, thetas)
                 + cycler(x_motor, E_vals))

    # TODO decide to include diffractometer motors below?
    # Before including pattern_args in metadata, replace objects with reprs.
    pattern_args['x_motor'] = repr(x_motor)
    _md = {'plan_args': {'detectors': list(map(repr, detectors)),
                         'x_motor': repr(x_motor), 'x_start': x_start,
                         'x_range': x_range,
                         'per_step': repr(per_step)},
           'extents': tuple([[x_start - x_range, x_start + x_range]]),
           'plan_name': 'EfixQapprox',
           'plan_pattern': 'scan',
           'plan_pattern_args': pattern_args,
           'num_points': npts,
           'num_inervals': int(npts-1),
           # this is broken TODO do we need this?
           # 'plan_pattern_module': plan_patterns.__name__,
           'hints': {}}
    try:
        dimensions = [(x_motor.hints['fields'], 'primary')]
        # print(dimensions)
    except (AttributeError, KeyError):
        pass
    else:
        _md['hints'].update({'dimensions': dimensions})
    _md.update(md or {})

    # this works without subs,
    Escan_plan = scan_nd(detectors, motor_pos, per_step=per_step, md=_md)

    reset_plan = bps.mv(x_motor, E_init, delta, delta_init, theta, theta_init)

    # Check for suitable syntax..
    # yield from print('Starting an Escan fix Q at ({:.4f}, {:.4f}, {:.4f})'.format(h_init,k_init,l_init))

    def plan_steps():
        yield from Escan_plan
        print('/nMoving back to motor positions immediately before scan/n')
        yield from reset_plan

    try:
        return (yield from plan_steps())

    except Exception:
        print('/nMoving back to motor positions immediately before scan/n')
        yield from reset_plan
        raise


# TODO make this number of points to match bsui standard, not the
# number of steps.
def EfixQ(detectors, E_start, E_end, npts, E_shift=0, *,
          per_step=None, md=None):
    '''Fixed Q energy scan based on an orientation matrix

    for CSX-1 mono (pgm.energy)

    If using higher order harmonic of mono, adjust H K L, not energy.

    Parameters


    ----------
    E_start : float
        starting energy [eV]
    E_stop : float
        stoping energy [eV]
    npts : integer
        number of points
    E_shift : float
        shift in energy calibration relative to orientation matrix (i.e, )
    per_step : callable, optional
        hook for cutomizing action of inner loop (messages per step)
        See docstring of bluesky.plans.one_nd_step (the default) for
        details.
    md : dict, optional
        metadata

    '''
    x_motor = pgm.energy  # This is CSX-1's mono motor name for energy
    x_start = E_start
    pattern_args = dict(x_motor=x_motor, x_start=E_start,
                        steps=npts, E_shift=E_shift)

    E_init = x_motor.setpoint.get()
    tardis.calc.energy = (x_motor.setpoint.get() + E_shift)/1000
    h_init = tardis.position.h
    k_init = tardis.position.k
    l_init = tardis.position.l
    delta_init = delta.user_setpoint.get()
    theta_init = theta.user_setpoint.get()
    gamma_init = gamma.user_setpoint.get()

    deltas = []
    thetas = []
    gammas = []

    # TODO no plus one, use npts as arugument.
    E_vals = np.linspace(E_start, E_end, npts)
    x_range = max(E_vals) - min(E_vals)

    for E_val in E_vals:
        tardis.calc.energy = (E_val + E_shift)/1000
        angles = tardis.forward([h_init, k_init, l_init])
        deltas.append(angles.delta)
        thetas.append(angles.theta)
        gammas.append(angles.gamma)

    motor_pos = (cycler(delta, deltas)
                 + cycler(theta, thetas)
                 + cycler(gamma, gammas)
                 + cycler(x_motor, E_vals))

    # TODO decide to include diffractometer motors below?
    # Before including pattern_args in metadata, replace objects with reprs.
    pattern_args['x_motor'] = repr(x_motor)
    _md = {'plan_args': {'detectors': list(map(repr, detectors)),
                         'x_motor': repr(x_motor), 'x_start': x_start,
                         'x_range': x_range,
                         'E_shift': E_shift,
                         'per_step': repr(per_step)},
           'extents': tuple([[x_start - x_range, x_start + x_range]]),
           'plan_name': 'EfixQ',
           'plan_pattern': 'scan',
           'plan_pattern_args': pattern_args,
           'num_points': npts,
           'num_inervals': int(npts-1),
           # this is broken TODO do we need this
           # 'plan_pattern_module': plan_patterns.__name__,
           'hints': {}}
    try:
        dimensions = [(x_motor.hints['fields'], 'primary')]
        # print(dimensions)
    except (AttributeError, KeyError):
        pass
    else:
        _md['hints'].update({'dimensions': dimensions})
    _md.update(md or {})

    # this works without subs,
    Escan_plan = scan_nd(detectors, motor_pos, per_step=per_step, md=_md)

    reset_plan = bps.mv(x_motor, E_init, delta, delta_init, theta,
                        theta_init, gamma, gamma_init)

    # yield from print('Starting an Escan fix Q at ({:.4f}, {:.4f}, {:.4f})'.format(h_init,k_init,l_init))

    def plan_steps():
        print('Starting fixed Q energy scan for '
              '({:.4f}, {:.4f}, {:.4f}\n\n)'.format(
                  tardis.h.position, tardis.k.position, tardis.l.position))
        yield from Escan_plan
        print('\nMoving back to motor positions immediately before scan\n')
        yield from reset_plan
        yield from bps.sleep(1)
        tardis.calc.energy = (pgm.energy.readback.get() + E_shift)/1000
        print('Returned to Q at ({:.4f}, {:.4f}, {:.4f})'.format(
            tardis.h.position, tardis.k.position, tardis.l.position))

    try:
        return (yield from plan_steps())

    except Exception:
        print('\nMoving back to motor positions immediately before scan\n')
        yield from reset_plan
        yield from bps.sleep(1)
        tardis.calc.energy = (pgm.energy.readback.get() + E_shift)/1000
        print('Returned to Q at ({:.4f}, {:.4f}, {:.4f})'.format(
            tardis.h.position, tardis.k.position, tardis.l.position))
        raise

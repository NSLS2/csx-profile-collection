from bluesky.magics import BlueskyMagics

from .startup import sd
from .detectors import *
from .endstation import *
from .accelerator import *
from .optics import *
from .tardis import *

#
# Setup of sup. data for plans
#
sd.monitors = []
sd.flyers = []
sd.baseline = [theta, delta, gamma, muR,
               sx, say, saz,
               cryoangle, sy, sz,
               epu1, epu2,
               slt1, slt2, slt3,
               m1a,
               m3a,
               #nanop, tardis,
               tardis, # unable to reproduce error: Plan failed: Type is not JSON serializable: _ResultTuple
               stemp, pgm,
               inout, es_diag1_y, diag6_pid,]# diag6.stats1.centroid_threshold ] ###TODOrecord_threshold_for_every_scan_and_PV_put_complete OR probably should link to diag6_pid as RO

#bec.disable_baseline() #no print to CLI, just save to datastore

sclr.names.read_attrs=['name1','name2','name3','name4','name5','name6']  # TODO  WHAT IS THIS??? - Dan Allan
sclr.channels.read_attrs=['chan1','chan2','chan3','chan4','chan5','chan6']
# Old-style hints config is replaced by the new 'kind' feature
# sclr.hints = {'fields': ['sclr_ch2', 'sclr_ch3', 'sclr_ch6']}
for i in [2, 3, 4, 5]:
    getattr(sclr.channels, f'chan{i}').kind = 'hinted'
    # getattr(sclr.channels, f'chan{i}').kind = 'normal' will remove the
    # hinted fields from LivePlot and LiveTable.

def relabel_fig(fig, new_label):
    fig.set_label(new_label)
    fig.canvas.manager.set_window_title(fig.get_label())

# fccd.hints = {'fields': ['fccd_stats1_total']}
for i in [1, 2, 3, 4, 5]:
    getattr(fccd, f'stats{i}').total.kind = 'hinted'

# cam_dif.hints = {'fields' : ['cam_dif_stats3_total','cam_dif_stats1_total']}
for i in [1, 3]:
    getattr(cam_dif, f'stats{i}').total.kind = 'hinted'

## 20180726 needed to comment due to IOC1 problems
#cube_beam.hints = {'fields': ['cube_beam_stats2_total', 'cube_beam_stats1_total']}
#for i in [1, 2]:
#     getattr(cube_beam, f'stats{i}').total.kind = 'hinted'

# This was imported in 00-startup.py #  used to generate the list: [thing.name for thing in get_all_positioners()]
"""
BlueskyMagics.positioners = [
    cryoangle,
    delta,
    diag2_y,
    diag3_y,
    diag5_y,
    diag6_pid,
    diag6_y,
    epu1.gap,
    epu1.phase,
    epu2.gap,
    epu2.phase,
    es_diag1_y,
    eta,
    gamma,
    m1a.z,
    m1a.y,
    m1a.x,
    m1a.pit,
    m1a.yaw,
    m1a.rol,
    m3a.x,
    m3a.pit,
    m3a.bdr,
    # muR,  # TODO turn this back on when safe
    # muT,  # TODO turn this back on when safe
    #nanop.tx,
    #nanop.ty,
    #nanop.tz,
    #nanop.bx,
    #nanop.by,
    #nanop.bz,
    say,
    saz,
    slt1.xg,
    slt1.xc,
    slt1.yg,
    slt1.yc,
    slt2.xg,
    slt2.xc,
    slt2.yg,
    slt2.yc,
    slt3.x,
    slt3.y,
    sx,
    sy,
    sz,
    tardis.h,
    tardis.k,
    tardis.l,
    tardis.theta,
    tardis.mu,
    tardis.chi,
    tardis.phi,
    tardis.delta,
    tardis.gamma,
    theta,
]
"""

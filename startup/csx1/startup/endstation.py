from ophyd import (PVPositioner, EpicsMotor, EpicsSignal, EpicsSignalRO,
                   PVPositionerPC, Device)
from ophyd import Component as Cpt

import bluesky.plan_stubs as bps

from ..devices.optics import (SamplePosVirtualMotor, Cryoangle,
                             Nanopositioner)

from ..devices.lakeshore import Lakeshore336
from ..devices.eps import EPSTwoStateDevice
from .tardis import tardis

from ..devices.devices import PMACKiller

# GVs
tardis_gv = EPSTwoStateDevice('XF:23ID1-VA{Diag:06-GV:1}',                             
                          state1='Closed', state2='Open',             
                          cmd_str1='Cls', cmd_str2='Opn',                        
                          nm_str1='Cls', nm_str2='Opn',                          
                          name='tardis_gv') 


# Diffo angles

delta = tardis.delta
gamma = tardis.gamma
theta = tardis.theta



# Sample positions

sx = EpicsMotor('XF:23ID1-ES{Dif-Ax:X}Mtr', name='sx', labels=['motors'])
sy = SamplePosVirtualMotor('XF:23ID1-ES{Dif-Ax:SY}', name='sy', labels=['motors'])
sz = SamplePosVirtualMotor('XF:23ID1-ES{Dif-Ax:SZ}', name='sz', labels=['motors'])
say = EpicsMotor('XF:23ID1-ES{Dif-Ax:Y}Mtr', name='say', labels=['motors'])
saz = EpicsMotor('XF:23ID1-ES{Dif-Ax:Z}Mtr', name='saz', labels=['motors'])
cryoangle = Cryoangle('', name='cryoangle', labels=['motors'])


kill_delta = PMACKiller(write_pv="XF:23ID1-ES{Dif-Ax:Del}Cmd:Kill-Cmd.PROC", read_pv="XF:23ID1-CT{MC:12-Ax:1}Kill-Sts")
kill_theta = PMACKiller(write_pv="XF:23ID1-ES{Dif-Ax:Th}Cmd:Kill-Cmd.PROC", read_pv="XF:23ID1-CT{MC:12-Ax:2}Kill-Sts")
kill_gamma = PMACKiller(write_pv="XF:23ID1-ES{Dif-Ax:Gam}Cmd:Kill-Cmd.PROC", read_pv="XF:23ID1-CT{MC:12-Ax:3}Kill-Sts")
kill_say = PMACKiller(write_pv="XF:23ID1-ES{Dif-Ax:Y}Cmd:Kill-Cmd.PROC", read_pv="XF:23ID1-CT{MC:12-Ax:7}Kill-Sts")
kill_saz = PMACKiller(write_pv="XF:23ID1-ES{Dif-Ax:Z}Cmd:Kill-Cmd.PROC", read_pv="XF:23ID1-CT{MC:12-Ax:8}Kill-Sts")

def kill_tardis():
    for mtr_kill in [kill_delta , kill_theta, kill_gamma, kill_say, kill_saz]:
        yield from bps.mv(mtr_kill, 0)

# Nano-positioners
# TODO This is the original setup.  Delete and replace with NEW
#nanop = Nanopositioner('XF:23ID1-ES{Dif:Lens', name='nanop')

# Diagnostic Axis

es_diag1_y = EpicsMotor('XF:23ID1-ES{Diag:1-Ax:Y}Mtr', name='es_diag1_y', labels=['motors'])
eta = EpicsMotor('XF:23ID1-ES{Diag:1-Ax:Eta}Mtr', name='eta')

# Lakeshore 336 Temp Controller

stemp = Lakeshore336('XF:23ID1-ES{TCtrl:1', name='stemp')

# Holography chamber motor
holoz = EpicsMotor('XF:23ID1-ES{Holo:Sample-Ax:Y}Mtr', name='holoz', labels=['motors'])

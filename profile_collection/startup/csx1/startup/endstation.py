from ophyd import (PVPositioner, EpicsMotor, EpicsSignal, EpicsSignalRO,
                   PVPositionerPC, Device)
from ophyd import Component as Cpt

from ..devices.optics import (SamplePosVirtualMotor, Cryoangle,
                             Nanopositioner)

from ..devices.lakeshore import Lakeshore336
from .tardis import tardis

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


# Nano-positioners
# TODO This is the original setup.  Delete and replace with NEW
#nanop = Nanopositioner('XF:23ID1-ES{Dif:Lens', name='nanop')

# Diagnostic Axis

es_diag1_y = EpicsMotor('XF:23ID1-ES{Diag:1-Ax:Y}Mtr', name='es_diag1_y', labels=['motors'])
eta = EpicsMotor('XF:23ID1-ES{Diag:1-Ax:Eta}Mtr', name='eta')

# Lakeshore 336 Temp Controller

stemp = Lakeshore336('XF:23ID1-ES{TCtrl:1', name='stemp')

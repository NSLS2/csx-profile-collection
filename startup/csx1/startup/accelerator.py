from ophyd import EpicsSignalRO, EpicsMotor
from ..devices.epu import EPU, BPM

#
# Ring Current for machine
#

ring_curr = EpicsSignalRO('XF:23ID-SR{}I-I', name='ring_curr')

#
# EPU Control
#

epu1 = EPU('XF:23ID-ID{EPU:1', epu_prefix='SR:C23-ID:G1A{EPU:1',
           ai_prefix='SR:C31-{AI}23', name='epu1')
epu2 = EPU('XF:23ID-ID{EPU:2', epu_prefix='SR:C23-ID:G1A{EPU:2',
           ai_prefix='SR:C31-{AI}23-2', name='epu2', labels=['source'])


#
# Beam Position Monitor
#

bpm = BPM('XF:23ID-ID{BPM}Val:', name = 'bpm')


#
# Canting magnet readback value
#

canter = EpicsSignalRO('SR:C23-MG:G1{MG:Cant-Ax:X}Mtr.RBV', name='canter')

#
# Phaser magnet control
#

phaser = EpicsMotor('SR:C23-MG:G1{MG:Phaser-Ax:Y}Mtr',name='phaser')
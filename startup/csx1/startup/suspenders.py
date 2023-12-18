from bluesky.suspenders import (SuspendBoolHigh,
                                SuspendBoolLow,
                                SuspendFloor,
                                SuspendCeil,
                                SuspendInBand,
                                SuspendOutBand)

from ophyd import EpicsSignal

from .startup import RE

#ring_suspender = SuspendFloor(EpicsSignal('XF:23ID-SR{}I-I'), 150, sleep=3*60)
# #ring_suspender = SuspendFloor(EpicsSignal('XF:23ID-SR{}I-I'), 250, sleep=3)
# fe_shut_suspender = SuspendBoolHigh(EpicsSignal('XF:23ID-PPS{Sh:FE}Pos-Sts'), sleep=10*60)
 #fe_shut_suspender = SuspendBoolHigh(EpicsSignal('XF:23ID-PPS{Sh:FE}Pos-Sts'), sleep=3)
#ps1_shut_suspender =  SuspendBoolHigh(EpicsSignal('XF:23IDA-PPS:1{PSh}Pos-Sts'),sleep=5*60)
# #ps1_shut_suspender =  SuspendBoolHigh(EpicsSignal('XF:23IDA-PPS:1{PSh}Pos-Sts'),sleep=3)

#RE.install_suspender(ring_suspender)
# RE.install_suspender(fe_shut_suspender)
#RE.install_suspender(ps1_shut_suspender)


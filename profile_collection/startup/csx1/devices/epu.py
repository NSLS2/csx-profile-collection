from ophyd import (PVPositioner, PVPositionerPC, Device,
                   EpicsSignal, EpicsSignalRO)
from ophyd import Component as Cpt
from ophyd import FormattedComponent as FmCpt

class EPUMotor(PVPositionerPC):
    readback = Cpt(EpicsSignalRO, 'Pos-I')
    setpoint = Cpt(EpicsSignal, 'Pos-SP')
    stop_signal = FmCpt(EpicsSignal,
                        '{self._stop_prefix}{self._stop_suffix}-Mtr.STOP')
    stop_value = 1

    def __init__(self, *args, parent=None, stop_suffix=None, **kwargs):
        self._stop_prefix = parent._epu_prefix
        self._stop_suffix = stop_suffix
        super().__init__(*args, parent=parent, **kwargs)


class Interpolator(Device):
    input = Cpt(EpicsSignal, 'Val:Inp1-SP')
    input_offset = Cpt(EpicsSignal, 'Val:InpOff1-SP')
    # {'Enabled', 'Disabled'}
    input_link = Cpt(EpicsSignal, 'Enbl:Inp1-Sel', string=True)
    input_pv = Cpt(EpicsSignal, 'Val:Inp1-SP.DOL$', string=True)
    output = Cpt(EpicsSignalRO, 'Val:Out1-I')
    # {'Enable', 'Disable'}
    output_link = Cpt(EpicsSignalRO, 'Enbl:Out1-Sel', string=True)
    output_pv = Cpt(EpicsSignal, 'Calc1.OUT$', string=True)
    output_deadband = Cpt(EpicsSignal, 'Val:DBand1-SP')
    output_drive = Cpt(EpicsSignalRO, 'Val:OutDrv1-I')
    interpolation_status = Cpt(EpicsSignalRO, 'Sts:Interp1-Sts', string=True)
#    table = Cpt(EpicsSignal, 'Val:Table-Sel', name='table')


class EPU(Device):
    gap = Cpt(EPUMotor, '-Ax:Gap}', stop_suffix='-Ax:Gap}')
    phase = Cpt(EPUMotor, '-Ax:Phase}', stop_suffix='-Ax:Phase}')
    x_off = FmCpt(EpicsSignalRO,'{self._ai_prefix}:FPGA:x_mm-I')
    x_ang = FmCpt(EpicsSignalRO,'{self._ai_prefix}:FPGA:x_mrad-I')
    y_off = FmCpt(EpicsSignalRO,'{self._ai_prefix}:FPGA:y_mm-I')
    y_ang = FmCpt(EpicsSignalRO,'{self._ai_prefix}:FPGA:y_mrad-I')
    flt = Cpt(Interpolator, '-FLT}')
    rlt = Cpt(Interpolator, '-RLT}')

    def __init__(self, *args, ai_prefix=None, epu_prefix=None, **kwargs):
        self._ai_prefix = ai_prefix
        self._epu_prefix = epu_prefix
        super().__init__(*args, **kwargs)

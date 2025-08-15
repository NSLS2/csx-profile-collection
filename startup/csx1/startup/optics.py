from ophyd import (EpicsMotor, PVPositioner, PVPositionerPC,
                   EpicsSignal, EpicsSignalRO, Device)
from ophyd import Component as Cpt
from ophyd import FormattedComponent as FmtCpt


from ..devices.optics import (FMBHexapodMirror, SlitsGapCenter,
                                    SlitsXY, FrontEndSlit)
from ..devices.eps import EPSTwoStateDevice

from ..devices.optics import (PGM, M3AMirror, PID)

from ..devices.motor_lookup import make_epics_motor_with_lookup_table

# M1A, M1B1, M1B2

m1a = FMBHexapodMirror('XF:23IDA-OP:1{Mir:1', name='m1a', labels=['optics'])

# VLS-PGM

pgm = PGM('XF:23ID1-OP{Mon',
          temp_pv='XF:23ID1-OP{TCtrl:1', name='pgm')

# M3A Mirror

m3a = M3AMirror('XF:23ID1-OP{Mir:3',  name='m3a', labels=['optics'])

# Slits

fe_slt = FrontEndSlit('FE:C23A-OP{Slt:12', name = 'FEslt', labels=['optics'])

slt1 = SlitsGapCenter('XF:23ID1-OP{Slt:1', name='slt1', labels=['optics'])
slt2 = SlitsGapCenter('XF:23ID1-OP{Slt:2', name='slt2', labels=['optics'])
slt3 = SlitsXY('XF:23ID1-OP{Slt:3', name='slt3', labels=['optics'])



# Diagnostic Manipulators

fs_diag1_x = make_epics_motor_with_lookup_table('-Ax:X}Mtr', motor_name='y', lut_suffix='Ax:X', num_rows=10)('XF:23IDA-BI:1{FS:1', name='fs_diag1_x')
diag2_y = EpicsMotor('XF:23ID1-BI{Diag:2-Ax:Y}Mtr', name='diag2_y', labels=['optics'])
diag3_y = EpicsMotor('XF:23ID1-BI{Diag:3-Ax:Y}Mtr', name='diag3_y', labels=['optics'])
diag5_y = EpicsMotor('XF:23ID1-BI{Diag:5-Ax:Y}Mtr', name='diag5_y', labels=['optics'])
diag6_y = EpicsMotor('XF:23ID1-BI{Diag:6-Ax:Y}Mtr', name='diag6_y', labels=['optics'])


# Setpoint for PID loop

diag6_pid = PID('XF:23ID1-OP{FBck}', name='diag6_pid', labels=['optics'])

## FCCD slow shutter

inout = EPSTwoStateDevice('XF:23IDA-EPS{DP:1-Sh:1}',
                          state1='Inserted', state2='Not Inserted',
                          cmd_str1='In', cmd_str2='Out',
                          nm_str1='In', nm_str2='Out',
                          name='inout')

dif_fs = EPSTwoStateDevice('XF:23ID1-ES{Dif-FS}', name='dif_fs',
                           state1='Inserted', state2='Not Inserted',
                           cmd_str1='In', cmd_str2='Out',
                           nm_str1='In', nm_str2='Out')

dif_diode = EPSTwoStateDevice('XF:23ID1-ES{Dif-Abs}', name='dif_diode',
                              state1='Inserted', state2='Not Inserted',
                              cmd_str1='In', cmd_str2='Out',
                              nm_str1='In', nm_str2='Out')


# Photon Shutters

ps_front_end = EPSTwoStateDevice('XF:23ID1-PPS{Sh:FE}}',
                               state1='Not Closed', state2='Closed',
                               cmd_str1='Opn', cmd_str2='Cls',
                               nm_str1='Opn', nm_str2='Cls',
                               name='FE_shutter')


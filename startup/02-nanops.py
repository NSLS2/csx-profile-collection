from ophyd.epics_motor import EpicsMotor
from ophyd.device import Component as Cpt
from ophyd.signal import EpicsSignal, EpicsSignalRO
from ophyd.status import (MoveStatus, DeviceStatus, wait as status_wait, StatusBase)
from ophyd import MotorBundle
from ophyd.utils.epics_pvs import raise_if_disconnected
#from epics import caget,caput


class NanoMotor(EpicsMotor):
    if type(EpicsMotor._default_configuration_attrs) in [list]:
        epics_config_attrs = EpicsMotor._default_configuration_attrs
    else:
        epics_config_attrs = []

    _default_configuration_attrs = (
        epics_config_attrs +
        ['dly', 'rtry', 'rdbd', 'rmod', 'cnen', 'pcof', 'icof' ])#TODO ensure not part of baseline data
    #user_setpoint = Cpt(EpicsSignal, 'PA_sm') #not v3 asmbly epics. .VAL should be
    dly = Cpt(EpicsSignal, '.DLY')
    rtry = Cpt(EpicsSignal, '.RTRY')
    rdbd = Cpt(EpicsSignal, '.RDBD')
    rmod = Cpt(EpicsSignal, '.RMOD')
    cnen = Cpt(EpicsSignal, '.CNEN')
    pcof = Cpt(EpicsSignal, '.PCOF')
    icof = Cpt(EpicsSignal, '.ICOF')
    stat = Cpt(EpicsSignal, '.STAT') #alarm status
    oplp = Cpt(EpicsSignal, '.STOP') #USING .STOP to open control loop.  normal .STOP should not do this


class NanoMotorOpenLoop(EpicsMotor): #TODO unverified for v3 asmbly epics
    #_default_configuration_attrs = (
    #    EpicsMotor._default_configuration_attrs +
    #    ('dly', 'rdbd', 'rmod', 'cnen', 'pcof', 'icof'))
    _default_read_attrs = ('user_setpoint', 'user_readback','done_signal')
    _default_configuration_attrs = ('velocity','t_settle')
    user_setpoint = Cpt(EpicsSignal, 'Abs')
    velocity = Cpt(EpicsSignal,'FQUnits')
    done_signal=Cpt(EpicsSignal,'DMOV')
    #acceleration=Cpt(EpicsSignal,'DMOV')
    user_readback = Cpt(EpicsSignal, 'AbsLast')
    dly = Cpt(EpicsSignal, '.DLY')
    # not needed here due to ppen loop mode.. but kept for symmetry
    rtry = Cpt(EpicsSignal, '.RTRY')
    rdbd = Cpt(EpicsSignal, '.RDBD')
    rmod = Cpt(EpicsSignal, '.RMOD')
    cnen = Cpt(EpicsSignal, '.CNEN')
    pcof = Cpt(EpicsSignal, '.PCOF')
    icof = Cpt(EpicsSignal, '.ICOF')
    #svrb = Cpt(EpicsSignalRO, 'scan_volt_RBV') #voltage of peizo expansion mode, should remain in baseline
    t_settle = Cpt(EpicsSignal, 'SETL')

    @property
    def connected(self):
         return True

    def remove_bad_signals(self):
        good_signals = list(self._default_read_attrs) + list(self._default_configuration_attrs)
        all_keys = list(self._signals.keys())
        for k in all_keys:
            if k not in good_signals:
                self._signals.pop(k, None)
        print(f'Signals: {list(self._signals.keys())}')

#COMMENT BELOW FOR NEW NANOP TESTING
    #def set(self, value):
    #    #self.done_signal.value=1
    #    st = DeviceStatus(self)
    #    # these arg sames matter
    #    #def am_done(old_value, value, **kwargs):
    #    def am_done(old_value,value, **kwargs):
    #        #if old_value == 1 and value == 0:
    #        #print("running",old_value,value,self.done_signal.value)
    #        #print("done_signal: ",smtr.done_signal.value)
    #        """
    #        if self.done_signal.value==1:
    #            if self.last_value==0:
    #                st._finished()
    #            self.last_value=1
    #        else:
    #            self.last_value=0
    #        """
    #        if old_value==0 and value==1:
    #            st._finished()

    #    self.done_signal.subscribe(am_done, run=False)
    #    self.user_setpoint.set(value)
    #    return st


class NanoMotorWithGentleStop(NanoMotor):
    @raise_if_disconnected
    def stop(self, *, success=False):
        # print(f'\n\n{self.name} motor status: {success}\n\n')
        # The status is always 'True' even on double Ctrl+C,
        # that is why we are using an extra condition to check
        # if the motor is still moving, to stop it.
        if not success or self.motor_is_moving.get():
            self.motor_stop.put(1, wait=False)
            super().stop(success=success)

class NanoSignal(Device):
    svolt = Cpt(EpicsSignalRO, 'scan_volt_RBV', name = 'svolt')
    svoltscan = Cpt(EpicsSignal, 'scan_volt_RBV.SCAN', name = 'svoltscan')
    noslip = Cpt(EpicsSignal, 'noslip_tog_RBV', name = 'noslip', write_pv = 'noslip_tog')
    # 20210216 added following John's idea of checking the temperatures as
    # a potential reason for periodic and unpredictable jumps during meshes
    temp = Cpt(EpicsSignalRO, 'Ch_Temp', name = 'temp')


class NanoBundleSignal(Device):
    tx = Cpt(NanoSignal, 'TopX}')
    ty = Cpt(NanoSignal, 'TopY}')
    tz = Cpt(NanoSignal, 'TopZ}')
    bx = Cpt(NanoSignal, 'BtmX}')
    by = Cpt(NanoSignal, 'BtmY}')
    bz = Cpt(NanoSignal, 'BtmZ}')


nanopsignal = NanoBundleSignal('XF:23ID1-ES{Dif:Nano-Ax:', name='nanopsignal')


class NanoBundle(MotorBundle):
    #tx = Cpt(NanoMotor, 'TopX}Mtr') # essentially open loop mode for v3 asmbly epics until we can update epics driver
    tx = Cpt(NanoMotorWithGentleStop, 'TopX}Mtr')
    ty = Cpt(NanoMotorWithGentleStop, 'TopY}Mtr')
    tz = Cpt(NanoMotorWithGentleStop, 'TopZ}Mtr')
    bx = Cpt(NanoMotorWithGentleStop, 'BtmX}Mtr')
    by = Cpt(NanoMotorWithGentleStop, 'BtmY}Mtr')
    bz = Cpt(NanoMotorWithGentleStop, 'BtmZ}Mtr')
    #bz = Cpt(NanoMotorOpenLoop, 'BtmZ}OL') #TODO no testing made for v3 asmbly epics
    #swap between the two above lines if BtmZ close/open loop is desired, respectively #TODO if this is still needed, make this OL part of the device, no restart bsui


# check if nanop already there and remove it
try:
    sd.baseline.remove(nanop)
except NameError:
    pass
except ValueError:
    pass

nanop = NanoBundle('XF:23ID1-ES{Dif:Nano-Ax:', name='nanop', labels=['motor, optics, nanops'])
pa_mot_th = NanoMotorWithGentleStop("XF:23ID1-ES{PA-Ax:Th}Mtr", name="pa_mot_th")
#nanop.bz.remove_bad_signals()  # solve the issue with disconnection errors #TODO is this needed for v3 asmbly epics


class MotorPairX(Device):
    tx = Cpt(NanoMotorWithGentleStop, nanop.tx.prefix)
    bx = Cpt(NanoMotorWithGentleStop, nanop.bx.prefix)
    def __init__(self, *args, **kwargs):
        """A bundle device to move a pair of motors to the same distance"""
        super().__init__(*args, **kwargs)
        self.update_diff()

    def update_diff(self):
        self.diff = self.tx.user_readback.get() - self.bx.user_readback.get()

    def set(self, value):
        st_leader = self.tx.set(value)  # leading motor
        st_follower = self.bx.set(value - self.diff)  # following motor
        return st_leader & st_follower


class MotorPairY(Device):
    ty = Cpt(NanoMotorWithGentleStop, nanop.ty.prefix)
    by = Cpt(NanoMotorWithGentleStop, nanop.by.prefix)
    def __init__(self, *args, **kwargs):
        """A bundle device to move a pair of motors to the same distance"""
        super().__init__(*args, **kwargs)
        self.update_diff()

    def update_diff(self):
        self.diff = self.ty.user_readback.get() - self.by.user_readback.get()

    def set(self, value):
        st_leader = self.ty.set(value)  # leading motor
        st_follower = self.by.set(value - self.diff)  # following motor
        return st_leader & st_follower


class MotorPairZ(Device):
    tz = Cpt(NanoMotorWithGentleStop, nanop.tz.prefix)
    bz = Cpt(NanoMotorWithGentleStop, nanop.bz.prefix)
    def __init__(self, *args, **kwargs):
        """A bundle device to move a pair of motors to the same distance"""
        super().__init__(*args, **kwargs)
        self.update_diff()

    def update_diff(self):
        self.diff = self.tz.user_readback.get() - self.bz.user_readback.get()

    def set(self, value):
        st_leader = self.tz.set(value)  # leading motor
        st_follower = self.bz.set(value - self.diff)  # following motor
        return st_leader & st_follower


mpx = MotorPairX(name='mpx')
mpy = MotorPairY(name='mpy')
mpz = MotorPairZ(name='mpz')

mpx.kind = 'hinted'
mpx.tx.kind = 'hinted'
mpx.bx.kind = 'hinted'

mpy.kind = 'hinted'
mpy.ty.kind = 'hinted'
mpy.by.kind = 'hinted'

mpz.kind = 'hinted'
mpz.tz.kind = 'hinted'
mpz.bz.kind = 'hinted'


# Velocity (tested 0.5, 0.1, 0.05, 0.01)

# this was 0.30 but AB tried scanning with larger step
# size and had difficulty in that bluesky will never trigger the
# cube_beam because bsui is waiting on this to move??


# Settling time (tested 0.1 - 0.3)
#temp_settle_time = 0.2
#COMMENT BELOW FOR NEW NANOP TESTING
#_base_nano_setting = {'velocity': 0.10,
#                      'acceleration': 0.20,
#                      'dly': temp_settle_time,
#                      'rtry': 3,
#                      'rdbd': 1e-5,
#                      'rmod': 1,
#                      'cnen': 1,
#                      'pcof': 0.1,
#                      'icof': 0.010
#                      }

for cpt in ['tx', 'ty', 'tz', 'bx', 'by', 'bz']:
    getattr(nanop, cpt).configuration_attrs.extend(['velocity', 'acceleration'])#TODO add offset, egu, offset dir, UEIP

#COMMENT BELOW FOR NEW NANOP TESTING
#for nn in nanop.component_names:
#    if nn == "bz (remove this if open loop)":
#        getattr(nanop, nn).configure({'velocity':0.10,'t_settle':temp_settle_time})
#        continue
#    getattr(nanop, nn).configure(_base_nano_setting)

# BlueskyMagics.positioners += [getattr(nanop, nn) for nn in nanop.component_names]

#sd.baseline += [getattr(nanop, nn) for nn in nanop.component_names]
sd.baseline += [nanop]

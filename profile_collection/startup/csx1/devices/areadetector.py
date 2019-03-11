from ophyd import (EpicsScaler, EpicsSignal, EpicsSignalRO, Device,
                   SingleTrigger, HDF5Plugin, ImagePlugin, StatsPlugin,
                   ROIPlugin, TransformPlugin, OverlayPlugin)

from ophyd.areadetector.cam import AreaDetectorCam
from ophyd.areadetector.detectors import DetectorBase
from ophyd.areadetector.filestore_mixins import FileStoreHDF5IterativeWrite
from ophyd.areadetector import ADComponent, EpicsSignalWithRBV
from ophyd.areadetector.plugins import PluginBase, ProcessPlugin
from ophyd import Component as Cpt
from ophyd.device import FormattedComponent as FCpt
from ophyd import AreaDetector

from ophyd.sim import NullStatus

from .devices import DelayGenerator
from .scaler import StruckSIS3820MCS

import numpy as np

from .stats_plugin import StatsPluginCSX


class StandardCam(SingleTrigger, AreaDetector):
    stats1 = Cpt(StatsPlugin, 'Stats1:')
    stats2 = Cpt(StatsPlugin, 'Stats2:')
    stats3 = Cpt(StatsPlugin, 'Stats3:')
    stats4 = Cpt(StatsPlugin, 'Stats4:')
    stats5 = Cpt(StatsPlugin, 'Stats5:')
    roi1 = Cpt(ROIPlugin, 'ROI1:')
    roi2 = Cpt(ROIPlugin, 'ROI2:')
    roi3 = Cpt(ROIPlugin, 'ROI3:')
    roi4 = Cpt(ROIPlugin, 'ROI4:')
    #proc1 = Cpt(ProcessPlugin, 'Proc1:')
    #trans1 = Cpt(TransformPlugin, 'Trans1:')


class NoStatsCam(SingleTrigger, AreaDetector):
    pass


class HDF5PluginSWMR(HDF5Plugin):
    swmr_active = Cpt(EpicsSignalRO, 'SWMRActive_RBV')
    swmr_mode = Cpt(EpicsSignalWithRBV, 'SWMRMode')
    swmr_supported = Cpt(EpicsSignalRO, 'SWMRSupported_RBV')
    swmr_cb_counter = Cpt(EpicsSignalRO, 'SWMRCbCounter_RBV')
    _default_configuration_attrs = (HDF5Plugin._default_configuration_attrs +
                                    ('swmr_active', 'swmr_mode',
                                     'swmr_supported'))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stage_sigs['swmr_mode'] = 1


class HDF5PluginWithFileStore(HDF5PluginSWMR, FileStoreHDF5IterativeWrite):
    # AD v2.2.0 (at least) does not have this. It is present in v1.9.1.
    file_number_sync = None

    def get_frames_per_point(self):
        return self.parent.cam.num_images.get()

    def make_filename(self):
        # stash this so that it is available on resume
        self._ret = super().make_filename()
        return self._ret

class FCCDCam(AreaDetectorCam):
    sdk_version = Cpt(EpicsSignalRO, 'SDKVersion_RBV')
    firmware_version = Cpt(EpicsSignalRO, 'FirmwareVersion_RBV')
    overscan_cols = Cpt(EpicsSignalWithRBV, 'OverscanCols')
    fcric_gain = Cpt(EpicsSignalWithRBV, 'FCRICGain')
    fcric_clamp = Cpt(EpicsSignalWithRBV, 'FCRICClamp')
    temp = FCpt(EpicsSignal, '{self._temp_pv}')

    def __init__(self, *args, temp_pv=None, **kwargs):
        self._temp_pv = temp_pv
        super().__init__(*args, **kwargs)


class FastCCDPlugin(PluginBase):
    _default_suffix = 'FastCCD1:'
    capture_bgnd = Cpt(EpicsSignalWithRBV, 'CaptureBgnd')
    enable_bgnd = Cpt(EpicsSignalWithRBV, 'EnableBgnd')
    enable_gain = Cpt(EpicsSignalWithRBV, 'EnableGain')
    enable_size = Cpt(EpicsSignalWithRBV, 'EnableSize')
    rows = Cpt(EpicsSignalWithRBV, 'Rows')#
    row_offset = Cpt(EpicsSignalWithRBV, 'RowOffset')
    overscan_cols = Cpt(EpicsSignalWithRBV, 'OverscanCols')



class ProductionCamBase(DetectorBase):
    # # Trying to add useful info..
    cam = Cpt(FCCDCam, "cam1:")
    stats1 = Cpt(StatsPluginCSX, 'Stats1:')
    stats2 = Cpt(StatsPluginCSX, 'Stats2:')
    stats3 = Cpt(StatsPluginCSX, 'Stats3:')
    stats4 = Cpt(StatsPluginCSX, 'Stats4:')
    stats5 = Cpt(StatsPluginCSX, 'Stats5:')
    roi1 = Cpt(ROIPlugin, 'ROI1:')
    roi2 = Cpt(ROIPlugin, 'ROI2:')
    roi3 = Cpt(ROIPlugin, 'ROI3:')
    roi4 = Cpt(ROIPlugin, 'ROI4:')
    trans1 = Cpt(TransformPlugin, 'Trans1:')
    proc1 = Cpt(ProcessPlugin, 'Proc1:')
    over1 = Cpt(OverlayPlugin, 'Over1:')
    fccd1 = Cpt(FastCCDPlugin, 'FastCCD1:')


    # This does nothing, but it's the right place to add code to be run
    # once at instantiation time.
    def __init__(self, *arg, readout_time=0.04, **kwargs):
        self.readout_time = readout_time
        super().__init__(*arg, **kwargs)

    def pause(self):
        self.cam.acquire.put(0)
        super().pause()

    def stage(self):
        from ophyd.utils import set_and_wait
        import time as ttime

        # pop both string and object versions to be paranoid
        self.stage_sigs.pop('cam.acquire', None)
        self.stage_sigs.pop(self.cam.acquire, None)

        # we need to take the detector out of acquire mode
        self._original_vals[self.cam.acquire] = self.cam.acquire.get()
        set_and_wait(self.cam.acquire, 0)
        # but then watch for when detector state
        while self.cam.detector_state.get(as_string=True) != 'Idle':
            ttime.sleep(.01)

        return super().stage()


class ProductionCamStandard(SingleTrigger, ProductionCamBase):

    hdf5 = Cpt(HDF5PluginWithFileStore,
               suffix='HDF1:',
               write_path_template='/GPFS/xf23id/xf23id1/fccd_data/%Y/%m/%d/',
               root='/GPFS/xf23id/xf23id1/',
               reg=None)  # placeholder to be set on instance as obj.hdf5.reg

    def stop(self):
        self.hdf5.capture.put(0)
        return super().stop()

    def pause(self):
        set_val = 0
        self.hdf5.capture.put(set_val)
        #val = self.hdf5.capture.get()
        ## Julien fix to ensure these are set correctly
        #print("pausing FCCD")
        #while (np.abs(val-set_val) > 1e-6):
            #self.hdf5.capture.put(set_val)
            #val = self.hdf5.capture.get()

        return super().pause()

    def resume(self):
        set_val = 1
        self.hdf5.capture.put(set_val)
        # The AD HDF5 plugin bumps its file_number and starts writing into a
        # *new file* because we toggled capturing off and on again.
        # Generate a new Resource document for the new file.

        # grab the stashed result from make_filename
        filename, read_path, write_path = self.hdf5._ret
        self.hdf5._fn = self.hdf5.file_template.get() % (read_path,
                                               filename,
                                               self.hdf5.file_number.get() - 1)
                                               # file_number is *next* iteration
        res_kwargs = {'frame_per_point': self.hdf5.get_frames_per_point()}
        self.hdf5._generate_resource(res_kwargs)
        # can add this if we're not confident about setting...
        #val = self.hdf5.capture.get()
        #print("resuming FCCD")
        #while (np.abs(val-set_val) > 1e-6):
            #self.hdf5.capture.put(set_val)
            #val = self.hdf5.capture.get()
        #print("Success")
        return super().resume()


class TriggeredCamExposure(Device):
    def __init__(self, *args, **kwargs):
        self._Tc = 0.004
        self._To = 0.0035
        self._readout = 0.080
        super().__init__(*args, **kwargs)

    def set(self, exp):
        # Exposure time = 0
        # Cycle time = 1

        if exp[0] is not None:
            Efccd = exp[0] + self._Tc + self._To
            # To = start of FastCCD Exposure
            aa = 0                          # Shutter open
            bb = Efccd - self._Tc + aa      # Shutter close
            cc = self._To * 3               # diag6 gate start
            dd = exp[0] - (self._Tc * 2)    # diag6 gate stop
            ee = 0                          # Channel Adv Start
            ff = 0.001                      # Channel Adv Stop
            gg = self._To                   # MCS Count Gate Start
            hh = exp[0] + self._To          # MCS Count Gate Stop

            # Set delay generator
            self.parent.dg1.A.set(aa)
            self.parent.dg1.B.set(bb)
            self.parent.dg1.C.set(cc)
            self.parent.dg1.D.set(dd)
            self.parent.dg1.E.set(ee)
            self.parent.dg1.F.set(ff)
            self.parent.dg1.G.set(gg)
            self.parent.dg1.H.set(hh)
            self.parent.dg2.A.set(0)
            self.parent.dg2.B.set(0.0005)

            # Set AreaDetector
            self.parent.cam.acquire_time.set(Efccd)

        # Now do period
        if exp[1] is not None:
            if exp[1] < (Efccd + self._readout):
                p = Efccd + self._readout
            else:
                p = exp[1]

        self.parent.cam.acquire_period.set(p)

        if exp[2] is not None:
            self.parent.cam.num_images.set(exp[2])

        return NullStatus()

    def get(self):
        return None


class ProductionCamTriggered(ProductionCamStandard):
    dg2 = FCpt(DelayGenerator, '{self._dg2_prefix}')
    dg1 = FCpt(DelayGenerator, '{self._dg1_prefix}')
    mcs = FCpt(StruckSIS3820MCS, '{self._mcs_prefix}')
    exposure = Cpt(TriggeredCamExposure, '')

    def __init__(self, *args, dg1_prefix=None, dg2_prefix=None,
                 mcs_prefix=None, **kwargs):
        self._dg1_prefix = dg1_prefix
        self._dg2_prefix = dg2_prefix
        self._mcs_prefix = mcs_prefix
        super().__init__(*args, **kwargs)


    def trigger(self):
        self.mcs.trigger()
        return super().trigger()

    def read(self):
        self.mcs.read()
        return super().read()




class StageOnFirstTrigger(ProductionCamTriggered):


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.trigger_staged = False

    def _trigger_stage(self):

        self._acquisition_signal.subscribe(self._acquire_changed)
        return super().stage()

    def stage(self):
        return [self]

    def unstage(self):
        super().unstage()
        self._acquisition_signal.clear_sub(self._acquire_changed)
        self.trigger_staged = False


    def trigger(self):
        from ophyd.utils import set_and_wait
        import time as ttime


        if not self.trigger_staged:
            self._trigger_stage()
            self.trigger_staged = True

        return super().trigger()





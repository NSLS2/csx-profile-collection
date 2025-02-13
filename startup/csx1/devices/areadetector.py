from ophyd import (EpicsScaler, EpicsSignal, EpicsSignalRO, Device, BlueskyInterface,
                   SingleTrigger, HDF5Plugin, ImagePlugin, StatsPlugin,
                   ROIPlugin, TransformPlugin, OverlayPlugin, ProsilicaDetector, TIFFPlugin)

from ophyd.areadetector.cam import AreaDetectorCam
from ophyd.areadetector.detectors import DetectorBase
from ophyd.areadetector.filestore_mixins import FileStoreHDF5IterativeWrite, FileStoreTIFFIterativeWrite
from ophyd.areadetector import ADComponent, EpicsSignalWithRBV
from ophyd.areadetector.plugins import PluginBase, ProcessPlugin, HDF5Plugin_V22, TIFFPlugin_V22
from ophyd import Component as Cpt, DeviceStatus
from ophyd.device import FormattedComponent as FCpt
from ophyd import AreaDetector
import time as ttime
import itertools

from ophyd.sim import NullStatus

from .devices import DelayGenerator
from .scaler import StruckSIS3820MCS

import numpy as np

from .stats_plugin import StatsPluginCSX


DEFAULT_TIMEOUT = 10  # Seconds

class ContinuousAcquisitionTrigger(BlueskyInterface):
    """
    TODO: Not currently operational. It saves `cam.num_images` images in a single file.
          The intended behavior is to save one file across all images in a BlueskyRun.

    This trigger mixin class records images when it is triggered.

    It expects the detector to *already* be acquiring, continously.
    """
    def __init__(self, *args, plugin_name=None, image_name=None, **kwargs):
        if plugin_name is None:
            raise ValueError("plugin name is a required keyword argument")
        super().__init__(*args, **kwargs)
        self._plugin = getattr(self, plugin_name)
        if image_name is None:
            image_name = '_'.join([self.name, 'image'])
        self._plugin.stage_sigs[self._plugin.auto_save] = 'No'
        self.cam.stage_sigs[self.cam.image_mode] = 'Continuous'
        self._plugin.stage_sigs[self._plugin.file_write_mode] = 'Capture'
        self._image_name = image_name
        self._status = None
        self._num_captured_signal = self._plugin.num_captured
        self._num_captured_signal.subscribe(self._num_captured_changed)
        self._save_started = False

    def stage(self):
        if self.cam.acquire.get() != 1:
             try:
                self.cam.acquire.put(1)
                print("Not currently acquiring...starting continuous acquisition.")
                ttime.sleep(1)
             except:
                 raise RuntimeError("The ContinuousAcuqisitionTrigger expects "
                                    "the detector to already be acquiring."
                                    "I was unable to fix it, please restart the ui.") #new line, DO
             
        # Stage the detector
        super().stage()

    def trigger(self):
        "Trigger one acquisition."
        if not self._staged:
            raise RuntimeError("This detector is not ready to trigger."
                               "Call the stage() method before triggering.")
        self._save_started = False
        self._status = DeviceStatus(self)
        self._desired_number_of_images = self.cam.num_images.get()
        self._plugin.num_capture.put(self._desired_number_of_images)
        self.generate_datum(self._image_name, ttime.time())
        # self.proc.reset_filter.put(1) TODO: Not sure if this is needed
        self._plugin.capture.put(1)
        return self._status

    def _num_captured_changed(self, value=None, old_value=None, **kwargs):
        "This is called when the `num_captured` signal changes."
        if self._status is None:
            # This means that we captured data before or in between triggers.
            # This leads to dropped frames so we don't want this to occur.
            raise RuntimeError("Dropped frames detected. There is a race condition between the HDF plugin and the trigger method.")
        # If we have captured the desired number of images, write the file.
        # TODO: we don't want to write the file here, we want to write the file in `unstage()`
        if value == self._desired_number_of_images:
            try:
                self._plugin.write_file.put(1)
            except Exception as e:
                print(e)
                raise
            self._save_started = True
        if value == 0 and self._save_started:
            self._status._finished()
            self._status = None
            self._save_started = False


##TODO why AreaDetector and not ProsilicaDetector for StandardCam Class below
class StandardCam(SingleTrigger, AreaDetector):#TODO is there something more standard for prosilica? seems only used on prosilica. this does stats, but no image saving (unsure if easy to configure or not and enable/disable)
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
    trans1 = Cpt(TransformPlugin, 'Trans1:')
    over1 = Cpt(OverlayPlugin, 'Over1:') ##for crosshairs in tiff


class AxisDetectorCam(AreaDetectorCam):
    """
    Custom AxisDetectorCam class to include a `wait_for_plugins` signal.
    """
    _default_configuration_attrs = AreaDetectorCam._default_configuration_attrs + (
        "gain",
        "tec"
    )
    wait_for_plugins = Cpt(EpicsSignal, "WaitForPlugins", string=True, kind="hinted")
    gain = Cpt(EpicsSignal, "GainMode", string=True, kind="config")
    tec = Cpt(EpicsSignal, "TEC", string=True, kind="config")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stage_sigs["wait_for_plugins"] = "Yes"

    def ensure_nonblocking(self):
        self.stage_sigs["wait_for_plugins"] = "Yes"
        for c in self.parent.component_names:
            cpt = getattr(self.parent, c)
            if cpt is self:
                continue
            if hasattr(cpt, "ensure_nonblocking"):
                cpt.ensure_nonblocking()


# TODO: Change from `SingleTrigger` to `ContinuousAcquisitionTrigger`
class StandardAxisCam(SingleTrigger, AreaDetector):
    cam = Cpt(AxisDetectorCam, "cam1:")
    stats1 = Cpt(StatsPlugin, 'Stats1:')
    stats2 = Cpt(StatsPlugin, 'Stats2:')
    stats3 = Cpt(StatsPlugin, 'Stats3:')
    stats4 = Cpt(StatsPlugin, 'Stats4:')
    stats5 = Cpt(StatsPlugin, 'Stats5:')
    roi1 = Cpt(ROIPlugin, 'ROI1:')
    roi2 = Cpt(ROIPlugin, 'ROI2:')
    roi3 = Cpt(ROIPlugin, 'ROI3:')
    roi4 = Cpt(ROIPlugin, 'ROI4:')
    proc1 = Cpt(ProcessPlugin, 'Proc1:')
    trans1 = Cpt(TransformPlugin, 'Trans1:')
    over1 = Cpt(OverlayPlugin, 'Over1:')


class NoStatsCam(SingleTrigger, AreaDetector):
    pass


class MonitorStatsCam(SingleTrigger, AreaDetector): #TODO does this subscribe/unsubsribe work or are we hacking EpicsSignals in custom plans
    stats1 = Cpt(StatsPlugin, "Stats1:")
    roi1 = Cpt(ROIPlugin, "ROI1:")
    proc1 = Cpt(ProcessPlugin, "Proc1:")

    def subscribe(self, *args, **kwargs):
        #TODO centroid.x was orignally cenx, neither work in substribing. - figure out later.
        return self.stats1.centroid.x.subscribe(*args, **kwargs) #TODO if this works, then add stats1.ceny too.
        # return self.stast1.centroid.y.subscribe(*args, **kwargs) #TODO if this works, then add stats1.ceny too.

    def unsubuscribe(self, *args, **kwargs):
        return self.stats1.centroid.x.unsubscribe(*args, **kwargs)
        # return self.stats1.centroid.y.unsubscribe(*args, **kwargs)


def update_describe_typing(dic, obj):
    """
    Function for updating dictionary result of `describe` to include better typing.
    Previous defaults did not use `dtype_str` and simply described an image as an array.

    Parameters
    ==========
    dic: dict
        Return dictionary of describe method
    obj: OphydObject
        Instance of plugin
    """
    key = obj.parent._image_name
    cam_dtype = obj.parent.cam.data_type.get(as_string=True)
    type_map = {'UInt8': '|u1', 'UInt16': '<u2', 'Float32':'<f4', "Float64":'<f8'}
    if cam_dtype in type_map:
        dic[key].setdefault('dtype_str', type_map[cam_dtype])


class HDF5PluginWithFileStorePlain(HDF5Plugin_V22, FileStoreHDF5IterativeWrite): ##SOURCED FROM BELOW FROM FCCD WITH SWMR removed

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # In CSS help: "N < 0: Up to abs(N) new directory levels will be created"
        self.stage_sigs.update({"create_directory": -3})
        # last=False turns move_to_end into move_to_start. Yes, it's silly.
        self.stage_sigs.move_to_end("create_directory", last=False)

    def get_frames_per_point(self):
        return self.parent.cam.num_images.get()

    def make_filename(self):
        # stash this so that it is available on resume
        self._ret = super().make_filename()
        return self._ret

    def describe(self):
        ret = super().describe()
        update_describe_typing(ret, self)
        return ret


class StandardProsilicaWithHDF5(StandardCam):
    hdf5 = Cpt(HDF5PluginWithFileStorePlain,
              suffix='HDF1:',
              write_path_template='/nsls2/data/csx/legacy/prosilica_data/hdf5/%Y/%m/%d',
              root='/nsls2/data/csx/legacy')
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hdf5.kind = "normal"


class TIFFPluginWithFileStore(TIFFPlugin_V22, FileStoreTIFFIterativeWrite): #RIPPED OFF FROM CHX because mutating H5 has wrong shape for color img
    """Add this as a component to detectors that write TIFFs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # In CSS help: "N < 0: Up to abs(N) new directory levels will be created"
        self.stage_sigs.update({"create_directory": -3})
        # last=False turns move_to_end into move_to_start. Yes, it's silly.
        self.stage_sigs.move_to_end("create_directory", last=False)

    def describe(self):
        ret = super().describe()
        key = self.parent._image_name
        color_mode = self.parent.cam.color_mode.get(as_string=True)
        if color_mode == 'Mono':
            ret[key]['shape'] = [
                self.parent.cam.num_images.get(),
                self.array_size.height.get(),
                self.array_size.width.get()
                ]
       
        elif color_mode in ['RGB1', 'Bayer']:
            ret[key]['shape'] = [self.parent.cam.num_images.get(), *self.array_size.get()]
        else:
            raise RuntimeError(f"Parent camera color mode for TIFFPluginWithFileStore, {color_mode}, "
                               f"not one of 'Mono', 'RGB1', nor 'Bayer'")
        update_describe_typing(ret, self)
        return ret


class StandardProsilicaWithTIFF(StandardCam): #RIPPED OFF FROM CHX and not using their custom StandardProcilica class (StandardCam here)
    tiff = Cpt(TIFFPluginWithFileStore,
               suffix='TIFF1:',              
               write_path_template='/nsls2/data/csx/legacy/prosilica_data/tiff/%Y/%m/%d',
               root='/nsls2/data/csx/legacy')
    def __init__(self, *args, **kwargs): #TODOandi-understand why must be self, #TODOclaudio should we do this for stats?
        super().__init__(*args, **kwargs)
        self.tiff.kind = "normal"
    

### LOOKS LIKE FCCD STUFF STARTS HERE
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


class AxisCam(StandardAxisCam):
    """
    Class for Axis detector with HDF5 file saving.

    The IOC is currently hosted on a Windows machine so the
    `write_path_template` must be specified as a Windows path.
    """
    hdf5 = Cpt(HDF5PluginWithFileStorePlain,
              suffix='HDF1:',
              read_path_template='/nsls2/data/csx/legacy/axis_data/hdf5/%Y/%m/%d',
              root='/nsls2/data/csx/legacy/axis_data/hdf5',
              write_path_template='Z:/hdf5/%Y/%m/%d', # From the IOC which is Windows
              path_semantics='windows')
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hdf5.kind = "normal"
        self.hdf5.file_path.path_semantics = "nt" # windows path semantics
        self.ensure_acquiring = False
        # Camera is currently UInt16, the default is wrong at Int8
        self.cam.data_type.set("UInt16")
        ttime.sleep(1)

    def stage(self):
        # Ensure we continue acquiring in case of failure
        self.ensure_acquiring = self.cam.image_mode.get() == "Continuous" and self.cam.acquire.get() == 1
        super().stage()

    def unstage(self):
        super().unstage()
        # If the image mode was continuous, start acquiring again
        if self.ensure_acquiring:
            self.cam.image_mode.put("Continuous")
            ttime.sleep(1)
            self.cam.acquire.put(1)


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
    capture_bgnd = Cpt(EpicsSignalWithRBV, 'CaptureBgnd', write_timeout=5,
                       auto_monitor=False, put_complete=True)
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

        # pop both string and object versions to be paranoid
        self.stage_sigs.pop('cam.acquire', None)
        self.stage_sigs.pop(self.cam.acquire, None)

        # we need to take the detector out of acquire mode
        self._original_vals[self.cam.acquire] = self.cam.acquire.get()
        self.cam.acquire.set(0).wait(DEFAULT_TIMEOUT)
        # but then watch for when detector state
        while self.cam.detector_state.get(as_string=True) != 'Idle':
            ttime.sleep(.01)

        return super().stage()


class ProductionCamStandard(SingleTrigger, ProductionCamBase):

    hdf5 = Cpt(HDF5PluginWithFileStore,
               suffix='HDF1:',
               #write_path_template='/GPFS/xf23id/xf23id1/fccd_data/%Y/%m/%d/',
               write_path_template='/nsls2/data/csx/legacy/fccd_data/%Y/%m/%d/',
               #root='/GPFS/xf23id/xf23id1/',
               root='/nsls2/data/csx/legacy',
               reg=None)  # placeholder to be set on instance as obj.hdf5.reg

    def make_data_key(self):
        """
        Override the base class to get the array shape from the HDF5 plugin.

        The base class gets the shape from self.cam.array_size.  This does not
        correctly represent the shape of the array written by the custom HDF5
        plugin used on this detector, so we need to get the shape from the
        plugin.
        """
        source = 'PV:{}'.format(self.prefix)
        # This shape is expected to match arr.shape for the array.
        shape = (
            self.cam.num_images.get(),
            self.hdf5.height.get(),
            self.hdf5.width.get(),
        )
        return dict(shape=shape, source=source, dtype='array',
                    external='FILESTORE:')

    def stop(self):
        self.hdf5.capture.put(0)
        return super().stop()

    def pause(self):
        set_val = 0
        self.hdf5.capture.set(set_val).wait(DEFAULT_TIMEOUT)
        #val = self.hdf5.capture.get()
        ## Julien fix to ensure these are set correctly
        #print("pausing FCCD")
        #while (np.abs(val-set_val) > 1e-6):
            #self.hdf5.capture.put(set_val)
            #val = self.hdf5.capture.get()

        return super().pause()

    def resume(self):
        set_val = 1
        self.hdf5.capture.set(set_val).wait(DEFAULT_TIMEOUT)
        self.hdf5._point_counter = itertools.count()
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

        # A NullStatus is always immediate 'done'.
        # This will be AND-ed with non-null statuses below, if applicable,
        # effectively reporting that this set operation is done when *all* the
        # individual set operations are done.
        status = NullStatus()  

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
            status &= self.parent.dg1.A.set(aa)
            status &= self.parent.dg1.B.set(bb)
            status &= self.parent.dg1.C.set(cc)
            status &= self.parent.dg1.D.set(dd)
            status &= self.parent.dg1.E.set(ee)
            status &= self.parent.dg1.F.set(ff)
            status &= self.parent.dg1.G.set(gg)
            status &= self.parent.dg1.H.set(hh)
            status &= self.parent.dg2.A.set(0)
            status &= self.parent.dg2.B.set(0.0005)

            # Set AreaDetector
            status &= self.parent.cam.acquire_time.set(Efccd)

        # Now do period
        if exp[1] is not None:
            if exp[1] < (Efccd + self._readout):
                p = Efccd + self._readout
            else:
                p = exp[1]

        status &= self.parent.cam.acquire_period.set(p)

        if exp[2] is not None:
            status &= self.parent.cam.num_images.set(exp[2])

        return status

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
        import time as ttime

        if not self.trigger_staged:
            self._trigger_stage()
            self.trigger_staged = True

        return super().trigger()

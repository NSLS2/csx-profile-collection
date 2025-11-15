import logging
from typing import Union, Optional
from functools import reduce

import networkx as nx
from ophyd import (EpicsScaler, EpicsSignal, EpicsSignalRO, Device, BlueskyInterface,
                   SingleTrigger, HDF5Plugin, ImagePlugin, StatsPlugin,
                   ROIPlugin, TransformPlugin, OverlayPlugin, ProsilicaDetector, TIFFPlugin, Signal, Staged, CamBase)

from ophyd.areadetector.cam import AreaDetectorCam, ADBase
from ophyd.areadetector.detectors import DetectorBase
from ophyd.areadetector.filestore_mixins import FileStoreHDF5IterativeWrite, FileStoreTIFFIterativeWrite, resource_factory
from ophyd.areadetector import ADComponent, EpicsSignalWithRBV
from ophyd.areadetector.plugins import PluginBase, ProcessPlugin, HDF5Plugin_V22, TIFFPlugin_V22, CircularBuffPlugin_V34, CircularBuffPlugin, PvaPlugin
from ophyd import Component as Cpt, DeviceStatus
from ophyd.device import FormattedComponent as FCpt
from ophyd import AreaDetector
from ophyd.status import Status, SubscriptionStatus
from pathlib import PurePath
import time as ttime
import itertools

from ophyd.sim import NullStatus

from .devices import DelayGenerator
from .scaler import StruckSIS3820MCS

import numpy as np

from .stats_plugin import StatsPluginCSX

logger = logging.getLogger(__name__)
DEFAULT_TIMEOUT = 10  # Seconds


class ExternalFileReference(Signal):
    """
    A pure software signal where a Device can stash a datum_id.

    For example, it can store timestamps from HDF5 files. It needs
    a `shape` because an HDF5 file can store multiple frames which
    have multiple timestamps.
    """
    def __init__(self, *args, shape, **kwargs):
        super().__init__(*args, **kwargs)
        self.shape = shape

    def describe(self):
        res = super().describe()
        res[self.name].update(
            dict(external="FILESTORE:", dtype="array", shape=self.shape)
        )
        return res


class TriggerStatus(DeviceStatus):
    """
    TODO: REMOVE WHEN THIS PR IS RELEASED IN A NEW CONDA ENV: https://github.com/bluesky/ophyd/pull/1240 
    """

    def __init__(
        self, tracking_signal: Signal, target: Signal | int, device, *args, **kwargs
    ):
        super().__init__(device, *args, **kwargs)
        self.start_ts = ttime.time()
        self.tracking_signal = tracking_signal

        # Notify watchers (things like progress bars) of new values
        # at the device's natural update rate.
        if not self.done:
            self.tracking_signal.subscribe(self._notify_watchers)
            # some state needed only by self._notify_watchers
            self._name = self.device.name
            self._initial_count = self.tracking_signal.get()
            self._target_count = target.get() if isinstance(target, Signal) else target

    @property
    def target(self):
        return self._target_count

    def watch(self, func):
        self._watchers.append(func)

    def __and__(self, other):
        return super().__and__(other)

    def __or__(self, other):
        return super().__or__(other)

    def __xor__(self, other):
        return super().__xor__(other)

    def _notify_watchers(self, value, *args, **kwargs):
        # *args and **kwargs catch extra inputs from pyepics, not needed here
        if self.done:
            self.tracking_signal.clear_sub(self._notify_watchers)
        if not self._watchers:
            return
        # Always start progress bar at 0 regardless of starting value of
        # array_counter.
        current = value - self._initial_count
        target = self._target_count - self._initial_count
        time_elapsed = ttime.time() - self.start_ts
        progress = (current / target)
        try:
            proportion_remaining = 1 - progress
        except ZeroDivisionError:
            proportion_remaining = 0
        except Exception:
            proportion_remaining = None
            time_remaining = None
        else:
            time_remaining = time_elapsed / proportion_remaining if proportion_remaining != 0 else 0
        for watcher in self._watchers:
            watcher(
                name=self._name,
                current=current,
                initial=0,
                target=target,
                unit="images",
                precision=0,
                fraction=proportion_remaining,
                time_elapsed=time_elapsed,
                time_remaining=time_remaining,
            )


class NDCircularBuffTriggerStatus(TriggerStatus):
    """
    TODO: REMOVE WHEN THIS PR IS RELEASED IN A NEW CONDA ENV: https://github.com/bluesky/ophyd/pull/1240 
    """

    def __init__(self, device, *args, **kwargs):
        if not hasattr(device, "cb"):
            raise RuntimeError(
                "NDCircularBuffTriggerStatus must be initialized with a device that has a CircularBuffPlugin"
            )
        super().__init__(
            device.cb.post_trigger_qty, device.cb.post_count, device, *args, **kwargs
        )

class NDHDF5WriteStatus(TriggerStatus):
    """
    TODO: REMOVE WHEN THIS PR IS RELEASED IN A NEW CONDA ENV: https://github.com/bluesky/ophyd/pull/1240 
    """

    def __init__(self, device, target_count, *args, **kwargs):
        if not hasattr(device, "hdf5"):
            raise RuntimeError(
                "NDHDF5WriteStatus must be initialized with a device that has a HDF5Plugin"
            )
        super().__init__(
            device.hdf5.num_captured, target=target_count, *args, **kwargs
        )

class ContinuousAcquisitionTrigger(BlueskyInterface):
    """
    TODO: REMOVE WHEN THIS PR IS RELEASED IN A NEW CONDA ENV: https://github.com/bluesky/ophyd/pull/1240 
    This trigger mixin class takes frames from a circular buffer filled
    by continuous acquisitions from the detector.

    We assume that the circular buffer is pre-configured and this is what
    will be "triggered" instead of the detector.

    In practice, this means that all other plugins should be configured to be
    downstream of the circular buffer, rathern than the detector driver.
    """

    _status_type = NDCircularBuffTriggerStatus

    def __init__(self, *args, image_name=None, **kwargs):
        super().__init__(*args, **kwargs)
        if image_name is None:
            image_name = "_".join([self.name, "image"])
        self._image_name = image_name

        if not hasattr(self, "cam"):
            raise RuntimeError("Detector must have a camera configured.")

        if not hasattr(self, "cb"):
            raise RuntimeError("Detector must have a CircularBuffPlugin configured.")

        # Order of operations is important here.
        self.stage_sigs.update(
            [
                ("cam.image_mode", self.cam.ImageMode.CONTINUOUS),  # 'Continuous' mode
                ("cam.acquire", 1),  # Start acquiring
                ("cb.flush_on_soft_trigger", 0),  # Flush the buffer on new image
                ("cb.preset_trigger_count", 0),  # Keep the buffer capturing forever
                # TODO: Figure out why this leaks an extra frame
                # Tested this with the HDF5 plugin and it writes an extra frame to
                # the file when `pre_count` is non-zero.
                # Possibly a bug in the NDCircularBuff plugin?
                ("cb.pre_count", 0),  # The number of frames to take before the trigger
                ("cb.capture", 1),  # Start filling the buffer
            ]
        )
        self._trigger_signal = self.cb.trigger_
        self._status = None

    def stage(self):
        self._trigger_signal.subscribe(self._trigger_changed)
        super().stage()

    def unstage(self):
        super().unstage()
        self._trigger_signal.clear_sub(self._trigger_changed)

    def trigger(self):
        if self._staged != Staged.yes:
            raise RuntimeError(
                "This detector is not ready to trigger."
                "Call the stage() method before triggering."
            )
        self._status = self._status_type(self)
        self._trigger_signal.put(1, wait=False)
        self.generate_datum(self._image_name, ttime.time(), {})
        return self._status

    def _trigger_changed(self, value=None, old_value=None, **kwargs):
        if self._status is None:
            return
        if (old_value == 1) and (value == 0):
            self._status.set_finished()
            self._status = None


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
        "gain", 'prnu',
        "tec",
        "bin_mode",
        # there are functions to add attrs for image corrections when testing (see .startup.detectors)
        # we should add more functionality to enable/disable triggering. if enabled, add atts
    )
    wait_for_plugins = Cpt(EpicsSignal, "WaitForPlugins", string=True, kind="hinted")
    capture = Cpt(EpicsSignalWithRBV, "Capture", kind="omitted")
    gain = Cpt(EpicsSignalWithRBV, "GainMode", string=True, kind="config")
    prnu = Cpt(EpicsSignalWithRBV, "PRNU", string=True, kind="config")
    tec = Cpt(EpicsSignalWithRBV, "TEC", string=True, kind="config")
    auto_tec = Cpt(EpicsSignalWithRBV, "AutoTEC", string=True, kind="config")
    bin_mode = Cpt(EpicsSignalWithRBV, "BinMode", string=True, kind="config")
    temperature = Cpt(EpicsSignalRO, "Temperature_RBV")
    retry_on_timeout = Cpt(EpicsSignalWithRBV, "RetryOnTimeout", string=True, kind="config")
    num_retries = Cpt(EpicsSignalWithRBV, "NumRetries", kind="config")
    frame_speed = Cpt(EpicsSignalWithRBV, "FrameSpeed", kind="config")
    bit_depth = Cpt(EpicsSignalWithRBV, "BitDepth", kind="config")
    auto_exposure = Cpt(EpicsSignalWithRBV, "AutoExposure", string=True, kind="config")
    fan_gear = Cpt(EpicsSignalWithRBV, "FanGear", string=True, kind="config")
    auto_levels = Cpt(EpicsSignalWithRBV, "AutoLevels", string=True, kind="config")
    histogram = Cpt(EpicsSignalWithRBV, "Histogram", string=True, kind="config")
    enhance = Cpt(EpicsSignalWithRBV, "Enhance", string=True, kind="config")
    defect_correction = Cpt(EpicsSignalWithRBV, "DefectCorrection", string=True, kind="config")
    enable_denoise = Cpt(EpicsSignalWithRBV, "EnableDenoise", string=True, kind="config")
    flat_correction = Cpt(EpicsSignalWithRBV, "FlatCorrection", string=True, kind="config")
    dyn_rge_correction = Cpt(EpicsSignalWithRBV, "DynRgeCorrection", string=True, kind="config")
    frame_format = Cpt(EpicsSignalWithRBV, "FrameFormat", string=True, kind="config")
    brightness = Cpt(EpicsSignalWithRBV, "Brightness", kind="config")
    black_level = Cpt(EpicsSignalWithRBV, "BlackLevel", kind="config")
    sharpness = Cpt(EpicsSignalWithRBV, "Sharpness", kind="config")
    noise_level = Cpt(EpicsSignalWithRBV, "NoiseLevel", kind="config")
    hdr_k = Cpt(EpicsSignalWithRBV, "HDRK", kind="config")
    gamma = Cpt(EpicsSignalWithRBV, "Gamma", kind="config")
    contrast = Cpt(EpicsSignalWithRBV, "Contrast", kind="config")
    left_levels = Cpt(EpicsSignalWithRBV, "LeftLevels", kind="config")
    right_levels = Cpt(EpicsSignalWithRBV, "RightLevels", kind="config")
    trigger_edge = Cpt(EpicsSignalWithRBV, "TriggerEdge", string=True, kind="config")
    trigger_exposure = Cpt(EpicsSignalWithRBV, "TriggerExposure", string=True, kind="config")
    trigger_delay = Cpt(EpicsSignalWithRBV, "TriggerDelay", kind="config")
    software_trigger = Cpt(EpicsSignal, "SoftwareTrigger", string=True)
    trigger_out1_mode = Cpt(EpicsSignalWithRBV, "TriggerOut1Mode", string=True)
    trigger_out1_edge = Cpt(EpicsSignalWithRBV, "TriggerOut1Edge", string=True)
    trigger_out1_delay = Cpt(EpicsSignalWithRBV, "TriggerOut1Delay", kind="config")
    trigger_out1_width = Cpt(EpicsSignalWithRBV, "TriggerOut1Width", kind="config")
    trigger_out2_mode = Cpt(EpicsSignalWithRBV, "TriggerOut2Mode", string=True)
    trigger_out2_edge = Cpt(EpicsSignalWithRBV, "TriggerOut2Edge", string=True)
    trigger_out2_delay = Cpt(EpicsSignalWithRBV, "TriggerOut2Delay", kind="config")
    trigger_out2_width = Cpt(EpicsSignalWithRBV, "TriggerOut2Width", kind="config")
    trigger_out3_mode = Cpt(EpicsSignalWithRBV, "TriggerOut3Mode", string=True)
    trigger_out3_edge = Cpt(EpicsSignalWithRBV, "TriggerOut3Edge", string=True)
    trigger_out3_delay = Cpt(EpicsSignalWithRBV, "TriggerOut3Delay", kind="config")
    trigger_out3_width = Cpt(EpicsSignalWithRBV, "TriggerOut3Width", kind="config")
    acquire_period = ADComponent(EpicsSignalWithRBV, "AcquirePeriod", tolerance=0.01, timeout=5, kind="config")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)



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
    _default_read_attrs = ("time_stamp",)
    # Captures the datum id for the timestamp recorded in the HDF5 file
    time_stamp = Cpt(ExternalFileReference, value="", kind="normal", shape=[])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # In CSS help: "N < 0: Up to abs(N) new directory levels will be created"
        self.stage_sigs.update({"create_directory": -3})
        # last=False turns move_to_end into move_to_start. Yes, it's silly.
        self.stage_sigs.move_to_end("create_directory", last=False)

        # Setup for timestamping using the detector
        self._ts_datum_factory = None
        self._ts_resource_uid = ""
        self._ts_counter = None

    def stage(self):
        # Start the timestamp counter
        self._ts_counter = itertools.count()
        return super().stage()

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

    def _generate_resource(self, resource_kwargs):
        super()._generate_resource(resource_kwargs)
        fn = PurePath(self._fn).relative_to(self.reg_root)

        # Update the shape that describe() will report
        # Multiple images will have multiple timestamps
        fpp = self.get_frames_per_point()
        self.time_stamp.shape = [fpp] if fpp > 1 else []

        # Query for the AD_HDF5_TS timestamp
        # See https://github.com/bluesky/area-detector-handlers/blob/master/area_detector_handlers/handlers.py#L230
        resource, self._ts_datum_factory = resource_factory(
            spec="AD_HDF5_DET_TS",
            root=str(self.reg_root),
            resource_path=str(fn),
            resource_kwargs=resource_kwargs,
            path_semantics=self.path_semantics,
        )

        self._ts_resource_uid = resource["uid"]
        self._asset_docs_cache.append(("resource", resource))

    def generate_datum(self, key, timestamp, datum_kwargs):
        ret = super().generate_datum(key, timestamp, datum_kwargs)
        datum_kwargs = datum_kwargs or {}
        datum_kwargs.update({"point_number": next(self._ts_counter)})
        # make the timestamp datum, in this case we know they match
        datum = self._ts_datum_factory(datum_kwargs)
        datum_id = datum["datum_id"]

        # stash so that we can collect later
        self._asset_docs_cache.append(("datum", datum))
        # put in the soft-signal so it gets auto-read later
        self.time_stamp.put(datum_id)
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


class PvaPluginWithPluginAttributes(PvaPlugin):
    nd_array_port = Cpt(EpicsSignalWithRBV, "NDArrayPort", kind="config")
    enable = Cpt(EpicsSignalWithRBV, "EnableCallbacks", string=True, kind="config")


def set_plugin_graph(graph: dict[PluginBase, Union[CamBase, PluginBase]]) -> None:
    for target, source in graph.items():
        target.nd_array_port.set(source.port_name.get()).wait(0.5)

    for plugin in graph.keys():
        plugin.enable.set(1).wait(0.5)


class AxisCamBase(AreaDetector):
    """
    Class for Axis detector with HDF5 file saving.

    The IOC is currently hosted on a Windows machine so the
    `write_path_template` must be specified as a Windows path.
    """
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
    proc2 = Cpt(ProcessPlugin, 'Proc2:')
    trans1 = Cpt(TransformPlugin, 'Trans1:')
    trans2 = Cpt(TransformPlugin, 'Trans2:')
    over1 = Cpt(OverlayPlugin, 'Over1:')
    hdf5 = Cpt(HDF5PluginWithFileStorePlain,
              suffix='HDF1:',
              read_path_template='/nsls2/data/csx/legacy/axis_data/hdf5/%Y/%m/%d',
              root='/nsls2/data/csx/legacy/axis_data/hdf5',
              write_path_template='/nsls2/data/csx/legacy/axis_data/hdf5/%Y/%m/%d',
              path_semantics='posix')
    pva1 = Cpt(PvaPluginWithPluginAttributes, 'Pva1:')
    _default_plugin_graph: Optional[dict[PluginBase, Union[CamBase, PluginBase]]] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hdf5.kind = "normal"
        self.hdf5.file_path.path_semantics = "posix"
        self.ensure_acquiring = False
        # Camera is currently UInt16, the default is wrong at Int8
        self.cam.data_type.set("UInt16")
        self.additional_timeout = 0.0

        self._use_default_plugin_graph: bool = True
        self._plugin_graph_cache: Optional[dict[PluginBase, Union[CamBase, PluginBase]]] = None

    @property
    def default_plugin_graph(self) -> Optional[dict[PluginBase, Union[CamBase, PluginBase]]]:
        return self._default_plugin_graph

    def disable_default_plugin_graph(self):
        logger.warning(f"Disabling default plugin graph for {self.name}. This can lead to unexpected behavior.")
        self._use_default_plugin_graph = False

    def enable_default_plugin_graph(self):
        self._use_default_plugin_graph = True

    def _stage_plugin_graph(self, plugin_graph: dict[PluginBase, Union[CamBase, PluginBase]]):
        for target, source in plugin_graph.items():
            self.stage_sigs[target.nd_array_port] = source.port_name.get()
            self.stage_sigs[target.enable] = True

    def reset_plugin_graph(self):
        """Resets the plugin graph to the default state."""
        set_plugin_graph(self.default_plugin_graph)

    def stage(self):
        # Adjust timeout relative to acquire_time and acquire_period
        exposure_time = self.cam.acquire_time.get()
        acquire_period = self.cam.acquire_period.get()
        self.additional_timeout = exposure_time + acquire_period
        self.cam.acquire._timeout += self.additional_timeout

        # Configure the plugin graph to use the default configuration
        # Must use `stage_sigs` in order to reset on unstage
        if self._use_default_plugin_graph and self.default_plugin_graph is not None:
            self._stage_plugin_graph(self.default_plugin_graph)

        ret = super().stage()
        return ret

    def unstage(self):
        super().unstage()

        # Adjust timeout back to original value
        self.cam.acquire._timeout -= self.additional_timeout

    def ensure_nonblocking(self):
        self.stage_sigs["cam.wait_for_plugins"] = "No"
        for c in self.component_names:
            cpt = getattr(self, c)
            if cpt is self:
                continue
            if hasattr(cpt, "ensure_nonblocking"):
                cpt.ensure_nonblocking()

    def store_acquiring_state(self):
        """Store acquiring state if we are in continuous mode"""
        self.ensure_acquiring = (
            self.cam.image_mode.get(as_string=True) == "Continuous" and
            self.cam.acquire.get() == 1
        )


    def restore_acquiring_state(self):
        """Restore acquiring state if we were before"""
        # If the image mode was continuous, start acquiring again
        acquiring = self.cam.acquire.get()
        if self.ensure_acquiring and acquiring == 0:
            self.cam.acquire.set(1).wait(3.0)
        # Otherwise, we were in continuous mode but not acquiring
        # so stop the acquisiton again
        elif (not self.ensure_acquiring
              and self.cam.image_mode.get(as_string=True) == "Continuous"
              and acquiring == 1):
            self.cam.acquire.set(0).wait(3.0)


class StandardAxisCam(SingleTrigger, AxisCamBase):
    """Axis detector that runs in multiple acquisition mode.

    It runs in non-blocking mode by default so that capturing
    frames is not slowed down by the cumulative execution time of the plugins.

    This may mean that the file writing is not complete before subsequent acquisitions.

    The defualt plugin configuration is:
        AXIS1 -> HDF5
              -> STATS5
              -> PROC1 -> TRANS1 -> ROI1 -> STATS1
                                 -> ROI2 -> STATS2
                                 -> ROI3 -> STATS3
                                 -> ROI4 -> STATS4
              -> PROC2 -> TRANS2 -> OVER1 -> PVA1
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stage_sigs[self.cam.wait_for_plugins] = "No"
        self.stage_sigs[self.cam.trigger_mode] = "Software"
        # Changing image_mode stops acquisition every time
        # so using stage_sigs doesn't work
        self.stage_sigs.pop("cam.acquire")
        self.ensure_nonblocking()

        self._default_plugin_graph = {
            self.hdf5: self.cam,
            self.stats5: self.cam,
            self.proc1: self.cam,
            self.proc2: self.cam,
            self.trans1: self.proc1,
            self.trans2: self.proc2,
            self.roi1: self.trans1,
            self.roi2: self.trans1,
            self.roi3: self.trans1,
            self.roi4: self.trans1,
            self.stats1: self.roi1,
            self.stats2: self.roi2,
            self.stats3: self.roi3,
            self.stats4: self.roi4,
            self.over1: self.trans2,
            self.pva1: self.over1,
        }

    def stage(self):
        self.store_acquiring_state()

        # Manually stop acquiring
        if self.cam.acquire.get() == 1:
            self.cam.acquire.set(0).wait(3.0)

        # If not in software trigger mode and we are capturing, stop capturing so we can switch it
        if self.cam.trigger_mode.get(as_string=True) != "Software" and self.cam.capture.get() == 1:
            self.cam.capture.set(0).wait(3.0)

        # Process stage_sigs (includes trigger_mode switch to software)
        ret = super().stage()

        # If not capturing, start capturing to warm up the detector
        if self.cam.capture.get() == 0:
            self.cam.capture.set(1).wait(3.0)

        return ret

    def unstage(self):
        # Stop capturing if still
        if self.cam.capture.get() == 1:
            self.cam.capture.set(0).wait(3.0)

        # Reset stage_sigs to original values
        super().unstage()
        self.restore_acquiring_state()


class ContinuousAxisCam(ContinuousAcquisitionTrigger, AxisCamBase):
    """Axis detector that runs in continuous acquisition mode.

    It uses a circular buffer plugin to trigger capturing frames
    from the detector *driver* instead of directly from the detector.

    It runs in non-blocking mode by default so that any displays can
    update asynchronously from Bluesky plans.

    The defualt plugin configuration is:
        AXIS1 -> CB1 -> HDF5
              -> CB1 -> STATS5
              -> CB1 -> PROC1 -> TRANS1 -> ROI1 -> STATS1
                                        -> ROI2 -> STATS2
                                        -> ROI3 -> STATS3
                                        -> ROI4 -> STATS4
              -> PROC2 -> TRANS2 -> OVER1 -> PVA1
    """
    cb = Cpt(CircularBuffPlugin_V34, "CB1:")
    # This is for regulating exposure during possible movements
    should_skip_frame = Cpt(Signal, kind="config", value=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Changing the image_mode stops acquisition already
        # so we can't use stage_sigs
        self.stage_sigs[self.cam.trigger_mode] = "Free Run"
        self.stage_sigs.pop("cam.acquire")
        self.ensure_nonblocking()

        self._write_status: TriggerStatus | None = None
        self._num_triggered = 0
        self._trigger_mode_switched = False

        self._default_plugin_graph = {
            self.cb: self.cam,
            self.hdf5: self.cb,
            self.stats5: self.cb,
            self.proc1: self.cb,
            self.proc2: self.cam,
            self.trans1: self.proc1,
            self.trans2: self.proc2,
            self.roi1: self.trans1,
            self.roi2: self.trans1,
            self.roi3: self.trans1,
            self.roi4: self.trans1,
            self.stats1: self.roi1,
            self.stats2: self.roi2,
            self.stats3: self.roi3,
            self.stats4: self.roi4,
            self.over1: self.trans2,
            self.pva1: self.over1,
        }

    def stage(self):
        self.store_acquiring_state()
        self.stage_sigs[self.cb.post_count] = self.cam.num_images.get()

        # If not in free run mode and we are capturing, check if we are acquiring and stop it if so,
        # then stop capturing. We need to do this to switch the trigger mode to free run.
        if self.cam.trigger_mode.get(as_string=True) != "Free Run" and self.cam.capture.get() == 1:
            if self.cam.acquire.get() == 1:
                self.cam.acquire.set(0).wait(3.0)
            self.cam.capture.set(0).wait(3.0)
            self._trigger_mode_swtiched = True

        # Process stage_sigs (includes trigger_mode switch to free run)
        res = super().stage()

        self._num_triggered = 0

        # Manually start acquiring
        if self.cam.acquire.get() == 0:
            self.cam.acquire.set(1).wait(3.0)

        # Set up subscriptions for all leaf-node plugins
        # We need to wait for all leaf-node plugins downstream of the circular buffer to finish writing
        # before we can trigger the next acquisition.
        asyn_graph: tuple[nx.DiGraph, dict[str, ADBase]] = self.get_asyn_digraph()
        graph = asyn_graph[0]
        port_map = asyn_graph[1]
        reachable_nodes = nx.descendants(graph, self.cb.port_name.get())
        self._leaf_plugins: list[PluginBase] = [port_map[node] for node in reachable_nodes if graph.out_degree(node) == 0 and isinstance(port_map[node], PluginBase)]

        # Reset array counters to 0 so we can properly wait
        for plugin in self._leaf_plugins:
            plugin.array_counter.set(0).wait()

        return res

    def _skip_frame(self):
        current_frame_number = self.cam.num_images_counter.get()
        def frame_changed(value, old_value, **kwargs):
            return value > current_frame_number
        # Wait until one full frame finishes, timeout indicates that something is wrong
        SubscriptionStatus(self.cam.num_images_counter, frame_changed).wait(timeout=self.cam.acquire_period.get() * 2 - 1e-4)

    def _plugin_complete(self, old_value, value, **kwargs) -> bool:
        return value == self.cb.post_count.get() * self._num_triggered

    def trigger(self):
        """
        Since we are non-blocking at the EPICS level, we want to wait for the HDF5
        plugin to finish writing before we trigger the next acquisition.
        """
        if self.should_skip_frame.get():
            # We must wait until this first frame is complete before we can
            # start exposing a new frame. Otherwise, we may grab a frame
            # that was exposing during a movement (of a motor or energy or temperature value)
            self._skip_frame()

        # Trigger the circular buffer with the fully exposed frame
        self._num_triggered += 1
        super().trigger()

        # Return a Status that is done when all leaf-node plugins are complete
        statuses = [SubscriptionStatus(plugin.array_counter, self._plugin_complete) for plugin in self._leaf_plugins]
        return reduce(lambda a, b: a & b, statuses)

    def unstage(self):
        # If the trigger mode was switched during stage, we need to stop
        # acquiring and capturing to change it back.
        if self._trigger_mode_switched:
            if self.cam.acquire.get() == 1:
                self.cam.acquire.set(0).wait(3.0)
            if self.cam.capture.get() == 1:
                self.cam.capture.set(0).wait(3.0)

        super().unstage()
        self._num_triggered = 0
        self.restore_acquiring_state()


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

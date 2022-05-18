from ophyd import (EpicsScaler, EpicsSignal, EpicsSignalRO, Device,
                   SingleTrigger, HDF5Plugin, ImagePlugin, StatsPlugin,
                   ROIPlugin, TransformPlugin, OverlayPlugin, ProsilicaDetector)

from ophyd.areadetector.cam import AreaDetectorCam
from ophyd.areadetector.detectors import DetectorBase
from ophyd.areadetector.filestore_mixins import FileStoreHDF5IterativeWrite
from ophyd.areadetector import ADComponent, EpicsSignalWithRBV
from ophyd.areadetector.plugins import PluginBase, ProcessPlugin
from ophyd import Component as Cpt
from ophyd.device import FormattedComponent as FCpt
from ophyd import AreaDetector
from ophyd.utils import set_and_wait
import time as ttime
import itertools

from ophyd.sim import NullStatus

from .devices import DelayGenerator
from .scaler import StruckSIS3820MCS

import numpy as np

from .stats_plugin import StatsPluginCSX


class StandardCam(SingleTrigger, AreaDetector):#TODOpmab is there somethine more standard for prosilica? seems only used on prosilica. this does stats, but no image saving (unsure if easy to configure or not and enable/disable)
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


class MonitorStatsCam(SingleTrigger, AreaDetector): #TODO does this work or are we hacking EpicsSignals in custom plans
    stats1 = Cpt(StatsPlugin, "Stats1:")
    roi1 = Cpt(ROIPlugin, "ROI1:")
    proc1 = Cpt(ProcessPlugin, "Proc1:")

    def subscribe(self, *args, **kwargs):
        return self.stast1.cenx.subscribe(*args, **kwargs)

    def unbuscribe(self, *args, **kwargs):
        return self.stast1.cenx.unsubscribe(*args, **kwargs)


class HDF5PluginWithFileStorePlain(HDF5Plugin, FileStoreHDF5IterativeWrite): ##SOURCED FROM BELOW FROM FCCD WITH SWMR removed
    # AD v2.2.0 (at least) does not have this. It is present in v1.9.1.
    file_number_sync = None

    def get_frames_per_point(self):
        return self.parent.cam.num_images.get()

    def make_filename(self):
        # stash this so that it is available on resume
        self._ret = super().make_filename()
        return self._ret


#class StandardProsilicaSaving(StandardProsilica): #TODOpmab original from SIX, but moved up and removed SIX custom StandardProsilica
class StandardProsilicaSaving(StandardCam):#TODOpmab just random guess by andi to save for dif_cam1,2,3 will disable rois
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_hdf5()

    def enable_hdf5(self):
        self.hdf5 = Cpt(HDF5PluginWithFileStorePlain,
              suffix='HDF1:',
              #write_path_template='/nsls2/data/csx/legacy/prosilica_data/%Y/%m/%d',##TODOpmab - fix path if this works
              write_path_template='/nsls2/data/csx/legacy/datajunk/%Y/%m/%d',
              root='/nsls2/data/csx/legacy')
        ##TODOpmab - priority2, if works then stretch-TODO-overlays and image seperate (2 Tiffs or 2 H5 or 1 of each, but need to rebuild IOC for that)

    def disable_hdf5(self):
        self.hdf5 = None


###  #TODOpmab 2nd priority - STOLEN FROM SIX 21-areadetector.py --
# class StandardProsilica(SingleTrigger, ProsilicaDetector):
#     def __init__(self,*args,**kwargs):
#         super().__init__(*args,**kwargs)

#         for n in [1, 5]:
#             stats = getattr(self, f'stats{n}')
#             stats.kind |= Kind.normal
#             stats.total.kind = Kind.hinted
        
#     #image = Cpt(ImagePlugin, 'image1:')
#     stats1 = Cpt(StatsPlugin, 'Stats1:')
#     stats2 = Cpt(StatsPlugin, 'Stats2:')
#     stats3 = Cpt(StatsPlugin, 'Stats3:')
#     stats4 = Cpt(StatsPlugin, 'Stats4:')
#     stats5 = Cpt(StatsPlugin, 'Stats5:')
#     #trans1 = Cpt(TransformPlugin, 'Trans1:')
#     roi1 = Cpt(ROIPlugin, 'ROI1:')
#     roi2 = Cpt(ROIPlugin, 'ROI2:')
#     roi3 = Cpt(ROIPlugin, 'ROI3:')
#     roi4 = Cpt(ROIPlugin, 'ROI4:')
#     #proc1 = Cpt(ProcessPlugin, 'Proc1:')

# #TODOpmab 2nd priority - STOLEN FROM SIX 21-areadetector.py -- but we don't ever change ROIs, but we would want the ROIs to 
# be exactly the same metadata structure as FCCD.  This looks like the same (fccd.roi1.size.x )
# class StandardProsilicaROI(StandardProsilica):
#     '''
#     A class that is used to add the attributes 'roi_enable', 'roi_set', 'roi_read' and the group ('roiN_minM', roiN_sizeM) 
#     where N is 1-4 and M is x,y or z. to a camera with the roi plugin enabled.
#     '''    

#     def __init__(self,*args,**kwargs):
#         super().__init__(*args,**kwargs)
        
#         for i in range(1, 4):
#             for axis in ['x','y','z']:
#                 setattr(self,'roi{}_min{}'.format(i, axis),
#                         getattr(self, 'roi' + str(i) + '.min_xyz.min_{}'.format(axis)))
#                 setattr(self,'roi{}_size{}'.format(i, axis),
#                         getattr(self, 'roi' + str(i) + '.size.{}'.format(axis)))
    
    
#     def roi_set(self,min_x, size_x, min_y, size_y, min_z=None, size_z=None, roi_num=1):
#         ''' 
#         An attribute function for the camera that allows the user to set an roi size and position. setting
#         any of the values to 'None' means they are ignored(left as is).
#         TODO add a 'set' method tothe ROIPlugin class to supprt 'cam.roi1.set(...)'
            
#         Parameters
#         ----------
#         min_x : integer
#             The pixel number position of the left edge of the ROI.
#         size_x : integer
#             The pixel number width of the ROI.
#         min_y : integer
#             The pixel number position of the bottom edge of the ROI.
#         size_y : integer
#             The pixel number height of the ROI.
#         min_z : integer,optional
#             The pixel number minima of the intensity region of the ROI.
#         size_z : integer,optional
#             The pixel number maxima of the intensity region of the ROI.
#         roi_num : integer, optional
#             The roi number to act, default is 1 and it must be 1,2,3 or 4.        
#         '''

#         if min_x is not None:
#             getattr(self, 'roi' + str(roi_num) + '.min_xyz.min_x').put(min_x)
#         if size_x is not None:
#             getattr(self, 'roi' + str(roi_num) + '.size.x').put(size_x)
#         if min_y is not None:
#             getattr(self, 'roi' + str(roi_num) + '.min_xyz.min_y').put(min_y)
#         if size_y is not None:
#             getattr(self, 'roi' + str(roi_num) + '.size.y').put(size_y)
#         if min_z is not None:
#             getattr(self, 'roi' + str(roi_num) + '.min_xyz.min_z').put(min_z)
#         if size_z is not None:
#             getattr(self, 'roi' + str(roi_num) + '.size.z').put(size_z)

#     def roi_read(self, roi_num=1):
#         ''' 
#         An attribute function for the camera that allows the user to read the current values of  
#         an roi size and position.
#         Usage hints: to extract a specific value use "cam_name.roi_read()['keyword']" where 'keyword'
#         is min_x, size_x, min_y, size_y, min_z, size_z or status.        
            
#         Parameters
#         ----------
        
#         roi_num : integer, optional
#             The roi number to act, default is 1 and it must be 1,2,3 or 4.  
        
#         roi_dict : output
#             A dictionary which gives the current roi positions in the form: 
#             {'min_x':value,'size_x':value,'min_y':value,'size_y':value,'min_z':value,'size_z':value,'status':status}
#         '''
#         roi_dict={'min_x' : getattr(self, 'roi' + str(roi_num) + '.min_xyz.min_x').get(),
#                   'size_x': getattr(self, 'roi' + str(roi_num) + '.size.x').get(),
#                   'min_y': getattr(self, 'roi' + str(roi_num) + '.min_xyz.min_y').get(),
#                   'size_y': getattr(self, 'roi' + str(roi_num) + '.size.y').get(),
#                   'min_z' : getattr(self, 'roi' + str(roi_num) + '.min_xyz.min_z').get(),
#                   'size_z' : getattr(self, 'roi' + str(roi_num) + '.size.z').get(),
#                   'status' : getattr(self, 'roi' + str(roi_num) + '.enable').get()}
        
#         return roi_dict

#     def roi_enable(self, status, roi_num=1):
#         ''' 
#         An attribute function for the camera that allows the user to enable or disable an ROI.
      
            
#         Parameters
#         ----------
        
#         status : string
#             The string indicating the status to set for the ROI, must be 'Enable' or 'Disable'.
        
#         roi_num : integer, optional
#             The roi number to act, default is 1 and it must be 1,2,3 or 4.    
#         '''   

#         if status is 'Enable' or status is 'Disable':
#             getattr(self, 'roi' + str(roi_num) + '.enablE').set(status)
#         else:
#             raise RuntimeError('in roi_enable status must be Enable or Disable')

 #TODOpmab 2nd priority - STOLEN FROM SIX 21-areadetector.py --
# class StandardProsilicaSaving(StandardProsilicaROI):   ##TODOpmab SIX ORIGINAL to save, at one time only save or ROI so don't trust this works
#     hdf5 = Cpt(HDF5PluginWithFileStore,
#               suffix='HDF1:',
#               write_path_template='/nsls2/data/csx/legacy/prosilica_data/%Y/%m/%d',
#               root='/nsls2/data/csx/legacy')


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
        set_and_wait(self.cam.acquire, 0)
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
        set_and_wait(self.hdf5.capture, set_val)
        #val = self.hdf5.capture.get()
        ## Julien fix to ensure these are set correctly
        #print("pausing FCCD")
        #while (np.abs(val-set_val) > 1e-6):
            #self.hdf5.capture.put(set_val)
            #val = self.hdf5.capture.get()

        return super().pause()

    def resume(self):
        set_val = 1
        set_and_wait(self.hdf5.capture, set_val)
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

from ophyd.device import (Component as C, DynamicDeviceComponent as DDC)
from ophyd import (EpicsScaler, EpicsSignal, EpicsSignalRO, Device, SingleTrigger, HDF5Plugin,
                           ImagePlugin, StatsPlugin, ROIPlugin, TransformPlugin)
from ophyd.areadetector.cam import AreaDetectorCam
from ophyd.areadetector.detectors import DetectorBase
from ophyd.areadetector.filestore_mixins import FileStoreHDF5IterativeWrite
from ophyd.areadetector import ADComponent, EpicsSignalWithRBV
from ophyd.areadetector.plugins import PluginBase, ProcessPlugin
from ophyd import Component as Cpt
from ophyd import AreaDetector
from ophyd.sim import NullStatus
from collections import OrderedDict
import bluesky.plans as bp

from ..devices.scaler import PrototypeEpicsScaler, StruckSIS3820MCS
from ..devices.areadetector import (StandardCam, NoStatsCam,
                                    ProductionCamStandard,
                                    ProductionCamTriggered,
                                    StageOnFirstTrigger,
                                    MonitorStatsCam,
                                    StandardProsilicaWithHDF5, StandardProsilicaWithTIFF, #TODOpmab - added to try to save (inspired from SIX)
                                    StandardAxisCam, ContinuousAxisCam, set_plugin_graph)

from ..startup import db

def _setup_stats(cam_in):
    for k in (f'stats{j}' for j in range(1, 6)):
        cam_in.read_attrs.append(k)
        getattr(cam_in, k).read_attrs = ['total']
        getattr(cam_in, k).total.kind = 'hinted'



# #TODO delete, it is already in diag6
# diag6_pid_threshold = EpicsSignal('XF:23ID1-BI{Diag:6-Cam:1}Stats1:CentroidThreshold',
#         name = 'diag6_pid_threshold')
# diag6new_pid_threshold = EpicsSignal('XF:23ID1-BI{Diag:8-Cam:1}Stats1:CentroidThreshold',
#         name = 'diag6new_pid_threshold')

# #
# Scalers both MCS and Standard
#

sclr = PrototypeEpicsScaler('XF:23ID1-ES{Sclr:1}', name='sclr')

for sig in sclr.channels.component_names:
    getattr(sclr.channels, sig).name = 'sclr_' + sig.replace('an', '')

mcs = StruckSIS3820MCS('XF:23ID1-ES{Sclr:1}', name='mcs')

#
# Diagnostic Prosilica Cameras
#
cam_fs1_hdf5 = StandardProsilicaWithHDF5('XF:23IDA-BI:1{FS:1-Cam:1}', name = 'cam_fs1_hdf5')

cam_diag2 = StandardCam('XF:23ID1-BI{Diag:2-Cam:1}', name='cam_diag2')#TODOpmab optional imagesave w/ stats always
_setup_stats(cam_diag2) #diamond diagnostic

## 20180726 needed to comment due to IOC1 problems
cam_slt1 = StandardCam('XF:23ID1-BI{Slt:1-Cam:1}', name='cam_slt1')
_setup_stats(cam_slt1)

cam_diag3 = StandardCam('XF:23ID1-BI{Diag:3-Cam:1}', name='cam_diag3')
_setup_stats(cam_diag3)

cam_diag6 = MonitorStatsCam('XF:23ID1-BI{Diag:6-Cam:1}', name='cam_diag6') #TODO testing

#cam_diag6 = NoStatsCam('XF:23ID1-BI{Diag:6-Cam:1}', name='diag6') #TODO revert above test
#cam_diag6.stats1.centroid_threshold.kind = :normal' ## maybe can only subscribe diag6? ##TODOrecord_threshold_for_every_scan_and_PV_put_complete
#cam_diag6.stats1.kind = 'normal'
cam_diag6_hdf5 = StandardProsilicaWithHDF5('XF:23ID1-BI{Diag:6-Cam:1}', name='cam_diag6_hdf5') #TODO replace with DSSI project
#_setup_stats_cen(cam_diag6_hdf5)
## 20180726 needed to comment due to IOC1 problems - probably ok now, but not used.
cam_dif = StandardCam('XF:23ID1-ES{Diag:5-Cam:1}', name='cam_dif')
cam_dif_hdf5 = StandardProsilicaWithHDF5('XF:23ID1-ES{Diag:5-Cam:1}', name='cam_dif_hdf5')
_setup_stats(cam_dif)
#_setup_stats_cen(cam_dif_hdf5)

cam_slt3 = StandardCam('XF:23ID1-ES{Dif-Cam:Beam}', name='cam_slt3')
#cam_slt3_hdf5 = StandardProsilicaWithHDF5('XF:23ID1-ES{Dif-Cam:Beam}', name='cam_slt3_hdf5') #TODO replace with DSSI project
_setup_stats(cam_slt3)
#_setup_stats_cen(cam_slt3_hdf5)

axis_standard = StandardAxisCam("XF:23ID1-ES{AXIS}", name='axis_standard')
_setup_stats(axis_standard)
axis_cont = ContinuousAxisCam("XF:23ID1-ES{AXIS}", name='axis_cont')
_setup_stats(axis_cont)
### more roi metadata at the end

def axis_add_image_correction_to_config_attr(axis_detector_in_use, remove = False):
    config_list = ['enhance', 'defect_correction', 'enable_denoise', 'flat_correction', 'dyn_rge_correction', 'frame_format', 'brightness', 'black_level','sharpness', 'noise_level', 'hdr_k', 'gamma', 'contrast', 'left_levels', 'right_levels']
    cam_config_list = ['cam.'+item for item in  config_list]
    if cam_config_list[0] in axis_detector_in_use.configuration_attrs:
        if remove:
            print('removing from configuration attrs')
            print('restart bluesky for now. remove and pop dont seem to work')
            #for item in cam_config_list:
            #    axis_detector_in_use.configuration_attrs.remove(item)
        else:
            print('probablly all are in configuration attrs')
    else:
        print('adding to configuration attrs')
        axis_detector_in_use.configuration_attrs.extend(cam_config_list)

# Setup on 2018/03/16 for correlating fCCD and sample position - worked 
# DON'T NEED STATS to take pictures of sample/optics
#dif_cam1 = StandardCam('XF:23ID1-ES{Dif-Cam:1}', name='dif_cam1' )
#_setup_stats(dif_cam2) #comment to disable
cam_dif_micro = StandardProsilicaWithTIFF('XF:23ID1-ES{Dif-Cam:1}', name='cam_dif_micro')
cam_dif_top = StandardProsilicaWithTIFF('XF:23ID1-ES{Dif-Cam:2}', name='cam_dif_top')
cam_dif_side = StandardProsilicaWithTIFF('XF:23ID1-ES{Dif-Cam:3}', name='cam_dif_side')##TODO think how to fix with trans plugin

## 20201219 - Machine studies for source characterization #TODO save also images like real detector
cam_fs = StandardCam('XF:23IDA-BI:1{FS:1-Cam:1}', name='cam_fs') #TODOpmab optional imagesave w/ stats always


#cam_pa= StandardCam('XF:23ID1-BI{Diag:7-Cam:1}', name='cam_pa') #TODOpmab optional imagesave w/ stats always
#_setup_stats(cam_pa)

### SWITCH AS NEEDED per experiment
### OPT1
#cam_bs = StandardCam('XF:23ID1-BI{Diag:8-Cam:1}', name='cam_bs') #TODOpmab optional imagesave w/ stats always
#_setup_stats(cam_bs)
#cam_bs_hdf5 = StandardProsilicaWithHDF5('XF:23ID1-BI{Diag:8-Cam:1}', name='cam_bs_hdf5') #TODO replace with DSSI project
#_setup_stats(cam_bs)
#_setup_stats(cam_bs_hdf5)


# FastCCD

fccd = StageOnFirstTrigger('XF:23ID1-ES{FCCD}',
#fccd = ProductionCamTriggered('XF:23ID1-ES{FCCD}',
                              dg1_prefix='XF:23ID1-ES{Dly:1',
                              dg2_prefix='XF:23ID1-ES{Dly:2',
                              mcs_prefix='XF:23ID1-ES{Sclr:1}',
                              name='fccd')
fccd.read_attrs = ['hdf5','mcs.wfrm']
fccd.hdf5.read_attrs = []
#fccd.hdf5._reg = db.reg
configuration_attrs_list = ['cam.acquire_time',
                            'cam.acquire_period',
                            'cam.image_mode',
                            'cam.num_images',
                            'cam.sdk_version',
                            'cam.firmware_version',
                            'cam.overscan_cols',
                            'cam.fcric_gain',
                            'cam.fcric_clamp',
                            'dg1', 'dg2',
                            'dg2.A', 'dg2.B',
                            'dg2.C', 'dg2.D',
                            'dg2.E', 'dg2.F',
                            'dg2.G', 'dg2.H',
                            'dg1.A', 'dg1.B',
                            'dg1.C', 'dg1.D',
                            'dg1.E', 'dg1.F',
                            'dg1.G', 'dg1.H',
                            'fccd1.enable_bgnd',
                            'fccd1.enable_gain',
                            'fccd1.enable_size',
                            'fccd1.rows',
                            'fccd1.row_offset',
                            'fccd1.overscan_cols',
                            ]


roi_params = ['.min_xyz', '.min_xyz.min_y', '.min_xyz.min_x',
              '.size', '.size.y', '.size.x', '.name_']
configuration_attrs_list.extend(['roi' + str(i) + string for i in range(1,5) for string in roi_params])

##TODO make roi config attrs into generic function like _setup_stats so all areadetectors can have roi coordinates
for attr in configuration_attrs_list:
    getattr(fccd, attr).kind='config'
fccd.configuration_attrs.extend(['roi1', 'roi2', 'roi3','roi4'])
_setup_stats(fccd)


configuration_attrs_list = []                        
configuration_attrs_list.extend(['roi' + str(i) + string for i in range(1,5) for string in roi_params])
for attr in configuration_attrs_list:
    getattr(cam_dif_hdf5, attr).kind='config'
cam_dif_hdf5.configuration_attrs.extend(['roi1', 'roi2', 'roi3','roi4'])

for attr in configuration_attrs_list:
    getattr(axis_standard, attr).kind='config'
axis_standard.configuration_attrs.extend(['roi1', 'roi2', 'roi3','roi4'])

for attr in configuration_attrs_list:
    getattr(axis_cont, attr).kind='config'
axis_cont.configuration_attrs.extend(['roi1', 'roi2', 'roi3','roi4'])
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
                                    StandardProsilicaWithHDF5, StandardProsilicaWithTIFF) #TODOpmab - added to try to save (inspired from SIX)

from ..startup import db

def _setup_stats(cam_in):
    for k in (f'stats{j}' for j in range(1, 6)):
        cam_in.read_attrs.append(k)
        getattr(cam_in, k).read_attrs = ['total']

#TODO add to plugin
diag6_pid_threshold = EpicsSignal('XF:23ID1-BI{Diag:6-Cam:1}Stats1:CentroidThreshold',
        name = 'diag6_pid_threshold')

#
# Scalers both MCS and Standard
#

sclr = PrototypeEpicsScaler('XF:23ID1-ES{Sclr:1}', name='sclr')

for sig in sclr.channels.component_names:
    getattr(sclr.channels, sig).name = 'sclr_' + sig.replace('an', '')

mcs = StruckSIS3820MCS('XF:23ID1-ES{Sclr:1}', name='mcs')

#
# Diagnostic Prosilica Cameras
#

diag2 = StandardCam('XF:23ID1-BI{Diag:2-Cam:1}', name='diag2')#TODOpmab optional imagesave w/ stats always
_setup_stats(diag2) #diamond diagnostic

## 20180726 needed to comment due to IOC1 problems
slt1_cam = StandardCam('XF:23ID1-BI{Slt:1-Cam:1}', name='slt1_cam')
_setup_stats(slt1_cam)

diag3 = StandardCam('XF:23ID1-BI{Diag:3-Cam:1}', name='diag3')
_setup_stats(diag3)

diag6 = MonitorStatsCam('XF:23ID1-BI{Diag:6-Cam:1}', name='diag6') #TODO testing
#diag6 = NoStatsCam('XF:23ID1-BI{Diag:6-Cam:1}', name='diag6') #TODO revert above test

## 20180726 needed to comment due to IOC1 problems - probably ok now, but not used.
#cube_beam = StandardCam('XF:23ID1-BI{Diag:5-Cam:1}', name='cube_beam')
#_setup_stats(cube_beam)

dif_beam = StandardCam('XF:23ID1-ES{Dif-Cam:Beam}', name='dif_beam')
dif_beam_hdf5 = StandardProsilicaWithHDF5('XF:23ID1-ES{Dif-Cam:Beam}', name='dif_beam_hdf5') #TODO replace with DSSI project
_setup_stats(dif_beam)
_setup_stats(dif_beam_hdf5)


# Setup on 2018/03/16 for correlating fCCD and sample position - worked 
# DON'T NEED STATS to take pictures of sample/optics
#dif_cam1 = StandardCam('XF:23ID1-ES{Dif-Cam:1}', name='dif_cam1' )
#_setup_stats(dif_cam2) #comment to disable
dif_cam1 = StandardProsilicaWithTIFF('XF:23ID1-ES{Dif-Cam:1}', name='dif_cam1')
dif_cam2 = StandardProsilicaWithTIFF('XF:23ID1-ES{Dif-Cam:2}', name='dif_cam2')
dif_cam3 = StandardProsilicaWithTIFF('XF:23ID1-ES{Dif-Cam:3}', name='dif_cam3')##TODO think how to fix with trans plugin

## 20201219 - Machine studies for source characterization #TODO save also images like real detector
fs_cam = StandardCam('XF:23IDA-BI:1{FS:1-Cam:1}', name='fs_cam') #TODOpmab optional imagesave w/ stats always

#TODOpmab-andi plugin and start IOC
#bs_cam = StandardCam('XF:23ID1-BI{Diag:7-Cam:1}', name='bs_cam') #TODOpmab optional imagesave w/ stats always
#_setup_stats(bs_cam)

### SWITCH AS NEEDED per experiment
### OPT1
#pa_cam = StandardCam('XF:23ID1-BI{Diag:8-Cam:1}', name='pa_cam') #TODOpmab optional imagesave w/ stats always
#_setup_stats(pa_cam)
#pa_cam_hdf5 = StandardProsilicaWithHDF5('XF:23ID1-BI{Diag:8-Cam:1}', name='pa_cam_hdf5') #TODO replace with DSSI project
#_setup_stats(pa_cam)
#_setup_stats(pa_cam_hdf5)
### OPT2
diag6new = MonitorStatsCam('XF:23ID1-BI{Diag:8-Cam:1}', name='diag6new') #TODO testing
##diag6new = NoStatsCam('XF:23ID1-BI{Diag:8-Cam:1}', name='diag6new') #TODO revert above test

#TODOpmab-andi to clean up and add all hinted stats the we normally hint for prosilicas (dif_beam, etc)

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
    getattr(dif_beam_hdf5, attr).kind='config'
dif_beam_hdf5.configuration_attrs.extend(['roi1', 'roi2', 'roi3','roi4'])
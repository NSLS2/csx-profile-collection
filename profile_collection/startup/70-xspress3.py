from nslsii.detectors.xspress3 import (XspressTrigger,
                                       Xspress3Detector,
                                       Xspress3FileStore,
                                       Xspress3Channel,
                                       Xspress3ROI)
from ophyd.areadetector.plugins import PluginBase

class CSXXspress3Detector(XspressTrigger, Xspress3Detector):
    roi_data = Cpt(PluginBase, 'ROIDATA:')
    channel1 = Cpt(Xspress3Channel,
                   'C1_', channel_num=1,
                   read_attrs=['rois'])
    # arrsum = Cpt(Xspress3Detector, 'ARRSUM1:ArrayData', read_attrs=[], configuration_attrs=[])
    # arr1 =Cpt(Xspress3Detector, 'ARR1:ArrayData', read_attrs=[], configuration_attrs=[])

    hdf5 = Cpt(Xspress3FileStore, 'HDF5:',
               write_path_template='/GPFS/xf23id/xf23id1/xspress3_data/%Y/%m/%d/',
               root='/GPFS/xf23id/xf23id1/',
               reg=db.reg)

    def __init__(self, prefix, *, configuration_attrs=None, read_attrs=None,
                 **kwargs):
        if configuration_attrs is None:
            configuration_attrs = ['external_trig', 'total_points',
                                   'spectra_per_point', 'settings',
                                   'rewindable']
        if read_attrs is None:
            read_attrs = ['channel1', 'hdf5']
        super().__init__(prefix, configuration_attrs=configuration_attrs,
                         read_attrs=read_attrs, **kwargs)

    @property
    def hints(self):
        return {'fields': ['xsp3_roi_01','xsp3_roi_02']}


xsp3 = CSXXspress3Detector('XF:23ID1-ES{XP3}:', name='xsp3', roi_sums=True)


for j, r in enumerate(xsp3.channel1.all_rois):
    r.value.name = 'xsp3_roi_{:02d}'.format(j+1)
    r.value_sum.name = 'xsp3_accumulated_roi_{:02d}'.format(j+1)
    r.read_attrs = ['value', 'value_sum']

def setup_rois(channel, roi_width):
    for j, roi in enumerate(channel.all_rois):
        yield from bp.mv(roi.bin_low, j*roi_width)
        yield from bp.sleep(1)
        yield from bp.mv(roi.bin_high, (j+1)*roi_width)
        yield from bp.sleep(1)

def roi_for_elements(channel, list_of_elm):
    for roi, elm in zip(channel.all_rois, list_of_elm):
        low, high = get_range(elm)
        yield from bp.abs_set(roi.bin_low, low, wait=True, timeout=1)
        yield from bp.sleep(.5)
        yield from bp.abs_set(roi.bin_high, high, wait=True, timeout=1)
        yield from bp.sleep(.5)

def get_range(elm):
    elm_map = {'Fe': [70, 72],
               'Ni': [84, 86]}
    return elm_map[elm]

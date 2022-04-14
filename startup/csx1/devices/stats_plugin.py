from ophyd import Device, Component as Cpt
from ophyd.areadetector.base import (ADBase, ADComponent as ADCpt, ad_group,
                                     EpicsSignalWithRBV as SignalWithRBV)
from ophyd.areadetector.plugins import PluginBase
from ophyd.areadetector.trigger_mixins import TriggerBase, ADTriggerStatus
from ophyd.device import DynamicDeviceComponent as DDC, Staged
from ophyd.signal import (Signal, EpicsSignalRO, EpicsSignal)
from ophyd.quadem import QuadEM

import time as ttime


class StatsPluginCSX(PluginBase):
    """This supports changes to time series PV names in AD 3-3

    Due to https://github.com/areaDetector/ADCore/pull/333
    """
    _default_suffix = 'Stats1:'
    _suffix_re = 'Stats\d:'
    _html_docs = ['NDPluginStats.html']
    _plugin_type = 'NDPluginStats'

    _default_configuration_attrs = (PluginBase._default_configuration_attrs + (
        'centroid_threshold', 'compute_centroid', 'compute_histogram',
        'compute_profiles', 'compute_statistics', 'bgd_width',
        'hist_size', 'hist_min', 'hist_max', 'profile_size',
        'profile_cursor')
    )

    bgd_width = ADCpt(SignalWithRBV, 'BgdWidth')
    centroid_threshold = ADCpt(SignalWithRBV, 'CentroidThreshold')

    centroid = DDC(ad_group(EpicsSignalRO,
                            (('x', 'CentroidX_RBV'),
                             ('y', 'CentroidY_RBV'))),
                   doc='The centroid XY',
                   default_read_attrs=('x', 'y'))

    compute_centroid = ADCpt(SignalWithRBV, 'ComputeCentroid', string=True)
    compute_histogram = ADCpt(SignalWithRBV, 'ComputeHistogram', string=True)
    compute_profiles = ADCpt(SignalWithRBV, 'ComputeProfiles', string=True)
    compute_statistics = ADCpt(SignalWithRBV, 'ComputeStatistics', string=True)

    cursor = DDC(ad_group(SignalWithRBV,
                          (('x', 'CursorX'),
                           ('y', 'CursorY'))),
                 doc='The cursor XY',
                 default_read_attrs=('x', 'y'))

    hist_entropy = ADCpt(EpicsSignalRO, 'HistEntropy_RBV')
    hist_max = ADCpt(SignalWithRBV, 'HistMax')
    hist_min = ADCpt(SignalWithRBV, 'HistMin')
    hist_size = ADCpt(SignalWithRBV, 'HistSize')
    histogram = ADCpt(EpicsSignalRO, 'Histogram_RBV')

    max_size = DDC(ad_group(EpicsSignal,
                            (('x', 'MaxSizeX'),
                             ('y', 'MaxSizeY'))),
                   doc='The maximum size in XY',
                   default_read_attrs=('x', 'y'))

    max_value = ADCpt(EpicsSignalRO, 'MaxValue_RBV')
    max_xy = DDC(ad_group(EpicsSignalRO,
                          (('x', 'MaxX_RBV'),
                           ('y', 'MaxY_RBV'))),
                 doc='Maximum in XY',
                 default_read_attrs=('x', 'y'))

    mean_value = ADCpt(EpicsSignalRO, 'MeanValue_RBV')
    min_value = ADCpt(EpicsSignalRO, 'MinValue_RBV')

    min_xy = DDC(ad_group(EpicsSignalRO,
                          (('x', 'MinX_RBV'),
                           ('y', 'MinY_RBV'))),
                 doc='Minimum in XY',
                 default_read_attrs=('x', 'y'))

    net = ADCpt(EpicsSignalRO, 'Net_RBV')
    profile_average = DDC(ad_group(EpicsSignalRO,
                                   (('x', 'ProfileAverageX_RBV'),
                                    ('y', 'ProfileAverageY_RBV'))),
                          doc='Profile average in XY',
                          default_read_attrs=('x', 'y'))

    profile_centroid = DDC(ad_group(EpicsSignalRO,
                                    (('x', 'ProfileCentroidX_RBV'),
                                     ('y', 'ProfileCentroidY_RBV'))),
                           doc='Profile centroid in XY',
                           default_read_attrs=('x', 'y'))

    profile_cursor = DDC(ad_group(EpicsSignalRO,
                                  (('x', 'ProfileCursorX_RBV'),
                                   ('y', 'ProfileCursorY_RBV'))),
                         doc='Profile cursor in XY',
                         default_read_attrs=('x', 'y'))

    profile_size = DDC(ad_group(EpicsSignalRO,
                                (('x', 'ProfileSizeX_RBV'),
                                 ('y', 'ProfileSizeY_RBV'))),
                       doc='Profile size in XY',
                       default_read_attrs=('x', 'y'))

    profile_threshold = DDC(ad_group(EpicsSignalRO,
                                     (('x', 'ProfileThresholdX_RBV'),
                                      ('y', 'ProfileThresholdY_RBV'))),
                            doc='Profile threshold in XY',
                            default_read_attrs=('x', 'y'))

    set_xhopr = ADCpt(EpicsSignal, 'SetXHOPR')
    set_yhopr = ADCpt(EpicsSignal, 'SetYHOPR')
    sigma_xy = ADCpt(EpicsSignalRO, 'SigmaXY_RBV')
    sigma_x = ADCpt(EpicsSignalRO, 'SigmaX_RBV')
    sigma_y = ADCpt(EpicsSignalRO, 'SigmaY_RBV')
    sigma = ADCpt(EpicsSignalRO, 'Sigma_RBV')
    # ts_acquiring = ADCpt(EpicsSignal, 'TS:TSAcquiring')

    ts_centroid = DDC(ad_group(EpicsSignal,
                               (('x', 'TS:TSCentroidX'),
                                ('y', 'TS:TSCentroidY'))),
                      doc='Time series centroid in XY',
                      default_read_attrs=('x', 'y'))

    # ts_control = ADCpt(EpicsSignal, 'TS:TSControl', string=True)
    # ts_current_point = ADCpt(EpicsSignal, 'TS:TSCurrentPoint')
    ts_max_value = ADCpt(EpicsSignal, 'TS:TSMaxValue')

    ts_max = DDC(ad_group(EpicsSignal,
                          (('x', 'TS:TSMaxX'),
                           ('y', 'TS:TSMaxY'))),
                 doc='Time series maximum in XY',
                 default_read_attrs=('x', 'y'))

    ts_mean_value = ADCpt(EpicsSignal, 'TS:TSMeanValue')
    ts_min_value = ADCpt(EpicsSignal, 'TS:TSMinValue')

    ts_min = DDC(ad_group(EpicsSignal,
                          (('x', 'TS:TSMinX'),
                           ('y', 'TS:TSMinY'))),
                 doc='Time series minimum in XY',
                 default_read_attrs=('x', 'y'))

    ts_net = ADCpt(EpicsSignal, 'TS:TSNet')
    # ts_num_points = ADCpt(EpicsSignal, 'TS:TSNumPoints')
    ts_read = ADCpt(EpicsSignal, 'TS:TSRead')
    ts_sigma = ADCpt(EpicsSignal, 'TS:TSSigma')
    ts_sigma_x = ADCpt(EpicsSignal, 'TS:TSSigmaX')
    ts_sigma_xy = ADCpt(EpicsSignal, 'TS:TSSigmaXY')
    ts_sigma_y = ADCpt(EpicsSignal, 'TS:TSSigmaY')
    ts_total = ADCpt(EpicsSignal, 'TS:TSTotal')
    total = ADCpt(EpicsSignalRO, 'Total_RBV')

#!/usr/bin/env python3
import re
from collections import defaultdict

from caproto import (ChannelChar, ChannelData, ChannelDouble, ChannelEnum,
                     ChannelInteger, ChannelString)
from caproto.server import ioc_arg_parser, run, PVGroup

PLUGIN_TYPE_PVS = [
    (re.compile('image\\d:'), 'NDPluginStdArrays'),
    (re.compile('Stats\\d:'), 'NDPluginStats'),
    (re.compile('CC\\d:'), 'NDPluginColorConvert'),
    (re.compile('Proc\\d:'), 'NDPluginProcess'),
    (re.compile('Over\\d:'), 'NDPluginOverlay'),
    (re.compile('ROI\\d:'), 'NDPluginROI'),
    (re.compile('Trans\\d:'), 'NDPluginTransform'),
    (re.compile('netCDF\\d:'), 'NDFileNetCDF'),
    (re.compile('TIFF\\d:'), 'NDFileTIFF'),
    (re.compile('JPEG\\d:'), 'NDFileJPEG'),
    (re.compile('Nexus\\d:'), 'NDPluginNexus'),
    (re.compile('HDF\\d:'), 'NDFileHDF5'),
    (re.compile('Magick\\d:'), 'NDFileMagick'),
    (re.compile('TIFF\\d:'), 'NDFileTIFF'),
    (re.compile('HDF\\d:'), 'NDFileHDF5'),
    (re.compile('Current\\d:'), 'NDPluginStats'),
    (re.compile('SumAll'), 'NDPluginStats'),
]

exclusions: set[str] = {
    "XF:23ID1-ES{AXIS}cam1:GainMode",
    "XF:23ID1-ES{AXIS}cam1:BinMode",
    "XF:23ID1-ES{AXIS}cam1:TEC",
}


class ReallyDefaultDict(defaultdict):
    def __contains__(self, key):
        if "XF:23ID1-ES{AXIS}" in key and not any(key.startswith(prefix) for prefix in exclusions):
            return False
        return True

    def __missing__(self, key):
        if "XF:23ID1-ES{AXIS}" in key and not any(key.startswith(prefix) for prefix in exclusions):
            return None
        if (key.endswith('-SP') or key.endswith('-I') or
                key.endswith('-RB') or key.endswith('-Cmd')):
            key, *_ = key.rpartition('-')
            return self[key]
        if key.endswith('_RBV') or key.endswith(':RBV'):
            return self[key[:-4]]
        ret = self[key] = self.default_factory(key)
        return ret

class CSX_IOC(PVGroup):

    def __init__(self, *args, **kwargs):
        # Initialize the explicit pv properties
        super().__init__(prefix="", *args, **kwargs)
        # Overwrite the pvdb with the blackhole, while keeping the explicit pv properties
        self.old_pvdb = self.pvdb.copy()
        self.pvdb = ReallyDefaultDict(self.fabricate_channel)


    def fabricate_channel(self, key):
        # If the channel already exists from initialization, return it
        if key in self.old_pvdb:
            return self.old_pvdb[key]
        # Otherwise, fabricate new channels
        if 'PluginType' in key:
            for pattern, val in PLUGIN_TYPE_PVS:
                if pattern.search(key):
                    return ChannelString(value=val)
        elif 'ArrayPort' in key:
            return ChannelString(value="cam1")
        elif 'PortName' in key:
            # Extract port name from key format: <prefix><port-name>:PortName
            # Use regex to find the last component before :PortName
            match = re.search(r"[^:}_\-]+(?=:PortName)", key)
            if match:
                port_name = match.group(0)
                return ChannelString(value=port_name)
            # Fallback if regex doesn't match
            return ChannelString(value=key)
        elif 'name' in key.lower():
            return ChannelString(value=key)
        elif 'EnableCallbacks' in key:
            return ChannelEnum(value=0, enum_strings=['Disabled', 'Enabled'])
        elif 'BlockingCallbacks' in key:
            return ChannelEnum(value=0, enum_strings=['No', 'Yes'])
        elif 'Auto' in key:
            return ChannelEnum(value=0, enum_strings=['No', 'Yes'])
        elif 'ImageMode' in key:
            return ChannelEnum(value=0, enum_strings=['Single', 'Multiple', 'Continuous'])
        elif 'WriteMode' in key:
            return ChannelEnum(value=0, enum_strings=['Single', 'Capture', 'Stream'])
        elif 'ArraySize' in key:
            return ChannelData(value=10)
        elif 'TriggerMode' in key:
            return ChannelEnum(value=0, enum_strings=['Internal', 'External'])
        elif 'FileWriteMode' in key:
            return ChannelEnum(value=0, enum_strings=['Single'])
        elif 'FilePathExists' in key:
            return ChannelData(value=1)
        elif 'WaitForPlugins' in key:
            return ChannelEnum(value=0, enum_strings=['No', 'Yes'])
        elif ('file' in key.lower() and 'number' not in key.lower() and
            'mode' not in key.lower()):
            return ChannelChar(value='a' * 250)
        elif ('filenumber' in key.lower()):
            return ChannelInteger(value=0)
        elif 'Compression' in key:
            return ChannelEnum(value=0, enum_strings=['None', 'N-bit', 'szip', 'zlib', 'blosc'])
        return ChannelDouble(value=0.0)


def main():
    print('''
*** WARNING ***
This script spawns an EPICS IOC which responds to ALL caget, caput, camonitor
requests.  As this is effectively a PV black hole, it may affect the
performance and functionality of other IOCs on your network.

The script ignores the --interfaces command line argument, always
binding only to 127.0.0.1, superseding the usual default (0.0.0.0) and any
user-provided value.
*** WARNING ***

Press return if you have acknowledged the above, or Ctrl-C to quit.''')

    try:
        input()
    except KeyboardInterrupt:
        print()
        return
    print('''

                         PV blackhole started

''')
    _, run_options = ioc_arg_parser(
        default_prefix='',
        desc="PV black hole")
    run_options['interfaces'] = ['127.0.0.1']
    run(CSX_IOC().pvdb,
        **run_options)


if __name__ == '__main__':
    main()

"""Module for workinbg with sCMOS camera.
 This is a device currently under development by the supplier,
 and does not currently (as of 2022-08-02) use a proper AreaDetector driver"""

from ophyd import Device, EpicsSignal, EpicsSignalRO
from ophyd import Component as Cpt
from epics import caget, caput
from collections import OrderedDict
import time
import bluesky.plan_stubs as bps
from pathlib import Path


class SCMOSToDisk(Device):
    """
    sCMOS implementation that writes to the file system.

    at present this ignores the Querier and Readback PV, as they
    were not behaving as expected with caget/put.
    (Setting the PV would change the PV, but not the RVB PV).
    """

    output_path = Cpt(EpicsSignal, ":DVFilename")
    acquire_time = Cpt(EpicsSignal, ":AcquireTime")
    capture_mode = Cpt(EpicsSignal, ":CaptureMode")
    _trigger = Cpt(EpicsSignal, ":StartCapture")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.acq_count = 0
        self.path_dir = Path("/nsls2/data/csx/legacy/sCMOS/")
        self.path_prefix = ""

    def trigger(self):
        yield from bps.mv(self.capture_mode, 2)
        path = self.path_dir / f"{self.path_prefix}{self.acq_count}_background.raw"
        yield from bps.mv(self.output_path, str(path))
        yield from bps.mv(self._trigger, 0)

        yield from bps.mv(self.capture_mode, 2)
        path = self.path_dir / f"{self.path_prefix}{self.acq_count}_foreground.raw"
        yield from bps.mv(self.output_path, str(path))
        yield from bps.mv(self._trigger, 1)

        self.acq_count += 1

    def read(self):
        path = caget(self.output_path.cls, as_string=True)
        return OrderedDict(value=path, timestamp=time.time())

    def describe(self):
        return {}

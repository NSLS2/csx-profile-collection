"""Module for workinbg with sCMOS camera.
 This is a device currently under development by the supplier,
 and does not currently (as of 2022-08-02) use a proper AreaDetector driver"""
import threading
from ophyd import Device, EpicsSignal, Kind, DeviceStatus
from ophyd import Component as Cpt
import time
from pathlib import Path


class SCMOSToDisk(Device):
    """
    sCMOS implementation that writes to the file system.

    at present this ignores the Querier and Readback PV, as they
    were not behaving as expected with caget/put.
    (Setting the PV would change the PV, but not the RVB PV).
    """

    output_path = Cpt(EpicsSignal, ":DVFilename")
    acquire_time = Cpt(EpicsSignal, ":AcquireTime", kind=Kind.config)
    capture_mode = Cpt(EpicsSignal, ":CaptureMode", kind=Kind.config)
    _trigger_pv = Cpt(EpicsSignal, ":StartCapture", kind=Kind.omitted)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.acq_count = 0
        self.path_dir = Path("/nsls2/data/csx/legacy/sCMOS/")
        self.path_prefix = ""
        self.is_dark = False

    def _acquire(self, status: DeviceStatus):
        """
        This is how we would do this if there was a way to see were done with acquisition...
        Maffettone didn't see anything in the manual.
        acq_signal.put(1, wait=False, callback=done_acquisition)
        """
        if self.is_dark:
            self.capture_mode.put(2)
            path = self.path_dir / f"{self.path_prefix}_{self.acq_count}_background.raw"
            self.output_path.put(str(path))
            self._trigger_pv.put(0)
        else:
            self.capture_mode.put(2)
            path = self.path_dir / f"{self.path_prefix}_{self.acq_count}_foreground.raw"
            self.output_path.put(str(path))
            self._trigger_pv.put(1)
        time.sleep(self.acquire_time.get())
        status.set_finished()

    def trigger(self):
        status = DeviceStatus(self)
        threading.Thread(target=self._acquire, args=(status,), daemon=True).start()
        self.acq_count += 1
        return status

    """Ideally by managing kind effectively, read and describe can be auto assembled."""
    # def read(self):
    #     path = self.output_path.get()
    #     return OrderedDict(value=path, timestamp=time.time())
    #
    # def describe(self):
    #     return {}

###############################################################################
# TODO: remove this block once https://github.com/bluesky/ophyd/pull/959 is
# merged/released.
from datetime import datetime
from ophyd.signal import EpicsSignalBase, EpicsSignal, DEFAULT_CONNECTION_TIMEOUT

def print_now():
    return datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S.%f')

def wait_for_connection_base(self, timeout=DEFAULT_CONNECTION_TIMEOUT):
    '''Wait for the underlying signals to initialize or connect'''
    if timeout is DEFAULT_CONNECTION_TIMEOUT:
        timeout = self.connection_timeout
    # print(f'{print_now()}: waiting for {self.name} to connect within {timeout:.4f} s...')
    start = time.time()
    try:
        self._ensure_connected(self._read_pv, timeout=timeout)
        # print(f'{print_now()}: waited for {self.name} to connect for {time.time() - start:.4f} s.')
    except TimeoutError:
        if self._destroyed:
            raise DestroyedError('Signal has been destroyed')
        raise

def wait_for_connection(self, timeout=DEFAULT_CONNECTION_TIMEOUT):
    '''Wait for the underlying signals to initialize or connect'''
    if timeout is DEFAULT_CONNECTION_TIMEOUT:
        timeout = self.connection_timeout
    # print(f'{print_now()}: waiting for {self.name} to connect within {timeout:.4f} s...')
    start = time.time()
    self._ensure_connected(self._read_pv, self._write_pv, timeout=timeout)
    # print(f'{print_now()}: waited for {self.name} to connect for {time.time() - start:.4f} s.')

EpicsSignalBase.wait_for_connection = wait_for_connection_base
EpicsSignal.wait_for_connection = wait_for_connection
###############################################################################

from ophyd.signal import EpicsSignalBase
# EpicsSignalBase.set_default_timeout(timeout=10, connection_timeout=10)  # old style
EpicsSignalBase.set_defaults(timeout=10, connection_timeout=10)  # new style


import appdirs
import nslsii
from IPython import get_ipython
from bluesky.utils import PersistentDict
from pathlib import Path
from csx1.analysis.callbacks import BECwithTicks

ip = get_ipython()
nslsii.configure_base(ip.user_ns, 'csx', publish_documents_with_kafka=True, bec=False)
nslsii.configure_olog(ip.user_ns)

# Commenting out the DAMA-suggested location
# (~/.local/share/bluesky/runengine-metadata), as the BL staff prefers a
# separate location.
# runengine_metadata_dir = appdirs.user_data_dir(appname="bluesky") / Path("runengine-metadata")
runengine_metadata_dir = os.path.expanduser("/nsls2/data/csx/shared/config/RE-metadata")
RE.md = PersistentDict(runengine_metadata_dir)

bec = BECwithTicks()
peaks = bec.peaks  # just as alias for less typing
RE.subscribe(bec)



from csx1.startup import *

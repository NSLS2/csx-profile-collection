# DAMA hack on 2019-09-30 to avoid timeouts with the mono motor
# components:
# FailedStatus: MoveStatus(done=True, pos=pgm_energy, elapsed=0.4, success=False, settle_time=0.0)
#
# /opt/conda_envs/collection-2019-3.0-csx/lib/python3.7/site-packages/epics/ca.py:1459: UserWarning: ca.get('XF:23ID1-OP{Mono-Ax:GrtP}Mtr.TDIR') timed out after 1.00 seconds.
#   warnings.warn(msg % (name(chid), timeout))
# /opt/conda_envs/collection-2019-3.0-csx/lib/python3.7/site-packages/epics/ca.py:1459: UserWarning: ca.get('XF:23ID1-OP{Mono-Ax:GrtP}Mtr.HLS') timed out after 1.00 seconds.
#   warnings.warn(msg % (name(chid), timeout))
##
## import ophyd
## import functools
## ophyd.signal.EpicsSignalBase.wait_for_connection = functools.partialmethod(ophyd.signal.EpicsSignalBase.wait_for_connection, timeout=5)
## print(ophyd.signal.EpicsSignalBase.wait_for_connection.__dict__)
## It did not help, commenting it back.

from csx1.startup import *

from IPython import get_ipython
import nslsii

nslsii.configure_olog(get_ipython().user_ns)

from ophyd.signal import EpicsSignalBase, EpicsSignal

EpicsSignalBase.set_defaults(timeout=10, connection_timeout=10)
EpicsSignal.set_defaults(timeout=10, connection_timeout=10)

import os
import appdirs
import nslsii
from IPython import get_ipython
from bluesky.utils import PersistentDict
from pathlib import Path
import time as ttime
from csx1.analysis.callbacks import BECwithTicks
from tiled.client import from_profile
from bluesky_tiled_plugins import TiledWriter

class TiledInserter:
    def insert(self, name, doc):
        ATTEMPTS = 20
        error = None
        for attempt in range(ATTEMPTS):
            try:
                tiled_writing_client_raw.post_document(name, doc)
            except Exception as exc:
                print("Document saving failure:", repr(exc))
                error = exc
            else:
                break
            ttime.sleep(2)
        else:
            # Out of attempts
            raise error

tiled_writing_client = from_profile('nsls2', api_key=os.environ.get('TILED_BLUESKY_WRITING_API_KEY_CSX'))["csx"]
tiled_writing_client_raw = tiled_writing_client["raw"]
tiled_writing_client_sql = tiled_writing_client["migration"]

tiled_inserter = TiledInserter()
tw = TiledWriter(
        tiled_writing_client_sql,
        backup_directory="/tmp/tiled_backup",
        spec_to_mimetype={
            "AD_HDF5": "application/x-hdf5",
            "AD_TIFF": "multipart/related;type=image/tiff",
        })
tiled_reading_client_raw = from_profile("nsls2")["csx"]["raw"]
c = tiled_reading_client_sql = from_profile("nsls2")["csx"]["migration"]

ip = get_ipython()
nslsii.configure_base(
    ip.user_ns,
    'csx',
    publish_documents_with_kafka=False,
    bec=False,
    redis_url="xf23id1-csx-redis1.nsls2.bnl.gov",
    redis_port=6380,
    redis_ssl=True,
)
nslsii.configure_olog(ip.user_ns)

bec = BECwithTicks()
peaks = bec.peaks  # just as alias for less typing
RE.subscribe(bec)
RE.subscribe(tw)


from csx1.startup import *

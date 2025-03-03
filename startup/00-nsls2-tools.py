from ophyd.signal import EpicsSignalBase
EpicsSignalBase.set_defaults(timeout=10, connection_timeout=10)  # new style
# EpicsSignalBase.set_default_timeout(timeout=10, connection_timeout=10)  # old style



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

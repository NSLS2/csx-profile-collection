import os

# Set up a RunEngine and use metadata backed by a sqlite file.
from bluesky import RunEngine

from distutils.version import LooseVersion
import bluesky
if bluesky.__version__ < LooseVersion('1.6'):
    OLD_BLUESKY = True
else:
    OLD_BLUESKY = False

if OLD_BLUESKY:
    from bluesky.utils import get_history
    RE = RunEngine(get_history())
else:
    from bluesky.utils import PersistentDict
    RE = RunEngine({})
    RE.md = PersistentDict(os.path.expanduser('~/.config/bluesky/persistent-dict'))


# Set up a Broker.
from databroker import Broker
db = Broker.named('csx')

# Subscribe metadatastore to documents.
# If this is removed, data is not saved to metadatastore.
RE.subscribe(db.insert)

# Set up SupplementalData.
from bluesky import SupplementalData
sd = SupplementalData()
RE.preprocessors.append(sd)

# Add a progress bar.
from bluesky.utils import ProgressBarManager
pbar_manager = ProgressBarManager()
RE.waiting_hook = pbar_manager

# Register bluesky IPython magics.
from bluesky.magics import BlueskyMagics
get_ipython().register_magics(BlueskyMagics)

# Set up the BestEffortCallback.
from bluesky.callbacks.best_effort import BestEffortCallback
bec = BestEffortCallback()
RE.subscribe(bec)
peaks = bec.peaks  # just as alias for less typing
bec.disable_baseline()

# At the end of every run, verify that files were saved and
# print a confirmation message.
from bluesky.callbacks.broker import verify_files_saved
# RE.subscribe(post_run(verify_files_saved), 'stop')

# Import matplotlib and put it in interactive mode.
import matplotlib.pyplot as plt
plt.ion()

# Make plots update live while scans run.
# from bluesky.utils import install_qt_kicker
# install_qt_kicker()

# Optional: set any metadata that rarely changes.
# RE.md['beamline_id'] = 'YOUR_BEAMLINE_HERE'

# convenience imports
from bluesky.callbacks import *
from bluesky.callbacks.broker import *
from bluesky.simulators import *
from bluesky.plans import *
import numpy as np

# Uncomment the following lines to turn on verbose messages for
# debugging.
# import logging
# ophyd.logger.setLevel(logging.DEBUG)
# logging.basicConfig(level=logging.DEBUG)

asc = scan  # alias
rsc = relative_scan # alias

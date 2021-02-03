import os

# Optional: set any metadata that rarely changes.
# RE.md['beamline_id'] = 'YOUR_BEAMLINE_HERE'

# convenience imports
from bluesky.callbacks import *
from bluesky.callbacks.broker import *
from bluesky.simulators import *
from bluesky.plans import *
import numpy as np

asc = scan  # alias
rsc = relative_scan # alias

from IPython import get_ipython

ip = get_ipython()
RE = ip.user_ns['RE']
db = ip.user_ns['db']

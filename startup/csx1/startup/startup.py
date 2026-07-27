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
sd = ip.user_ns['sd']

def proposal_path():
    """
    Return the path to the proposal directory for this beamline.
    """
    return f"/nsls2/data/csx/proposals/{RE.d['cycle']}/{RE.md['data_session']}/"

def asset_path():
    """
    Return the path to the asset directory for this beamline.
    """
    return proposal_path() + "assets/"
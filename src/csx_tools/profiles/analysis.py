"""csx-tools analysis profile: a Tiled client and quick-look helpers.

Start with ``BS_PROFILE=analysis pixi run -e terminal start``. The profile only
reads from Tiled and creates no RunEngine or devices, so it can run beside the
collection session, e.g. while a plan is running there.
"""

from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.core.getipython import get_ipython
from IPython.terminal.interactiveshell import TerminalInteractiveShell
from tiled.client import from_uri

from csx_tools import quicklook

TILED_URI = "https://tiled.nsls2.bnl.gov"
# Catalog of Bluesky runs written by the collection session.
TILED_CATALOG = "csx/migration"

# Tiled prompts for login when the server requires it and no saved login exists.
c = from_uri(TILED_URI)[TILED_CATALOG]

recent_runs = partial(quicklook.recent_runs, c)
plot_scans = partial(quicklook.plot_scans, c)
show_images = partial(quicklook.show_images, c)
what_changed = partial(quicklook.what_changed, c)

ipython = get_ipython()
if ipython is not None and isinstance(ipython, TerminalInteractiveShell):
    plt.ion()  # Enable interactive mode for matplotlib

    # Make tracebacks minimal by default to reduce clutter.
    # Use %tb verbose to view full traceback.
    ipython.run_line_magic("xmode", "minimal")

print(
    f"Tiled catalog {TILED_CATALOG} is `c`; c[-1] is the newest run, c[41233] scan"
    " 41233.\n"
    "Quick looks: recent_runs()  plot_scans(-1, 41233)  show_images(-1, 41233)"
    "  what_changed(41233)\n"
    "Help on each, e.g.: quicklook.show_images?"
)

# csx_tools

Tools for CSX beamline at NSLS-II

## Quick look at recent runs

`csx_tools.quicklook` compares runs saved in Tiled: what was measured recently,
scans overlaid, detector images side by side, and what changed between two runs.
Each function takes the Tiled catalog as its first argument; the CSX IPython
session binds it to `c` with `functools.partial`.

```python
recent_runs()                               # newest runs, oldest first
plot_scans(-1, 41233)                       # overlay the hinted fields of two runs
plot_scans(-1, -2, y="sclr_ch4", norm="sclr_ch3")
show_images(-1, 41233, diff=True)           # images on one colour scale, plus A - B
show_images(-1, subtract=41200, log=True)   # subtract a dark run, log colour scale
what_changed(41233)                         # metadata, settings and baseline that differ
```

Runs are referenced as in the catalog: `-1` is the newest run, `-2` the one
before, positive integers are scan IDs, and strings are (partial) UIDs. A run's
data reach Tiled when it finishes, so an unfinished run cannot be compared yet.
Each kind of plot reuses one window, "Quick look: scans" or "Quick look: images".

The RunEngine session is busy while a plan runs. To look at data meanwhile, open
a second IPython session, e.g. `pixi run -e terminal ipython --matplotlib=qt`,
and bind the functions to a Tiled client:

```python
from functools import partial

from tiled.client import from_profile
from csx_tools import quicklook

c = from_profile("nsls2")["csx"]["migration"]
recent_runs = partial(quicklook.recent_runs, c)
plot_scans = partial(quicklook.plot_scans, c)
show_images = partial(quicklook.show_images, c)
what_changed = partial(quicklook.what_changed, c)
```

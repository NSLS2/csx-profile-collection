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

The RunEngine session is busy while a plan runs. To look at data meanwhile, start
the analysis profile in a second terminal:

```sh
BS_PROFILE=analysis pixi run -e terminal start
```

It logs in to Tiled (prompting only when no saved login exists), binds the
functions above to the catalog `c`, and imports `np`, `pd` and `plt`. It creates
no RunEngine or devices.

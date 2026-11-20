"""Quick-look comparison of recent Bluesky runs stored in Tiled.

These functions answer the questions asked most during beam time: what was
measured recently (`recent_runs`), how scans compare (`plot_scans`), how
detector images compare (`show_images`), and what changed between two runs
(`what_changed`). They only read from Tiled, so they also work in a second
IPython session while the RunEngine session is busy.

Each function takes the catalog of Bluesky runs to read as its first argument,
e.g. the ``c`` client of the CSX IPython session. Bind it with
`functools.partial` for interactive use::

    from functools import partial
    from csx_tools import quicklook

    plot_scans = partial(quicklook.plot_scans, c)
    plot_scans(-1, 41233, norm="sclr_ch3")

Runs are referenced as in the catalog: negative integers count back from the
newest run (``-1`` is the latest), positive integers are scan IDs, and strings
are (partial) UIDs. Run clients may be passed directly.

A run's data may reach Tiled only when the run finishes. Reading data the
server does not have yet raises `KeyError` naming the run as unfinished.

Figures are drawn with pyplot into the reused windows "Quick look: scans" and
"Quick look: images", so repeated calls update one window per kind instead of
opening new ones. They appear immediately when IPython runs with an
interactive backend, e.g. after ``%matplotlib qt``.
"""

import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime
from numbers import Real
from typing import Any, TypeAlias, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bluesky_tiled_plugins import BlueskyRun, CatalogOfBlueskyRuns
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

__all__ = [
    "IMAGES_FIGURE",
    "SCANS_FIGURE",
    "RunRef",
    "plot_scans",
    "recent_runs",
    "show_images",
    "what_changed",
]

RunRef: TypeAlias = int | str | BlueskyRun
"""A run client, scan ID, negative recency index, or (partial) UID."""

SCANS_FIGURE = "Quick look: scans"
IMAGES_FIGURE = "Quick look: images"

# Image data are fetched from Tiled in blocks of at most this many bytes.
_READ_BLOCK_BYTES = 64 * 2**20
# Pixel percentiles that set the colour limits shared by image panels.
_CLIM_PERCENTILES = (0.5, 99.5)
# Colour limits are estimated from at most about this many pixels per image.
_CLIM_SAMPLE_PIXELS = 1_000_000
# Start-document keys that differ between any two runs.
_RUN_IDENTITY_KEYS = frozenset({"uid", "time", "scan_id"})
_SUMMARY_COLUMNS = [
    "scan_id",
    "start",
    "plan",
    "motors",
    "detectors",
    "points",
    "seconds",
    "status",
    "uid",
]


def recent_runs(catalog: CatalogOfBlueskyRuns, n: int = 10) -> pd.DataFrame:
    """Summarise the ``n`` newest runs, oldest first.

    Parameters
    ----------
    catalog
        Catalog to read, already authenticated as needed.
    n
        Number of runs to list.

    Returns
    -------
    pandas.DataFrame
        One row per run indexed by scan ID: start time, plan, scanned motors,
        detectors, recorded primary points, duration in seconds, exit status
        (``"unfinished"`` while no stop document exists), and the first
        characters of the UID.
    """
    rows = [_summary(run.metadata) for run in catalog.values().tail(n)]
    table = pd.DataFrame(rows, columns=_SUMMARY_COLUMNS)
    table["points"] = table["points"].astype("Int64")
    return table.set_index("scan_id")


def plot_scans(
    catalog: CatalogOfBlueskyRuns,
    *runs: RunRef,
    y: str | Sequence[str] | None = None,
    x: str | None = None,
    norm: str | None = None,
    stream: str = "primary",
) -> Figure:
    """Overlay scalar readings of one or more runs.

    Each ``y`` field gets its own panel with one line per run; panels share
    the horizontal axis.

    Parameters
    ----------
    catalog
        Catalog to read, already authenticated as needed.
    *runs
        Runs to overlay; the newest run when none are given.
    y
        Field or fields to plot. Defaults to the hinted fields of the first
        run, the fields the live plots show, other than ``x`` and ``norm``.
    x
        Horizontal-axis field. Defaults to the scanned motor of the first run
        when it scanned one dimension, otherwise the point number ``seq_num``,
        so runs taken at different times line up.
    norm
        Field to divide every ``y`` by, e.g. a monitor scaler channel.
    stream
        Event stream to read.

    Returns
    -------
    matplotlib.figure.Figure
        The "Quick look: scans" figure.
    """
    resolved = [_run(catalog, ref) for ref in runs or (-1,)]
    first = resolved[0]
    x_key = x or _default_x(first.start, stream)
    if isinstance(y, str):
        y_keys = [y]
    elif y is not None:
        y_keys = list(y)
    else:
        y_keys = _hinted_fields(first[stream].metadata, exclude={x_key, norm})
        if not y_keys:
            raise ValueError(
                f"{_label(first)} has no hinted fields in its {stream!r} stream; "
                "choose fields with y="
            )
    keys = [x_key, *y_keys, *([norm] if norm else [])]

    size = (7.0, 1.0 + 2.4 * len(y_keys))
    fig, axes = _subplots(SCANS_FIGURE, (len(y_keys), 1), size, sharey=False)
    axes = axes[:, 0]
    for run in resolved:
        data = _read_fields(run, stream, keys)
        for ax, y_key in zip(axes, y_keys, strict=True):
            values = data[y_key] / data[norm] if norm else data[y_key]
            ax.plot(data[x_key], values, marker="o", markersize=3, label=_label(run))
    for ax, y_key in zip(axes, y_keys, strict=True):
        ax.set_ylabel(f"{y_key} / {norm}" if norm else y_key)
    axes[-1].set_xlabel(x_key)
    axes[0].legend(fontsize="small")
    return fig


def show_images(
    catalog: CatalogOfBlueskyRuns,
    *runs: RunRef,
    det: str | None = None,
    frame: int | None = None,
    subtract: RunRef | None = None,
    diff: bool = False,
    log: bool = False,
    rois: bool = True,
    stream: str = "primary",
) -> Figure:
    """Show detector images of one or more runs side by side.

    All run panels share one colour scale and zoom together. The image of a
    run is the mean of its frames unless ``frame`` selects one point.

    Parameters
    ----------
    catalog
        Catalog to read, already authenticated as needed.
    *runs
        Runs to show; the newest run when none are given.
    det
        Detector name (e.g. ``"axis_standard"``) or image field. Defaults to
        the only image field of the first run.
    frame
        Point (event) index to show instead of the mean over all points;
        negative values count from the end. Several frames recorded at one
        point are averaged.
    subtract
        Run whose mean image, e.g. a dark, is subtracted from every image.
    diff
        Add a panel with the difference of the two given runs, first minus
        second, on a symmetric diverging colour scale.
    log
        Use a logarithmic colour scale; non-positive pixels are left blank.
    rois
        Outline the detector ROIs recorded in each run's configuration.
    stream
        Event stream to read.

    Returns
    -------
    matplotlib.figure.Figure
        The "Quick look: images" figure.
    """
    resolved = [_run(catalog, ref) for ref in runs or (-1,)]
    if diff and len(resolved) != 2:
        raise ValueError(f"diff=True compares two runs, not {len(resolved)}")
    key = _image_key(resolved[0], stream, det)

    background = None
    background_label = ""
    if subtract is not None:
        dark = _run(catalog, subtract)
        background, _ = _mean_image(dark, stream, key, None)
        background_label = f" - {_short(dark)}"

    panels: list[tuple[BlueskyRun, np.ndarray, str]] = []
    for run in resolved:
        image, how = _mean_image(run, stream, key, frame)
        if background is not None:
            image = image - background
        panels.append((run, image, f"{_label(run)}\n{how}{background_label}"))

    n_axes = len(panels) + diff
    ncols = min(n_axes, 3)
    nrows = math.ceil(n_axes / ncols)
    size = (1.0 + 4.0 * ncols, 0.6 + 3.8 * nrows)
    fig, axes = _subplots(IMAGES_FIGURE, (nrows, ncols), size, sharey=True)
    axes = axes.ravel()

    color_norm = _shared_norm([image for _, image, _ in panels], log)
    for ax, (run, image, title) in zip(axes, panels, strict=False):
        ax.imshow(image, norm=color_norm)
        ax.set_title(title, fontsize="small")
        if rois:
            _draw_rois(ax, run[stream].metadata, key)
    shared_scale = ScalarMappable(norm=color_norm)
    fig.colorbar(shared_scale, ax=axes[: len(panels)].tolist(), shrink=0.8)

    if diff:
        (first, first_image, _), (second, second_image, _) = panels
        difference = first_image - second_image
        limit = float(np.nanpercentile(np.abs(difference), _CLIM_PERCENTILES[1]))
        limit = limit or 1.0
        ax = axes[len(panels)]
        artist = ax.imshow(difference, cmap="RdBu_r", vmin=-limit, vmax=limit)
        ax.set_title(f"{_short(first)} - {_short(second)}", fontsize="small")
        fig.colorbar(artist, ax=ax, shrink=0.8)
    for ax in axes[n_axes:]:
        ax.set_visible(False)
    return fig


def what_changed(
    catalog: CatalogOfBlueskyRuns,
    ref: RunRef,
    other: RunRef = -1,
    *,
    rtol: float = 1e-3,
    atol: float = 1e-6,
) -> pd.DataFrame:
    """List what differs between two runs.

    Compares the start-document metadata (apart from UID, time and scan ID),
    the device configuration of the primary stream (e.g. exposure time, ROIs),
    and the baseline readings taken when each run started.

    Parameters
    ----------
    catalog
        Catalog to read, already authenticated as needed.
    ref
        Run to compare from.
    other
        Run to compare to; the newest run by default.
    rtol, atol
        Relative and absolute tolerance within which numbers count as
        unchanged, as in `math.isclose`.

    Returns
    -------
    pandas.DataFrame
        One row per differing field, sorted by source and field: the source
        (``"start"``, ``"config"`` or ``"baseline"``), both values, and for
        numbers the change ``other - ref``. Fields recorded by only one run
        appear with the missing value as ``None``.
    """
    before, after = _run(catalog, ref), _run(catalog, other)
    sources: list[tuple[str, Callable[[BlueskyRun], dict[str, Any]]]] = [
        ("start", _start_values),
        ("config", _config_values),
        ("baseline", _baseline_values),
    ]
    fields, rows = [], []
    for source, read in sources:
        old, new = read(before), read(after)
        for field in sorted(old.keys() | new.keys()):
            a, b = old.get(field), new.get(field)
            if not _same(a, b, rtol, atol):
                fields.append(field)
                rows.append((source, a, b, _change(a, b)))
    return pd.DataFrame(
        rows,
        index=pd.Index(fields, name="field"),
        columns=["source", _short(before), _short(after), "change"],
    )


def _run(catalog: CatalogOfBlueskyRuns, ref: RunRef) -> BlueskyRun:
    if isinstance(ref, BlueskyRun):
        return ref
    return cast(BlueskyRun, catalog[ref])


def _summary(metadata: Mapping[str, Any]) -> dict[str, Any]:
    start = metadata["start"]
    stop = metadata.get("stop")
    return {
        "scan_id": start.get("scan_id"),
        "start": datetime.fromtimestamp(start["time"]).replace(microsecond=0),
        "plan": start.get("plan_name"),
        "motors": ", ".join(start.get("motors", ())),
        "detectors": ", ".join(start.get("detectors", ())),
        "points": stop.get("num_events", {}).get("primary") if stop else None,
        "seconds": round(stop["time"] - start["time"], 1) if stop else None,
        "status": stop["exit_status"] if stop else "unfinished",
        "uid": start["uid"][:8],
    }


def _short(run: BlueskyRun) -> str:
    return f"#{run.start.get('scan_id')}"


def _label(run: BlueskyRun) -> str:
    start = cast(Mapping[str, Any], run.start)
    started = datetime.fromtimestamp(start["time"]).strftime("%H:%M")
    return f"{_short(run)} {start.get('plan_name', '')} {started}"


def _subplots(
    label: str, grid: tuple[int, int], size: tuple[float, float], sharey: bool
) -> tuple[Figure, np.ndarray]:
    """Return the reused pyplot figure ``label`` with a new grid of axes.

    The window keeps a size chosen by the user while the grid stays the same.
    """
    if plt.fignum_exists(label):
        fig = plt.figure(label)
        previous = {
            spec.get_geometry()[:2]
            for spec in (ax.get_subplotspec() for ax in fig.axes)
            if spec is not None
        }
        fig.clear()
        if previous != {grid}:
            fig.set_size_inches(size)
    else:
        fig = plt.figure(label, figsize=size)
    fig.set_layout_engine("constrained")
    axes = fig.subplots(*grid, sharex=True, sharey=sharey, squeeze=False)
    return fig, axes


def _default_x(start: Mapping[str, Any], stream: str) -> str:
    """Return the scanned field of a one-dimensional scan, else ``seq_num``."""
    dimensions = start.get("hints", {}).get("dimensions")
    if dimensions is None:
        motors = start.get("motors")
        dimensions = [(motors, "primary")] if motors else []
    if len(dimensions) == 1:
        ((fields, dimension_stream),) = dimensions
        if dimension_stream == stream and fields and fields[0] != "time":
            return fields[0]
    return "seq_num"


def _hinted_fields(
    stream_metadata: Mapping[str, Any], exclude: set[str | None]
) -> list[str]:
    hints = stream_metadata.get("hints", {})
    fields = (field for hint in hints.values() for field in hint.get("fields", ()))
    return [field for field in fields if field not in exclude]


def _require(run: BlueskyRun, stream: str, keys: Iterable[str]) -> Any:
    """Return the stream client of ``run`` after checking it holds ``keys``.

    Tiled reads of fields it has no data for yet return empty results, so this
    raises instead.
    """
    client = run[stream]
    missing = sorted(set(keys).difference(client))
    if missing:
        message = f"{_label(run)} has no {', '.join(missing)} in its {stream!r} stream"
        if run.metadata.get("stop") is None:
            message += "; the run is unfinished and its data may not be in Tiled yet"
        raise KeyError(message)
    return client


def _read_fields(run: BlueskyRun, stream: str, keys: list[str]) -> dict[str, Any]:
    dataset = _require(run, stream, keys).read(variables=keys)
    return {key: dataset[key].values for key in keys}


def _image_key(run: BlueskyRun, stream: str, det: str | None) -> str:
    """Return the image field of ``run`` matching detector or field ``det``."""
    data_keys = run[stream].metadata["data_keys"]
    candidates = [
        key
        for key, data_key in data_keys.items()
        if len(data_key.get("shape") or ()) >= 2
        and (det is None or det in (key, data_key.get("object_name")))
    ]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        source = "" if det is None else f" from {det!r}"
        raise KeyError(f"{_label(run)} has no image{source} in its {stream!r} stream")
    raise ValueError(
        f"{_label(run)} has several images; choose one with det=: "
        + ", ".join(candidates)
    )


def _mean_image(
    run: BlueskyRun, stream: str, key: str, frame: int | None
) -> tuple[np.ndarray, str]:
    """Return the 2-D mean image of one point or of all points, and a caption.

    Images are stored as ``(points, ..., y, x)``; every axis before the last
    two is averaged.
    """
    array = _require(run, stream, [key])[key]
    if frame is not None:
        block = np.asarray(array[frame], dtype=np.float64)
        image = block.reshape(-1, *block.shape[-2:]).mean(axis=0)
        return image.astype(np.float32), f"frame {frame}"

    points = array.shape[0]
    frames = points * math.prod(array.shape[1:-2])
    bytes_per_point = math.prod(array.shape[1:]) * array.dtype.itemsize
    step = max(1, _READ_BLOCK_BYTES // bytes_per_point)
    total = np.zeros(array.shape[-2:], dtype=np.float64)
    for start in range(0, points, step):
        block = np.asarray(array[start : start + step], dtype=np.float64)
        total += block.reshape(-1, *block.shape[-2:]).sum(axis=0)
    return (total / frames).astype(np.float32), f"mean of {frames} frames"


def _shared_norm(images: list[np.ndarray], log: bool) -> Normalize:
    """Return colour limits covering the central pixel range of all images."""
    samples = []
    for image in images:
        stride = max(1, math.isqrt(image.size // _CLIM_SAMPLE_PIXELS))
        sample = image[::stride, ::stride]
        keep = np.isfinite(sample)
        if log:
            keep &= sample > 0
        samples.append(sample[keep])
    pixels = np.concatenate(samples)
    if pixels.size == 0:
        return LogNorm() if log else Normalize()
    low, high = np.percentile(pixels, _CLIM_PERCENTILES)
    return (LogNorm if log else Normalize)(float(low), float(high))


def _draw_rois(ax: Axes, stream_metadata: Mapping[str, Any], key: str) -> None:
    """Outline the areaDetector ROIs recorded in the image device configuration."""
    device = stream_metadata["data_keys"][key].get("object_name")
    if device is None:
        return
    config = stream_metadata.get("configuration", {}).get(device, {}).get("data", {})
    pattern = re.compile(rf"{re.escape(device)}_roi(\d+)_size_x")
    for field in config:
        match = pattern.fullmatch(field)
        if match is None:
            continue
        prefix = f"{device}_roi{match[1]}"
        width, height = config[field], config.get(f"{prefix}_size_y", 0)
        if width <= 0 or height <= 0:
            continue
        # Pixel centres sit on integer coordinates in imshow.
        corner = (
            config.get(f"{prefix}_min_xyz_min_x", 0) - 0.5,
            config.get(f"{prefix}_min_xyz_min_y", 0) - 0.5,
        )
        ax.add_patch(Rectangle(corner, width, height, fill=False, edgecolor="red"))
        name = config.get(f"{prefix}_name_") or f"roi{match[1]}"
        ax.annotate(
            name,
            corner,
            xytext=(2, -2),
            textcoords="offset points",
            va="top",
            color="red",
            fontsize="x-small",
        )


def _start_values(run: BlueskyRun) -> dict[str, Any]:
    return {k: v for k, v in run.start.items() if k not in _RUN_IDENTITY_KEYS}


def _config_values(run: BlueskyRun) -> dict[str, Any]:
    configuration = run["primary"].metadata.get("configuration", {})
    return {
        field: value
        for device in configuration.values()
        for field, value in device.get("data", {}).items()
    }


def _baseline_values(run: BlueskyRun) -> dict[str, Any]:
    """Return the baseline readings taken when ``run`` started."""
    keys = list(run["baseline"].metadata["data_keys"])
    dataset = _require(run, "baseline", keys).read(variables=keys)
    return {key: dataset[key].values[0].tolist() for key in keys}


def _same(a: Any, b: Any, rtol: float, atol: float) -> bool:
    if isinstance(a, Real) and isinstance(b, Real):
        a, b = float(a), float(b)
        if math.isnan(a) and math.isnan(b):
            return True
        return math.isclose(a, b, rel_tol=rtol, abs_tol=atol)
    return a == b


def _change(a: Any, b: Any) -> float | None:
    if isinstance(a, Real) and isinstance(b, Real):
        return float(b) - float(a)
    return None

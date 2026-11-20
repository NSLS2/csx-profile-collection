import time

import h5py
import matplotlib
import numpy as np
import pandas as pd
import pytest
from bluesky_tiled_plugins import TiledWriter
from event_model import compose_run, compose_stream_resource
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from tiled.client import from_uri
from tiled.server import SimpleTiledServer
from tiled.structures.core import Spec

from csx_tools.quicklook import plot_scans, recent_runs, show_images, what_changed

matplotlib.use("Agg")

IMAGE_SHAPE = (4, 5)


def _number(source="sim"):
    return {"dtype": "number", "shape": [], "source": source}


def write_run(
    writer,
    assets,
    *,
    scan_id,
    images,
    theta=None,
    sclr_ch2=None,
    acquire_time=0.1,
    energy=700.0,
    temperature=20.0,
    sample="A",
    finished=True,
):
    """Write a CSX-like run: scalers, AXIS images with ROIs, and a baseline.

    ``images`` has shape (points, frames per point, y, x); ``theta`` makes the
    run a one-dimensional scan, otherwise it is a count.
    """
    points = images.shape[0]
    sclr_ch2 = np.arange(1.0, points + 1) if sclr_ch2 is None else sclr_ch2
    metadata = {
        "scan_id": scan_id,
        "plan_name": "count" if theta is None else "scan",
        "detectors": ["sclr", "axis_standard"],
        "num_points": points,
        "sample": sample,
    }
    if theta is not None:
        metadata["motors"] = ["tardis_theta"]
        metadata["hints"] = {"dimensions": [[["tardis_theta"], "primary"]]}
    run = compose_run(metadata=metadata)
    writer("start", run.start_doc)

    config = {
        "axis_standard_cam_acquire_time": acquire_time,
        "axis_standard_roi1_min_xyz_min_x": 1,
        "axis_standard_roi1_min_xyz_min_y": 2,
        "axis_standard_roi1_size_x": 2,
        "axis_standard_roi1_size_y": 1,
        "axis_standard_roi1_name_": "spot",
        "axis_standard_roi2_min_xyz_min_x": 0,
        "axis_standard_roi2_min_xyz_min_y": 0,
        "axis_standard_roi2_size_x": 0,
        "axis_standard_roi2_size_y": 0,
        "axis_standard_roi2_name_": "",
    }
    scalars = ["sclr_ch2", "sclr_ch3", "axis_standard_stats1_total"]
    data_keys = {key: _number() for key in scalars}
    data_keys["axis_standard_image"] = {
        "dtype": "array",
        "dtype_numpy": "<u2",
        "shape": list(images.shape[1:]),
        "source": "sim",
        "external": "STREAM:",
    }
    object_keys = {
        "sclr": ["sclr_ch2", "sclr_ch3"],
        "axis_standard": ["axis_standard_image", "axis_standard_stats1_total"],
    }
    hints = {
        "sclr": {"fields": ["sclr_ch2", "sclr_ch3"]},
        "axis_standard": {"fields": ["axis_standard_stats1_total"]},
    }
    if theta is not None:
        scalars.append("tardis_theta")
        data_keys["tardis_theta"] = _number()
        object_keys["tardis"] = ["tardis_theta"]
        hints["tardis"] = {"fields": ["tardis_theta"]}
    primary = run.compose_descriptor(
        name="primary",
        data_keys=data_keys,
        object_keys=object_keys,
        hints=hints,
        configuration={
            "axis_standard": {
                "data": config,
                "timestamps": dict.fromkeys(config, 0.0),
                "data_keys": {
                    key: {
                        "dtype": "string" if isinstance(value, str) else "number",
                        "shape": [],
                        "source": "sim",
                    }
                    for key, value in config.items()
                },
            }
        },
    )
    writer("descriptor", primary.descriptor_doc)

    baseline_keys = {
        "pgm_energy": _number(),
        "stemp_temperature": _number(),
        "inout_status": {"dtype": "string", "shape": [], "source": "sim"},
    }
    baseline = run.compose_descriptor(
        name="baseline",
        data_keys=baseline_keys,
        object_keys={
            "pgm": ["pgm_energy"],
            "stemp": ["stemp_temperature"],
            "inout": ["inout_status"],
        },
    )
    writer("descriptor", baseline.descriptor_doc)

    def write_baseline(seq_num):
        reading = {
            "pgm_energy": energy + seq_num - 1,
            "stemp_temperature": temperature,
            "inout_status": "Out",
        }
        timestamps = dict.fromkeys(baseline_keys, 0.0)
        event = baseline.compose_event(
            data=reading, timestamps=timestamps, seq_num=seq_num
        )
        writer("event", event)

    write_baseline(1)
    path = assets / f"axis_{scan_id}.h5"
    with h5py.File(path, "w") as f:
        f["/entry/data/data"] = images.astype("<u2")
    resource = compose_stream_resource(
        start=run.start_doc,
        mimetype="application/x-hdf5",
        uri=f"file://localhost{path}",
        data_key="axis_standard_image",
        parameters={"dataset": "/entry/data/data"},
    )
    writer("stream_resource", resource.stream_resource_doc)
    for i in range(points):
        readings = {
            "sclr_ch2": float(sclr_ch2[i]),
            "sclr_ch3": 2.0,
            "axis_standard_stats1_total": float(images[i].sum()),
        }
        if theta is not None:
            readings["tardis_theta"] = float(theta[i])
        writer(
            "event",
            primary.compose_event(
                data=readings,
                timestamps=dict.fromkeys(readings, time.time()),
                seq_num=i + 1,
            ),
        )
        writer(
            "stream_datum",
            resource.compose_stream_datum(
                indices={"start": i, "stop": i + 1},
                seq_nums={"start": i + 1, "stop": i + 2},
                descriptor=primary.descriptor_doc,
            ),
        )
    if finished:
        write_baseline(2)
        writer("stop", run.compose_stop())


@pytest.fixture(scope="module")
def catalog(tmp_path_factory):
    assets = tmp_path_factory.mktemp("assets")
    with SimpleTiledServer(readable_storage=[assets]) as server:
        root = from_uri(server.uri)
        with root.context:
            root.create_container(
                "migration", specs=[Spec("CatalogOfBlueskyRuns", version="3.0")]
            )
            catalog = root["migration"]
            writer = TiledWriter(catalog)
            theta = [10.0, 11.0, 12.0]
            point = np.arange(3).reshape(3, 1, 1, 1)
            # Two frames per point: point i holds 10 * i and 10 * i + 1.
            frame = np.arange(2).reshape(1, 2, 1, 1)
            two_frames = 10 * point + frame + np.zeros((3, 2, *IMAGE_SHAPE))
            write_run(writer, assets, scan_id=101, theta=theta, images=two_frames)
            write_run(
                writer,
                assets,
                scan_id=102,
                theta=theta,
                sclr_ch2=np.array([2.0, 4.0, 6.0]),
                images=100 * (point + 1) + np.zeros((3, 1, *IMAGE_SHAPE)),
                acquire_time=0.5,
                energy=710.0,
                temperature=20.00001,
                sample="B",
            )
            dark = np.full((2, 1, *IMAGE_SHAPE), 5)
            write_run(writer, assets, scan_id=103, images=dark)
            unfinished = np.ones((2, 1, *IMAGE_SHAPE))
            write_run(writer, assets, scan_id=104, images=unfinished, finished=False)
            yield catalog
    plt.close("all")


def pixels(ax):
    """Return the image data drawn in ``ax``."""
    return np.asarray(ax.images[0].get_array())


def test_recent_runs_lists_newest_runs_oldest_first(catalog):
    recent = recent_runs(catalog, 3)

    assert list(recent.index) == [102, 103, 104]
    assert list(recent["plan"]) == ["scan", "count", "count"]
    assert list(recent["motors"]) == ["tardis_theta", "", ""]
    assert list(recent["status"]) == ["success", "success", "unfinished"]
    assert recent.loc[102, "points"] == 3
    assert pd.isna(recent.loc[104, "points"])


def test_plot_scans_overlays_hinted_fields_against_scanned_motor(catalog):
    fig = plot_scans(catalog, 101, 102)

    panels = [ax for ax in fig.axes if ax.get_ylabel()]
    assert [ax.get_ylabel() for ax in panels] == [
        "sclr_ch2",
        "sclr_ch3",
        "axis_standard_stats1_total",
    ]
    first, second = panels[0].get_lines()
    np.testing.assert_array_equal(first.get_xdata(), [10.0, 11.0, 12.0])
    np.testing.assert_array_equal(first.get_ydata(), [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(second.get_ydata(), [2.0, 4.0, 6.0])
    assert str(first.get_label()).startswith("#101")
    assert str(second.get_label()).startswith("#102")
    assert panels[-1].get_xlabel() == "tardis_theta"


def test_plot_scans_of_count_uses_point_number_and_normalizes(catalog):
    fig = plot_scans(catalog, 103, y="sclr_ch2", norm="sclr_ch3")

    (ax,) = fig.axes
    (line,) = ax.get_lines()
    assert ax.get_xlabel() == "seq_num"
    np.testing.assert_array_equal(line.get_xdata(), [1, 2])
    np.testing.assert_array_equal(line.get_ydata(), [0.5, 1.0])


def test_unfinished_run_data_reports_unfinished(catalog):
    with pytest.raises(KeyError, match="unfinished"):
        plot_scans(catalog, -1, y="sclr_ch2")


def test_show_images_average_frames_subtract_background_and_diff(catalog):
    fig = show_images(catalog, 102, 101, subtract=103, diff=True)

    latest, earlier, difference = (ax for ax in fig.axes if ax.images)
    # 102: points of 100, 200, 300; 101: frames of 0, 1, 10, 11, 20, 21; dark: 5
    np.testing.assert_allclose(pixels(latest), 195.0)
    np.testing.assert_allclose(pixels(earlier), 5.5)
    np.testing.assert_allclose(pixels(difference), 189.5)
    assert latest.images[0].get_clim() == earlier.images[0].get_clim()
    (roi,) = (p for p in latest.patches if isinstance(p, Rectangle))
    assert roi.get_xy() == (0.5, 1.5)
    assert (roi.get_width(), roi.get_height()) == (2, 1)


def test_show_images_frame_selects_one_point(catalog):
    fig = show_images(catalog, 101, frame=-1, rois=False)

    (panel,) = (ax for ax in fig.axes if ax.images)
    np.testing.assert_allclose(pixels(panel), 20.5)
    assert not panel.patches


def test_what_changed_reports_differences_beyond_tolerance(catalog):
    changes = what_changed(catalog, 101, 102)

    assert set(changes.index) == {
        "sample",
        "axis_standard_cam_acquire_time",
        "pgm_energy",
    }
    assert changes.loc["pgm_energy", "source"] == "baseline"
    assert changes.loc["pgm_energy", "change"] == pytest.approx(10.0)
    assert changes.loc["axis_standard_cam_acquire_time", "#102"] == 0.5
    assert changes.loc["sample", "#101"] == "A"

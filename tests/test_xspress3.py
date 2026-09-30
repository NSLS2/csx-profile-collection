import asyncio
import runpy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from bluesky import RunEngine
from bluesky.run_engine import call_in_bluesky_event_loop
from nslsii.ophyd_async.devices import Xspress3Detector
from ophyd_async.core import (
    DetectorTrigger,
    StaticFilenameProvider,
    StaticPathProvider,
    TriggerInfo,
    callback_on_mock_put,
    set_mock_value,
)
from ophyd_async.epics.adcore import ADBaseColorMode, ADBaseDataType, ADState

SCRIPT_PATH = Path(__file__).parents[1] / "startup" / "70-xspress3.py.notnow"


def load_script(tmp_path: Path):
    path_provider = StaticPathProvider(
        StaticFilenameProvider("xspress3"),
        tmp_path,
    )
    re = SimpleNamespace(
        md={"cycle": "2099-1", "data_session": "pass-000000"}
    )
    with patch(
        "nslsii.ophyd_async.NSLS2PathProvider",
        return_value=path_provider,
    ):
        return runpy.run_path(
            str(SCRIPT_PATH),
            init_globals={"RE": re},
        )


async def connect_and_seed(detector) -> None:
    await detector.connect(mock=True)
    for signal, value in (
        (detector.driver.array_size_x, 4096),
        (detector.driver.array_size_y, 1),
        (detector.driver.array_size_z, 0),
        (detector.driver.data_type, ADBaseDataType.UINT32),
        (detector.driver.color_mode, ADBaseColorMode.MONO),
        (detector.driver.detector_state, ADState.IDLE),
        (detector.hdf.file_path_exists, True),
        (detector.hdf.num_frames_chunks, 1),
        (detector.hdf.num_captured, 0),
    ):
        set_mock_value(signal, value)


async def read_roi_state(channel):
    state = []
    for roi in channel.rois.values():
        state.append(
            (
                await roi.min_x.get_value(),
                await roi.size_x.get_value(),
                await roi.enabled.get_value(),
            )
        )
    return state


@pytest.mark.asyncio
async def test_script_has_shared_detector_channels_rois_suffixes_and_hints(tmp_path):
    detector = load_script(tmp_path)["xsp3"]
    assert type(detector) is Xspress3Detector
    await detector.connect(mock=True)

    assert tuple(detector.channels) == (1,)
    assert tuple(detector.channels[1].rois) == tuple(range(1, 17))
    assert detector.driver.acquire.source == (
        "mock+ca://XF:23ID1-ES{XP3}:det1:Acquire_RBV"
    )
    assert detector.hdf.capture.source == (
        "mock+ca://XF:23ID1-ES{XP3}:HDF1:Capture_RBV"
    )
    assert detector.hints == {
        "fields": ["xsp3-channel1-roi1", "xsp3-channel1-roi2"]
    }


@pytest.mark.asyncio
async def test_internal_trigger_emits_only_standard_spectrum_and_roi_streams(tmp_path):
    detector = load_script(tmp_path)["xsp3"]
    await connect_and_seed(detector)
    expected_keys = {
        "xsp3",
        "xsp3-channel1",
        *(f"xsp3-channel1-roi{number}" for number in range(1, 17)),
    }

    await detector.stage()
    try:
        await detector.prepare(
            TriggerInfo(
                trigger=DetectorTrigger.INTERNAL,
                number_of_events=1,
            )
        )
        assert set(await detector.describe()) == expected_keys

        def finish_acquisition(value):
            if value:
                loop = asyncio.get_running_loop()

                def finish():
                    set_mock_value(detector.hdf.num_captured, 1)
                    set_mock_value(detector.driver.detector_state, ADState.IDLE)
                    set_mock_value(detector.driver.acquire, False)

                loop.call_soon(finish)

        callback_on_mock_put(detector.driver.acquire, finish_acquisition)
        await detector.trigger()

        documents = [document async for document in detector.collect_asset_docs()]
        resources = [
            document
            for name, document in documents
            if name == "stream_resource"
        ]
        datums = [
            document for name, document in documents if name == "stream_datum"
        ]

        assert len(expected_keys) == 18
        assert len(resources) == 18
        assert {resource["data_key"] for resource in resources} == expected_keys
        assert len(datums) == 18
        assert all(
            datum["indices"] == {"start": 0, "stop": 1} for datum in datums
        )
    finally:
        await detector.unstage()

    assert await detector.driver.acquire.get_value() is False
    assert await detector.hdf.capture.get_value() is False


def test_roi_helpers_preserve_enabled_state_and_half_open_ranges(tmp_path):
    script = load_script(tmp_path)
    detector = script["xsp3"]
    setup_rois = script["setup_rois"]
    roi_for_elements = script["roi_for_elements"]
    event_loop = asyncio.new_event_loop()
    run_engine = RunEngine({}, loop=event_loop)
    try:
        call_in_bluesky_event_loop(detector.connect(mock=True))
        channel = detector.channels[1]
        enabled_states = [number % 2 == 0 for number in channel.rois]
        for roi, enabled in zip(
            channel.rois.values(), enabled_states, strict=True
        ):
            set_mock_value(roi.enabled, enabled)

        run_engine(setup_rois(channel, 10))
        expected = [
            (index * 10, 10, enabled)
            for index, enabled in enumerate(enabled_states)
        ]
        assert call_in_bluesky_event_loop(read_roi_state(channel)) == expected

        run_engine(roi_for_elements(channel, ["Fe", "Ni"]))
        expected[:2] = [
            (70, 2, enabled_states[0]),
            (84, 2, enabled_states[1]),
        ]
        assert call_in_bluesky_event_loop(read_roi_state(channel)) == expected
    finally:
        event_loop.call_soon_threadsafe(event_loop.stop)
        run_engine._th.join()
        event_loop.close()

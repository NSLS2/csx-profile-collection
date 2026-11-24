import importlib
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest
from ophyd.sim import make_fake_device

STARTUP_DIR = Path(__file__).parents[1] / "startup"


@pytest.fixture
def areadetector():
    """Import the CSX area detector classes without building the beamline.

    ``csx1.startup`` creates every device (and needs hkl); these classes only
    use its ``asset_path``.
    """
    startup = types.SimpleNamespace(asset_path=lambda: "/tmp/assets/")
    with (
        patch.dict(sys.modules, {"csx1.startup": startup}),
        patch.object(sys, "path", [str(STARTUP_DIR), *sys.path]),
    ):
        yield importlib.import_module("csx1.devices.areadetector")


@pytest.mark.parametrize("class_name", ["StandardAxisCam", "ContinuousAxisCam"])
def test_axis_records_the_transform_types_of_its_rois_and_display(
    areadetector, class_name
):
    detector = make_fake_device(getattr(areadetector, class_name))(
        "XF:23ID1-ES{AXIS}", name="axis"
    )
    # Plugin configuration includes that of its asyn source, so wire the ports
    # as the default plugin graph does.
    detector.cam.port_name.sim_put("AXIS1")
    graph = detector.default_plugin_graph
    for plugin in graph:
        if "port_name" in plugin.component_names:
            plugin.port_name.sim_put(plugin.attr_name.upper())
    for plugin, source in graph.items():
        plugin.nd_array_port.sim_put(source.port_name.get())
    detector.trans1.transform_type.sim_put("Rot90Mirror")
    detector.trans2.transform_type.sim_put("Rot180")

    configuration = detector.read_configuration()

    transforms = {k: v["value"] for k, v in configuration.items() if "trans" in k}
    assert transforms == {
        "axis_trans1_transform_type": "Rot90Mirror",
        "axis_trans2_transform_type": "Rot180",
    }

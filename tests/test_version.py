from importlib.metadata import version

import csx_tools
from csx_tools._version import __version__


def test_package_imports():
    assert csx_tools.__name__ == "csx_tools"


def test_version_matches_installed_metadata():
    assert __version__ == version("csx-tools")

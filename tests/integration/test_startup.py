# tests/conftest.py
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

from IPython.core.interactiveshell import InteractiveShell
import pytest
import matplotlib
from ophyd.sim import FakeEpicsSignal, FakeEpicsSignalWithRBV, FakeEpicsSignalRO

# This is needed to prevent matplotlib from trying to use the X server
matplotlib.use('Agg')
# CRITICAL: Set this BEFORE importing ophyd or epics
# This prevents EPICS C library from being loaded
os.environ['OPHYD_CONTROL_LAYER'] = 'dummy'


@pytest.fixture(scope='session', autouse=True)
def mock_ophyd_signals():

    with patch("ophyd.signal.EpicsSignal.wait_for_connection", return_value=True), \
         patch("ophyd.signal.EpicsSignalBase.wait_for_connection", return_value=True):
        yield


@pytest.fixture(scope='session')
def mock_nslsii():
    """
    Mock NSLS-II specific configuration functions.
    
    These try to connect to Kafka, Redis, Olog, etc.
    """
    
    def mock_configure_base(ipython_user_ns, beamline_name, **kwargs):
        """Mock nslsii.configure_base - populates namespace like the real one"""
        from bluesky import RunEngine
        from bluesky.callbacks.best_effort import BestEffortCallback
        
        # Create the real objects that nslsii.configure_base would create
        ipython_user_ns['RE'] = RunEngine({})
        ipython_user_ns['db'] = MagicMock()
        ipython_user_ns['sd'] = BestEffortCallback()
        
    with patch('nslsii.configure_base', side_effect=mock_configure_base), \
         patch('nslsii.configure_olog') as mock_olog:
        
        mock_olog.return_value = None
        yield

@pytest.fixture
def startup_dir():
    """Return the startup directory."""
    profile_dir = Path(__file__).parent.parent.parent
    startup_dir = profile_dir / "startup"
    
    sys.path.insert(0, str(startup_dir))
    yield startup_dir
    sys.path.remove(str(startup_dir))


def test_startup(mock_nslsii, startup_dir):
    """Test that IPython startup files load without error."""
    shell = InteractiveShell.instance()
    try:
        for file in sorted(startup_dir.glob("*.py")):
            with open(file, "r") as f:
                code = f.read()
            result = shell.run_cell(code, store_history=True, silent=True)
            result.raise_error()
        
        namespace = shell.user_ns
        assert "RE" in namespace, "RunEngine not found in namespace"
        assert "db" in namespace, "Databroker not found in namespace"
    finally:
        InteractiveShell.clear_instance()
    
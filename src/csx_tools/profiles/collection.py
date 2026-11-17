"""csx-tools bluesky collection profile."""

import argparse
import os
from datetime import datetime

import numpy as np
from bluesky import plan_stubs as bps
from bluesky import plans as bp
from bluesky import preprocessors as bpp
from bluesky.callbacks.best_effort import BestEffortCallback
from bluesky.callbacks.buffer import BufferingWrapper
from bluesky.magics import BlueskyMagics
from bluesky.run_engine import RunEngine, autoawait_in_bluesky_event_loop
from bluesky.utils import ProgressBarManager, ts_msg_hook
from bluesky_tiled_plugins import TiledWriter
from IPython.core.getipython import get_ipython
from IPython.terminal.interactiveshell import TerminalInteractiveShell
from nslsii import configure_bluesky_json_logging
from nslsii.ophyd_async.providers import NSLS2PathProvider
from nslsii.utils import (
    ProposalIDPrompt,
    open_redis_client,
    print_proposal_info,
    print_version_info,
)
from ophyd_async.core import init_devices, print_connection_status
from redis_json_dict.redis_json_dict import RedisJSONDict
from tiled.client import from_uri, simple

import csx_tools

# Have ophyd-async keep detector state constant when not explicitly overridden
os.environ["OPHYD_ASYNC_PRESERVE_DETECTOR_STATE"] = "YES"
TLA = (
    os.environ.get("ENDSTATION_ACRONYM")
    or os.environ.get("BEAMLINE_ACRONYM")
    or "Unknown"
)

# Mechanism for passing arguments to profile.
_parser = argparse.ArgumentParser("Collection profile for csx-tools.")
_parser.add_argument(
    "--mock",
    action="store_true",
    help="Run the profile in mock mode with simulated devices and metadata.",
)

# Determine if we are running in CI/mock mode
MOCK = vars(_parser.parse_args())["mock"]

# Where TiledWriter backs up documents it could not write to Tiled.
if MOCK:
    TILED_BACKUP_ROOT = "/tmp/bluesky-docs"
else:
    TILED_BACKUP_ROOT = f"/nsls2/data/{TLA.lower()}/shared/config/bluesky/tiled_backup"

print_version_info(additional_packages=["csx_tools"])

# Construct a RunEngine with either an in-process metadata dict,
# or a redis-backed metadata store.
# The Redis host comes from $REDIS_HOST, or nslsii derives it from the hostname.
if not MOCK:
    RE = RunEngine(RedisJSONDict(open_redis_client(redis_ssl=True), ""))  # type: ignore
else:
    RE = RunEngine(
        {
            "data_session": "pass-123456",
            "cycle": (
                f"{datetime.today().year}-{(datetime.today().month - 1) // 4 + 1}"
            ),
            "proposal": {
                "title": "Mock proposal",
                "type": "commissioning",
                "pi_name": os.getenv("USER", "Unknown"),
            },
        }
    )

# Display active proposal information in the console.
print_proposal_info(RE.md)

# Add tools package version alongside bluesky/ophyd/ophyd-async etc.
RE.md["versions"]["csx_tools"] = csx_tools.__version__

# Beamline and facility metadata
RE.md["facility"] = "NSLS-II"
RE.md["beamline_id"] = ""  # Set the beamline ID (i.e. 27-ID-1)

# Attempt to extract TLA from environment variables.
RE.md["group"] = TLA.upper()

# Get the IPython shell and set up the custom prompt to show the current proposal ID.
ipython = get_ipython()
if ipython is not None and isinstance(ipython, TerminalInteractiveShell):
    ipython.prompts = ProposalIDPrompt(RE, ipython)

    BlueskyMagics.RE = RE
    ipython.register_magics(BlueskyMagics)
    autoawait_in_bluesky_event_loop()

    import matplotlib.pyplot as plt

    plt.ion()  # Enable interactive mode for matplotlib

    # Make tracebacks minimal by default to reduce clutter.
    # Use %tb verbose to view full traceback.
    ipython.run_line_magic("xmode", "minimal")

# Write log records of bluesky, ophyd, ophyd-async, nslsii, this package, and IPython,
# plus a summary of every document RE emits, as JSON lines for log collectors such as
# Grafana Alloy/Loki. Files go to $BLUESKY_JSON_LOG_DIR (default /var/log/bluesky).
if MOCK:
    os.environ.setdefault("BLUESKY_JSON_LOG_DIR", "/tmp/bluesky-logs")
    os.makedirs(os.environ["BLUESKY_JSON_LOG_DIR"], exist_ok=True)
configure_bluesky_json_logging(
    ipython=ipython,
    RE=RE,
    extra_loggers=["csx_tools"],
)

# Set up the Tiled clients for data acquisition and reading based on the environment.
if not MOCK:
    # Typically $TLA/raw. Replace with beamline-specific catalog.
    daq_tiled_catalog = f"{TLA.lower()}/raw"

    tiled_writing_client = from_uri(
        "https://tiled.nsls2.bnl.gov",
        api_key=os.environ.get(f"TILED_BLUESKY_WRITING_API_KEY_{TLA}", ""),
    )[daq_tiled_catalog]
    if ipython is not None and isinstance(ipython, TerminalInteractiveShell):
        tiled_reading_client = c = from_uri("https://tiled.nsls2.bnl.gov")[
            daq_tiled_catalog
        ]
else:
    tiled_writing_client = tiled_reading_client = c = simple()


# Subscribe the tiled writer to the RunEngine, buffered so slow writes don't block it
tiled_writer = TiledWriter(tiled_writing_client, backup_directory=TILED_BACKUP_ROOT)
RE.subscribe(BufferingWrapper(tiled_writer))

# Set up our default path provider
path_provider = NSLS2PathProvider(RE.md)

# Initialize all async devices within this context manager
with init_devices(mock=MOCK, timeout=2.0, on_connect_hook=print_connection_status):
    pass

# Subscribe the best effort callback
bec = BestEffortCallback()
RE.subscribe(bec)

# Disable bec printout of baseline readings
# since it is kind of noisy
bec.disable_baseline()

# Confiugre the default terminal progress bar manager
RE.waiting_hook = ProgressBarManager()  # type: ignore

# Optionally enable ts_msg_hook for debugging
# RE.msg_hook = ts_msg_hook

# Optionally configure baseline supplemental data to include in the metadata
# of every run. Add baseline=[<device1>, <device2>, ...] to SupplementalData as needed.
sd = bpp.SupplementalData()
RE.preprocessors.append(sd)

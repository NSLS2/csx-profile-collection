from ophyd.signal import EpicsSignalBase, EpicsSignal

EpicsSignalBase.set_defaults(timeout=10, connection_timeout=10)
EpicsSignal.set_defaults(timeout=10, connection_timeout=10)

import os
import copy
import appdirs
import nslsii
from nslsii.sync_experiment import sync_experiment as sync_exp
from IPython import get_ipython
from bluesky.utils import PersistentDict
from pathlib import Path
import time as ttime
from csx1.analysis.callbacks import BECwithTicks
from tiled.client import from_profile
from bluesky_tiled_plugins import TiledWriter
from databroker import Broker
import numpy


def sync_experiment(proposal_number):
    sync_exp(proposal_number, beamline="csx", redis_db=0, redis_ssl=True)


def tiled_login():
    if "tiled_reading_client" not in globals():
        raise RuntimeError(
            "tiled_reading_client is not defined."
            'Please create it by first calling: tiled_reading_client = from_profile("nsls2")["csx"]'
        )
    tiled_reading_client.login()
    tiled_reading_client_raw = tiled_reading_client["raw"]
    c = tiled_reading_client_sql = tiled_reading_client["migration"]


# check the current logged in + active user
def whoami():
    try:
        print(f"\nLogged in to Tiled as: {c.context.whoami()['identities'][0]['id']}\n")
    except TypeError as e:
        print("\nNot authenticated with Tiled! Please login...\n")
    print(f"To login as a different user, call 'tiled_login()'")


# check the currently active proposal
def whichproposal():
    try:
        print(f"\nThe currently active proposal is: {RE.md['data_session']}\n")
    except KeyError as e:
        print("\nNo active proposal! Please activate a proposal...\n")
    print(
        f"To activate a different proposal, use 'sync_experiment(proposal_number_here)'"
    )


def patch_descriptor(doc):
    # This was labeled "integer" but it is actually "string".
    INOUT_KEY = "inout_status"
    if INOUT_KEY in doc["data_keys"]:
        doc["data_keys"][INOUT_KEY]["dtype"] = "string"
    # If this is turned on, we get errors about the number of bytes sent.
    # This probably debuggable in pure Tiled -- something about <i2.
    if "fccd_image" in doc["data_keys"]:
        doc["data_keys"]["fccd_image"]["dtype_str"] = "<i2"
    tardis_keys = [
        "tardis_h",
        "tardis_h_setpoint",
        "tardis_k",
        "tardis_k_setpoint",
        "tardis_l",
        "tardis_l_setpoint",
        "tardis_theta",
        "tardis_theta_user_setpoint",
        "tardis_mu",
        "tardis_chi",
        "tardis_phi",
        "tardis_delta",
        "tardis_delta_user_setpoint",
        "tardis_gamma",
        "tardis_gamma_user_setpoint",
    ]
    for key in tardis_keys:
        if key in doc["data_keys"]:
            doc["data_keys"][key]["dtype_str"] = "<f8"
    if "slt3_x_user_setpoint" in doc["data_keys"]:
        doc["data_keys"]["slt3_x_user_setpoint"]["dtype_str"] = "<f8"
    for i in range(1, 33):
        if f"fccd_mcs_wfrm_wfrm_{i}" in doc["data_keys"]:
            doc["data_keys"][f"fccd_mcs_wfrm_wfrm_{i}"]["dtype_str"] = "<i8"
    if "es_diag1_y_user_setpoint" in doc["data_keys"]:
        doc["data_keys"]["es_diag1_y_user_setpoint"]["dtype_str"] = "<f8"

    # Ensure dtype_str has the proper numpy format (to pass the EventModel validator)
    for key, val in doc["data_keys"].items():
        if "dtype_str" in val:
            val["dtype_str"] = numpy.dtype(val["dtype_str"]).str

    return doc


def patch_resource(doc):

    kwargs = doc.get("resource_kwargs", {})

    # Fix the resource path
    root = doc.get("root", "")
    if not doc["resource_path"].startswith(root):
        doc["resource_path"] = os.path.join(root, doc["resource_path"])
    doc["root"] = ""

    if doc.get("spec") in ["AD_HDF5"]:
        kwargs.update({"dataset": 'entry/instrument/detector/data'})
        kwargs["join_method"] = "stack"
    elif doc.get("spec") in ["AD_TIFF"]:
        kwargs["template"] = "/" + kwargs["template"].lstrip("/")    # Ensure leading slash
        kwargs["join_method"] = "stack"
    elif doc.get("spec") in ["AD_HDF5_DET_TS"]:
        kwargs.update({"dataset": '/entry/instrument/NDAttributes/NDArrayTimeStamp'})
        kwargs["join_method"] = "stack"

    return doc

class TiledInserter:
    name = "csx"

    def insert(self, name, doc):
        tiled_writing_client_raw.post_document(name, doc)

tiled_writing_client = from_profile('nsls2', api_key=os.environ.get('TILED_BLUESKY_WRITING_API_KEY_CSX'))["csx"]
tiled_writing_client.context.http_client.headers['tiled-qos'] = 'acquisition'
tiled_writing_client_raw = tiled_writing_client["raw"]
tiled_writing_client_sql = tiled_writing_client["migration"]

tiled_inserter = TiledInserter()
tw = TiledWriter(
        tiled_writing_client_sql,
        backup_directory="/tmp/tiled_backup",
        patches={"descriptor": patch_descriptor,
                 "resource": patch_resource},
        spec_to_mimetype={
            "AD_HDF5": "application/x-hdf5",
            "AD_HDF5_DET_TS": "application/x-hdf5",
            "AD_TIFF": "multipart/related;type=image/tiff",
        })
tiled_reading_client = from_profile("nsls2")["csx"]
tiled_reading_client.context.http_client.headers['tiled-qos'] = 'acquisition'
tiled_reading_client_raw = tiled_reading_client["raw"]
c = tiled_reading_client_sql = tiled_reading_client["migration"]

ip = get_ipython()
nslsii.configure_base(
    ip.user_ns,
    tiled_inserter,
    publish_documents_with_kafka=True,
    bec=False,
    redis_url="xf23id1-csx-redis1.nsls2.bnl.gov",
    redis_port=6380,
    redis_ssl=True,
)
nslsii.configure_olog(ip.user_ns)
db = Broker(tiled_reading_client_raw) # for legacy support

bec = BECwithTicks()
peaks = bec.peaks  # just as alias for less typing
RE.subscribe(bec)
RE.subscribe(tw)


from csx1.startup import *

print("#" * 50)
whoami()
whichproposal()
print()
print("#" * 50)

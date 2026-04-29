"""Quarry Connectors — implementations of the Connector protocol."""

from quarry_connectors.fof_stack import FOFStackConnector
from quarry_connectors.hdf5 import HDF5Connector
from quarry_connectors.pixc import PIXCConnector
from quarry_connectors.sentinel2 import Sentinel2Connector
from quarry_connectors.slc import SLCConnector

__all__ = [
    "FOFStackConnector",
    "HDF5Connector",
    "PIXCConnector",
    "SLCConnector",
    "Sentinel2Connector",
]

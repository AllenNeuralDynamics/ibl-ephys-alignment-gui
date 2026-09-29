"""IBL Ephys Alignment GUI"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ibl-ephys-alignment-gui")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"

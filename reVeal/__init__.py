"""reVeal"""

from pathlib import Path

import pyproj

from reVeal._version import __version__

# stop to_crs() bugs
pyproj.network.set_network_enabled(active=False)

PACKAGE_DIR = Path(__file__).parent

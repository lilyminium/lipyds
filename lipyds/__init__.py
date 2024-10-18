"""
lipyds
A toolkit for leaflet-based membrane analysis
"""

from importlib.metadata import version

# Add imports here
from .leafletfinder.leafletfinder import LeafletFinder
from .core.topologyattrs import *


__version__ = version("lipyds")

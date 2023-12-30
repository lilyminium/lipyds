"""
lipyds
A toolkit for leaflet-based membrane analysis
"""

# Add imports here
from .leafletfinder.leafletfinder import LeafletFinder
# from .analysis import AreaPerLipid, LipidEnrichment, LipidFlipFlop
# from .analysis import AreaPerLipid, LipidTilt, MembraneThickness, ProjectedDisplacement, Curvature, LipidEnrichment
from .core.topologyattrs import *

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions

from . import _version
__version__ = _version.get_versions()['version']

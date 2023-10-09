import numpy as np

from MDAnalysis.core.topologyattrs import ResidueAttr, ResidueStringAttr


class Leaflet(ResidueAttr):
    """Leaflet assignment for residues."""

    attrname = "leaflets"
    singular = "leaflet"
    dtype = int

    @staticmethod
    def _gen_initial_values(na, nr, ns):
        return np.ones(nr, dtype=int) * -1
    

class LipidHeadgroup(ResidueStringAttr):
    attrname = "lipid_headgroups"
    singular = "lipid_headgroup"
    dtype = object

class LipidTailSaturation(ResidueStringAttr):
    attrname = "lipid_tail_saturations"
    singular = "lipid_tail_saturation"
    dtype = object
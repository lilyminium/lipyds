
__all__ = [
    "Martini_double_membrane",  # for testing leaflet finder more
    "DPPC_vesicle_only", # leaflet finder on vesicle
    "DPPC_vesicle_plus", # leaflet finder on vesicle
    "NEURONAL_GLYT2",  # neuronal membrane, DOI 10.1016/j.bbadva.2021.100010
    "NEURONAL_DDAT",  # neuronal membrane, DOI 10.1016/j.bbadva.2021.100010
    "NEURONAL_HDAT",  # neuronal membrane, DOI 10.1016/j.bbadva.2021.100010
    "NEURONAL_HSERT",  # neuronal membrane, DOI 10.1016/j.bbadva.2021.100010
    "DDAT_POPC_TPR",
    "DDAT_POPC_XTC",
    "SURFACE_POINTS",
]

from pkg_resources import resource_filename

Martini_double_membrane = resource_filename(__name__, 'data/martini_double_bilayer.gro')
DPPC_vesicle_only = resource_filename(__name__, 'data/fatslim_dppc_vesicle.pdb')
DPPC_vesicle_plus = resource_filename(__name__, 'data/fatslim_dppc_vesicle_plus.gro')

NEURONAL_GLYT2 = resource_filename(__name__, 'data/glyt2_neuronal.gro')
NEURONAL_DDAT = resource_filename(__name__, 'data/ddat_neuronal.gro')
NEURONAL_HDAT = resource_filename(__name__, 'data/hdat_neuronal.gro')
NEURONAL_HSERT = resource_filename(__name__, 'data/hsert_neuronal.gro')

DDAT_POPC_TPR = resource_filename(__name__, "data/dDAT_POPC-CHOL_r1_nowater.tpr")
DDAT_POPC_XTC = resource_filename(__name__, "data/dDAT_POPC-CHOL_r1_10ns.xtc")
SURFACE_POINTS = resource_filename(__name__, "data/surface_points.dat")

# This should be the last line: clean up namespace
del resource_filename

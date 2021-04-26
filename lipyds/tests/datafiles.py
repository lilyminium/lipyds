
__all__ = [
    "Martini_double_membrane",  # for testing leaflet finder more
    "DPPC_vesicle_only", # leaflet finder on vesicle
    "DPPC_vesicle_plus", # leaflet finder on vesicle
]

from pkg_resources import resource_filename

Martini_double_membrane = resource_filename(__name__, 'data/martini_double_bilayer.gro')
DPPC_vesicle_only = resource_filename(__name__, 'data/fatslim_dppc_vesicle.pdb')
DPPC_vesicle_plus = resource_filename(__name__, 'data/fatslim_dppc_vesicle_plus.gro')

# This should be the last line: clean up namespace
del resource_filename
